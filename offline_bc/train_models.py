import os
import sys
import json
import argparse
import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_dropout, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched
from optim.misc import build_optimizer

from offline_bc.config.default import get_config
from offline_bc.data.dataset import NavDemoDataset, collate_fn
from offline_bc.data.loader import build_dataloader

from offline_bc.models.onav_vis_models import (
    NavILRecurrentNet, NavILTransformer
)
from offline_bc.models.onav_imap_models import (
    NavImapSingleTransformer
)

dataset_factory = {
    'NavDemoDataset': (NavDemoDataset, collate_fn),
}
model_factory = {
    'NavILRecurrentNet': NavILRecurrentNet,
    'NavILTransformer': NavILTransformer,
    'NavImapSingleTransformer': NavImapSingleTransformer,
}

def main(config):
    config.defrost()
    if isinstance(config.DATASET.num_ft_views, str):
        config.MODEL.num_ft_views = int(config.DATASET.num_ft_views.split('_')[0])
    else:
        config.MODEL.num_ft_views = config.DATASET.num_ft_views
    default_gpu, n_gpu, device = set_cuda(config)
    # config.freeze()

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(config.local_rank != -1)
            )
        )
 
    seed = config.SEED
    if config.local_rank != -1:
        seed += config.rank
    set_random_seed(seed)

    # load data training set
    dataset_class, dataset_collate_fn = dataset_factory[config.DATASET.dataset_class]

    trn_dataset = dataset_class(**config.DATASET)
    trn_data_loader, trn_pre_epoch = build_dataloader(
        trn_dataset, dataset_collate_fn, True, config
    )
    LOGGER.info(f'#num_steps_per_epoch: {len(trn_data_loader)}')
    config.num_train_steps = len(trn_data_loader) * config.num_epochs
    config.freeze()

    if len(config.DATASET.val_scene_ids) > 0:
        val_dataset = dataset_class(**config.DATASET, validation=True)
        val_data_loader, _ = build_dataloader(
            val_dataset, dataset_collate_fn, False, config
        )
    else:
        val_data_loader = None

    # setup loggers
    if default_gpu:
        save_training_meta(config)
        TB_LOGGER.create(os.path.join(config.output_dir, 'logs'))
        pbar = tqdm(total=config.num_train_steps)
        model_saver = ModelSaver(os.path.join(config.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(config.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # Prepare model
    model_class = model_factory[config.MODEL.model_class]
    model = model_class(config.MODEL, device)

    LOGGER.info("Model: nweights %d nparams %d" % (model.num_parameters))
    LOGGER.info("Model: trainable nweights %d nparams %d" % (model.num_trainable_parameters))
    # for k, v in model.named_parameters():
    #     if v.requires_grad:
    #         print(k, v.size())

    if config.resume_file:
        checkpoint = torch.load(config.resume_file, map_location=lambda storage, loc: storage)
        LOGGER.info('resume: #params %d' % (len(checkpoint)))
        model.load_state_dict(checkpoint, strict=True)
        # model.load_state_dict(checkpoint, strict=False)
    
    model.train()
    set_dropout(model, config.MODEL.dropout_rate)
    model = wrap_model(model, device, config.local_rank)
    global_step = 0

    # Prepare optimizer
    optimizer = build_optimizer(model, config)
    if config.resume_optimizer:
        optimizer_state = torch.load(config.resume_optimizer)
        print('load optimizer step: %d, weights: %d' % (
            optimizer_state['step'], len(optimizer_state['optimizer']))
        )
        optimizer.load_state_dict(optimizer_state['optimizer'])
        global_step = optimizer_state['step']

    LOGGER.info(f"***** Running training with {config.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", config.train_batch_size if config.local_rank == -1 else config.train_batch_size * config.world_size)
    LOGGER.info("  Accumulate steps = %d", config.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", config.num_train_steps)

    # to compute training statistics
    meteors = {
        key: RunningMeter(key) for key in ['loss', 'acc']
    }
    
    start_time = time.time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()

    for epoch_id in range(config.num_epochs):
        # In distributed mode, calling the set_epoch() method at the beginning of each epoch
        trn_pre_epoch(epoch_id)

        for step, batch in enumerate(trn_data_loader):
            # forward pass
            loss, logits = model(batch, compute_loss=True)
            if isinstance(loss, dict):
                loss_dict = loss
                for loss_key, loss_value in loss.items():
                    if loss_key == 'overall': continue
                    if loss_key not in meteors:
                        meteors[loss_key] = RunningMeter(loss_key)
                    meteors[loss_key](loss_value.item())
                loss = loss['overall']
            
            # backward pass
            if config.gradient_accumulation_steps > 1: # average loss 
                loss = loss / config.gradient_accumulation_steps
            loss.backward()

            meteors['loss'](loss.item())
            if isinstance(logits, tuple):
                logits, subgoal_logits = logits
                if 'subgoal_action_acc' not in meteors:
                    meteors['subgoal_action_acc'] = RunningMeter('subgoal_action_acc')
                acc = torch.mean(
                    (subgoal_logits[:, :3].max(dim=1)[1].data.cpu() == batch['subgoal_actions'])[batch['subgoal_actions'] != -100].float()
                )
                meteors['subgoal_action_acc'](acc.item())
                if 'subgoal_turn_acc' not in meteors:
                    meteors['subgoal_turn_acc'] = RunningMeter('subgoal_turn_acc')
                acc = torch.mean(
                    (subgoal_logits[:, 3:25].max(dim=1)[1].data.cpu() == batch['subgoal_turn_degrees'])[batch['subgoal_actions'] == 2].float()
                )
                meteors['subgoal_turn_acc'](acc.item())
                if 'subgoal_goto_mae' not in meteors:
                    meteors['subgoal_goto_mae'] = RunningMeter('subgoal_goto_mae')
                meteors['subgoal_goto_mae'](loss_dict['subgoal_goto_loss'].sqrt().item() * config.DATASET.max_goto_dist)
            
            acc = torch.mean(
                (logits.max(dim=-1)[1].data.cpu() == batch['demonstration'])[batch['inflection_weight'] > 0].float()
            )
            meteors['acc'](acc.item())
            TB_LOGGER.add_scalar('step/loss', loss.item(), global_step)
            TB_LOGGER.add_scalar('step/acc', acc.item(), global_step)

            # optimizer update and logging
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, config)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.log_scalar_dict({ll.name: ll.val for ll in meteors.values()})
                TB_LOGGER.step()

                # update model params
                if config.grad_norm != -1:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm
                    )
                    # print(global_step, grad_norm)
                    # for k, v in model.named_parameters():
                    #     if v.grad is not None:
                    #         v = torch.norm(v).data.item()
                    #         print(k, v)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

            if global_step % config.log_steps == 0:
                # monitor training throughput
                LOGGER.info(f'==============Epoch {epoch_id} Step {global_step}===============')
                LOGGER.info(', '.join(['%s:%.4f'%(ll.name, ll.val) for ll in meteors.values()]))
                LOGGER.info('===============================================')

            if global_step % config.valid_steps == 0:
                model_saver.save(model, global_step, optimizer=optimizer)  
                if val_data_loader is not None:
                    LOGGER.info(f'-----Epoch {epoch_id} Step {global_step}: start validation-----')
                    validate(model, val_data_loader, TB_LOGGER)

            if global_step > config.num_train_steps:
                break

        if global_step > config.num_train_steps:
            break

    if global_step % config.valid_steps != 0:
        LOGGER.info(f'==============Epoch {epoch_id} Step {global_step}===============')
        LOGGER.info(', '.join(['%s:%.4f'%(ll.name, ll.val) for ll in meteors.values()]))
        LOGGER.info('===============================================')
        model_saver.save(model, global_step, optimizer=optimizer)
        if val_data_loader is not None:
            LOGGER.info(f'-----Epoch {epoch_id} Step {global_step}: start validation-----')
            validate(model, val_data_loader, TB_LOGGER)
    

@torch.no_grad()
def validate(model, data_loader, TB_LOGGER=None):
    model.eval()

    st = time.time()
    val_loss, n_correct, n_total = 0, 0, 0
    n_stop_correct, n_stop_total = 0, 0
    for i, batch in enumerate(data_loader):
        logits = model(batch, compute_loss=False).data.cpu()
        preds = logits.max(dim=-1)[1]
        labels = batch['demonstration']
        # print(preds)
        # print(labels)
        loss = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction='sum', ignore_index=-100)
        val_loss += loss.item()
        n_correct += (preds == labels)[labels != -100].float().sum().item()
        n_total += (labels != -100).float().sum().item()
        n_stop_correct += (preds == labels)[labels == 0].float().sum().item()
        n_stop_total += (labels == 0).float().sum().item()
    val_loss = sum(all_gather(val_loss))
    n_correct = sum(all_gather(n_correct))
    n_total = sum(all_gather(n_total))
    n_stop_correct = sum(all_gather(n_stop_correct))
    n_stop_total = sum(all_gather(n_stop_total))

    val_loss /= n_total
    acc = n_correct / n_total
    stop_acc = n_stop_correct / n_stop_total
    val_log = {'loss': val_loss, 'acc': acc, 'stop_acc': stop_acc}

    tot_time = time.time() - st
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.4f}, acc: {acc*100:.2f}, stop_acc: {stop_acc*100:.2f}")

    if TB_LOGGER is not None:
        TB_LOGGER.log_scalar_dict(
            {f'valid/{k}': v for k, v in val_log.items()}
        )

    model.train()


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    config = get_config(args.exp_config, args.opts)
    
    if os.path.exists(config.output_dir) and os.listdir(config.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                config.output_dir
            )
        )

    return config

if __name__ == '__main__':
    config = build_args()
    main(config)
