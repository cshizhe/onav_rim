#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import argparse
import random
import time
import numpy as np

import torch
import torch.multiprocessing as mp

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config

def evaluation_worker(config):
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    trainer.eval()

def multiprocess_evaluate(config, num_gpus):
    if config.TASK_CONFIG.DATASET.CONTENT_SCENES[0] == '*':
        content_dir = os.path.join(
            os.path.dirname(config.TASK_CONFIG.DATASET.DATA_PATH.format(split=config.EVAL.SPLIT)),
            'content'
        )
        if os.path.exists(content_dir):
            scenes = [x.split('.')[0] for x in os.listdir(content_dir) if x.endswith('.json.gz')]
        else:
            scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    else:
        scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    assert num_gpus <= len(scenes), '#scenes: %d should >= #process: %d' % (len(scenes), num_gpus)
    scenes.sort()

    os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)

    def even_split_scenes(scenes, nparts):
        splited_scenes = [[] for _ in range(nparts)]
        for i, scene in enumerate(scenes):
            idx = i % nparts
            splited_scenes[idx].append(scene)
        return splited_scenes

    node_rank = int(os.environ.get('SLURM_NODEID', 0))
    num_gpus_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', 1))
    cur_gpu_id = int(os.environ.get('SLURM_LOCALID', 0))
    scenes = even_split_scenes(scenes, num_gpus)[cur_gpu_id + node_rank * num_gpus_per_node]

    num_process_per_gpu = config.NUM_PROCESSES
    splited_scenes = even_split_scenes(scenes, num_process_per_gpu)

    processes = []
    for i in range(num_process_per_gpu):
        i_config = copy.deepcopy(config)
        i_config.defrost()
        i_config.TRAINER_NAME = 'il-trainer'
        i_config.TORCH_GPU_ID = cur_gpu_id
        i_config.SIMULATOR_GPU_ID = cur_gpu_id
        i_config.NUM_PROCESSES = 1
        i_config.TASK_CONFIG.DATASET.CONTENT_SCENES = splited_scenes[i]
        i_config.freeze()
        print('gpu_id %d/%d, proc %d/%d, scenes %d\n%s' %(
            cur_gpu_id+node_rank*num_gpus_per_node+1, num_gpus, i+1, num_process_per_gpu, len(splited_scenes[i]),
            ', '.join(splited_scenes[i])
        ))
        processes.append(
            mp.Process(target=evaluation_worker, args=(i_config, ))
        )
        processes[-1].start()
    
    for proc in processes:
        proc.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
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
    run_exp(**vars(args))


def execute_exp(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    if run_type == "train":
        trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
        assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
        trainer = trainer_init(config)
        trainer.train()

    elif run_type == "eval":
        result_file = os.path.join(config.EVAL_RESULTS_DIR, f'{config.EVAL.SPLIT}_pred_trajectories.jsonl')
        if os.path.exists(result_file):
            print(f'eval result dir {result_file} already exists')
            return
        mp.set_start_method('spawn')
        multiprocess_evaluate(config, int(os.environ.get('SLURM_NTASKS', 1)))

        # trainer.eval()


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    execute_exp(config, run_type)


if __name__ == "__main__":
    main()
