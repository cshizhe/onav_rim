#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import copy
import jsonlines
import collections
from filelock import FileLock

from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.il.env_based.common.rollout_storage import RolloutStorage
from habitat_baselines.il.env_based.algos.agent import ILAgent
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.utils.env_utils import make_env_fn, construct_envs
from habitat_baselines.il.env_based.policy.rednet import load_rednet

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
    quaternion_to_list,
)

try:
    from habitat_baselines.il.common.encoders.semantic_predictor import SegmentationModel
except:
    pass

def write_to_jsonlines(filepath, item):
    with FileLock(filepath + '.lock'):
        with jsonlines.open(filepath, mode='a') as outf:
            outf.write(item)

def convert_offline_model(ckpt_dict, is_map_cmt=False):
    state_dict = {}
    for k, v in ckpt_dict.items():
        if k.split('.')[0] in ['action_distribution', 'fine_goto_action_fc']:
            if is_map_cmt:
                newk = 'model.' + k
            else:
                newk = k.replace('action_distribution', 'model.action_distribution.linear')
        else:
            if k.startswith('vis_pred_layer') or k.startswith('map_pred_layer') or k.startswith('sem_pred_layer'):
                continue
            newk = 'model.net.' + k
        state_dict[newk] = v
    ckpt_dict = {'state_dict': state_dict}
    return ckpt_dict

def process_batch_gps_compass(batch, gpscompass_noise_type):
    if gpscompass_noise_type == 'zero':
        batch['gps'][:] = 0
        batch['compass'][:] = 0
    elif gpscompass_noise_type == 'gaussian':
        device = batch['gps'].device
        batch['gps'] += torch.randn(batch['gps'].size(), device=device)
        batch['compass'] += torch.randn(batch['compass'].size(), device=device)
    return batch

@baseline_registry.register_trainer(name="il-trainer")
class ILEnvTrainer(BaseRLTrainer):
    r"""Trainer class for behavior cloning.
    """
    supported_tasks = ["ObjectNav-v1"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []
        if config is not None:
            logger.info(f"config: {config}")

    def _setup_actor_critic_agent(self, il_cfg: Config, model_config: Config, single_env=False) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        if not single_env:
            observation_space = self.envs.observation_spaces[0]
            action_space = self.envs.action_spaces[0]
            num_envs = self.envs.num_envs
        else:
            observation_space = self.envs.observation_space
            action_space = self.envs.action_space
            num_envs = 1

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.obs_space = observation_space

        model_config.defrost()
        model_config.TORCH_GPU_ID = self.config.TORCH_GPU_ID
        model_config.freeze()

        policy = baseline_registry.get_policy(self.config.IL.POLICY.name)
        self.policy = policy.from_config(
            self.config, observation_space, action_space
        )
        self.policy.to(self.device)

        self.semantic_predictor = None
        if model_config.USE_PRED_SEMANTICS:
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=model_config.SEMANTIC_ENCODER.rednet_ckpt,
                resize=True, # since we train on half-vision
                num_classes=model_config.SEMANTIC_ENCODER.num_classes
            )
            self.semantic_predictor.eval()

        if model_config.USE_PRETRAINED_SEGMENTATION:
            self.segmentation_model = SegmentationModel(self.device)

        self.agent = ILAgent(
            model=self.policy,
            num_envs=num_envs,
            num_mini_batch=il_cfg.num_mini_batch,
            lr=il_cfg.lr,
            eps=il_cfg.eps,
            max_grad_norm=il_cfg.max_grad_norm,
        )

    def _make_results_dir(self, split="val"):
        r"""Makes directory for saving eqa-cnn-pretrain eval results."""
        for s_type in ["rgb", "seg", "depth", "top_down_map"]:
            dir_name = self.config.RESULTS_DIR.format(split=split, type=s_type)
            os.makedirs(dir_name, exist_ok=True)

    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "room_visitation_map", "exploration_metrics"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()

        # fetch actions and environment state from replay buffer
        next_actions = rollouts.get_next_actions()
        actions = next_actions.long().unsqueeze(-1)
        step_data = [a.item() for a in next_actions.long().to(device="cpu")]

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        profiling_wrapper.range_pop()  # compute actions

        outputs = self.envs.step(step_data)
        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        if self.config.MODEL.USE_PRED_SEMANTICS and self.current_update >= self.config.MODEL.SWITCH_TO_PRED_SEMANTICS_UPDATE:
            batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
            # Subtract 1 from class labels for THDA YCB categories
            if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                batch["semantic"] = batch["semantic"] - 1
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward  # type: ignore
        running_episode_stats["count"] += 1 - masks  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v  # type: ignore

        current_episode_reward *= masks

        rollouts.insert(
            batch,
            actions,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()

        total_loss, rnn_hidden_states = self.agent.update(rollouts)

        rollouts.after_update(rnn_hidden_states)

        return (
            time.time() - t_update_model,
            total_loss,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        il_cfg = self.config.IL.BehaviorCloning
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(il_cfg, self.config.MODEL)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorage(
            il_cfg.num_steps,
            self.envs.num_envs,
            self.obs_space,
            self.envs.action_spaces[0],
            self.config.MODEL.STATE_ENCODER.hidden_size,
            self.config.MODEL.STATE_ENCODER.num_recurrent_layers,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])
            # Use first semantic observations from RedNet predictor as well
            if sensor == "semantic" and self.config.MODEL.USE_PRED_SEMANTICS:
                semantic_obs = self.semantic_predictor(batch["rgb"], batch["depth"])
                # Subtract 1 from class labels for THDA YCB categories
                if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                    semantic_obs = semantic_obs - 1
                rollouts.observations[sensor][0].copy_(semantic_obs)

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=il_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),  # type: ignore
        )
        self.possible_actions = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")
                
                self.current_update = update

                if il_cfg.use_linear_lr_decay and update > 0:
                    lr_scheduler.step()  # type: ignore

                if il_cfg.use_linear_clip_decay and update > 0:
                    self.agent.clip_param = il_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                profiling_wrapper.range_push("rollouts loop")
                for _step in range(il_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps
                profiling_wrapper.range_pop()  # rollouts loop

                (
                    delta_pth_time,
                    total_loss
                ) = self._update_agent(il_cfg, rollouts)
                pth_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0) # number of episodes

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, count_steps)

                losses = [total_loss]
                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["action"])},
                    count_steps,
                )

                # log stats
                if update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\tloss: {:.3f}".format(
                            update, count_steps / (time.time() - t_start), total_loss
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

   
                if update == self.config.MODEL.SWITCH_TO_PRED_SEMANTICS_UPDATE - 1:
                    self.save_checkpoint(
                        f"ckpt_gt_best.{count_checkpoints}.pth",
                        dict(step=count_steps),
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _eval_checkpoint(
        self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        torch.set_grad_enabled(False)

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        if self.config.EVAL_CKPT_FROM_OFFLINEBC:
            ckpt_dict = convert_offline_model(ckpt_dict)

        self._make_results_dir(self.config.EVAL.SPLIT)

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        il_cfg = config.IL.BehaviorCloning

        config.defrost()
        config.NUM_PROCESSES = 1 # one episode at a time
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.DATASET.TYPE = "ObjectNav-v1"
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 500
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")   # Just for self-test
        config.freeze()

        use_subgoal = getattr(config.MODEL, 'use_subgoal', False)
        episode_iterative = getattr(config, 'EPISODE_ITERATIVE', False)
        single_env = use_subgoal or episode_iterative

        if config.TORCH_GPU_ID == 0:
            logger.info(f"env config: {config}")
        if use_subgoal or episode_iterative:
            self.envs = make_env_fn(
                config, get_env_class(config.ENV_NAME)
            )
        else:
            self.envs = construct_envs(
                config, get_env_class(config.ENV_NAME), 
                auto_reset_done=False if config.MODEL.model_class != 'ObjectNavRNN' else True
            )
        
        self._setup_actor_critic_agent(il_cfg, config.MODEL, single_env=single_env)

        self.agent.load_state_dict(ckpt_dict["state_dict"], strict=True)    
        self.policy = self.agent.model
        self.policy.eval()

        number_of_eval_episodes = config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            if single_env:
                number_of_eval_episodes = self.envs.number_of_episodes
            else:
                number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            if single_env:
                total_num_eps = self.envs.number_of_episodes
            else:
                total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pred_outfile = os.path.join(config.EVAL_RESULTS_DIR, '%s_pred_trajectories.jsonl'%(config.EVAL.SPLIT))
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        
        model_class = config.MODEL.model_class
        if model_class == 'ObjectNavTransformer':
            if episode_iterative:
                stats_episodes = self._eval_checkpoint_transformer_iterative_setting(
                    config, number_of_eval_episodes, pred_outfile, writer, checkpoint_index
                )
            else:
                stats_episodes = self._eval_checkpoint_transformer(
                    config, number_of_eval_episodes, pred_outfile, writer, checkpoint_index
                )
        elif model_class == 'ObjectNavImapSingleTransformer':
            stats_episodes = self._eval_checkpoint_recursive_transformer(
                 config, number_of_eval_episodes, pred_outfile, writer, checkpoint_index
            )
        elif model_class == 'ObjectNavRNN':
            stats_episodes = self._eval_checkpoint_rnn(
                config, number_of_eval_episodes, pred_outfile, writer, checkpoint_index
            )
        else:
            raise NotImplementedError('unsupported model class: %s' % str(model_class))


        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k not in ["reward", "pred_reward"]}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()

    def _eval_checkpoint_rnn(
        self, config, number_of_eval_episodes, pred_outfile, writer, checkpoint_index
    ):
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        test_recurrent_hidden_states = torch.zeros(
            config.MODEL.STATE_ENCODER.num_recurrent_layers,
            config.NUM_PROCESSES,
            config.MODEL.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        ) - 1
        not_done_masks = torch.zeros(
            config.NUM_PROCESSES, 1, device=self.device
        )
        current_episode_steps = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        pred_trajectories = [
            {'actions': [], 'infos': []} for _ in range(config.NUM_PROCESSES)
        ]
        rgb_frames = [
            [] for _ in range(config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        action_masks = torch.zeros(self.envs.num_envs, 6).bool().to(self.device)
        last_compass = [obs['compass'] for obs in observations]
        last_gps = [obs['gps'] for obs in observations]
        is_collides = [False for _ in observations]

        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
            for i in range(config.NUM_PROCESSES):
                frame = observations_to_image(
                    {"rgb": batch["rgb"][i]}, {}
                )
                rgb_frames[i].append(frame)

        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                if self.semantic_predictor is not None:
                    batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
                    if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                        batch["semantic"] = batch["semantic"] - 1
                if (not is_collides[0]) or (config.MODEL.enc_collide_steps):
                    (
                        logits,
                        test_recurrent_hidden_states,
                    ) = self.policy(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                    )
                current_episode_steps += 1

                logits = logits.masked_fill(action_masks[:logits.size(0)].to(logits.device), -float('inf'))
                actions = torch.argmax(logits, dim=1)
                prev_actions.copy_(actions.unsqueeze(1))  # type: ignore

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            #print('action', step_data)
            #print('info', infos)
            for ib, aidx in enumerate(step_data):
                pred_trajectories[ib]['actions'].append(aidx)
                pred_trajectories[ib]['infos'].append(infos[ib])

                # heuristic rule: collide
                if all(np.isclose(last_compass[ib], observations[ib]['compass'])) and \
                   all(np.isclose(last_gps[ib], observations[ib]['gps'])) and aidx in [1, 2, 3]:
                    is_collides[ib] = True
                    action_masks[ib, aidx] = True
                else:
                    is_collides[ib] = False
                    action_masks[ib, :] = False
            last_compass = [obs['compass'] for obs in observations]
            last_gps = [obs['gps'] for obs in observations]
                
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            prev_actions.masked_fill_(not_done_masks.bool().logical_not(), -1) # the init action is -1

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    current_episode_steps[i] = 0

                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    pred_trajectories[i].update({'episode_id': current_episodes[i].episode_id})
                    write_to_jsonlines(pred_outfile, pred_trajectories[i])
                    pred_trajectories[i] = {'actions': [], 'infos': []}

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []
                        frame = observations_to_image(
                            {"rgb": batch["rgb"][i]}, infos[i]
                        )
                        rgb_frames[i].append(frame)

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {"rgb": batch["rgb"][i]}, infos[i]
                    )
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        return stats_episodes
     
    def _eval_checkpoint_transformer(
        self, config, number_of_eval_episodes, 
        pred_outfile, writer, checkpoint_index
    ): 
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        gpscompass_noise_type = getattr(self.config.MODEL, 'gpscompass_noise_type', None)
        print(gpscompass_noise_type)
        for _ in tqdm.trange(number_of_eval_episodes):
            pred_trajectories = {
                'actions': [], 'infos': []
            }
            rgb_frames = []
            
            observations = self.envs.reset()
            current_episodes = self.envs.current_episodes()

            batch = batch_obs(observations, device=self.device)
            batch = process_batch_gps_compass(batch, gpscompass_noise_type)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            if self.semantic_predictor is not None:
                batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
                if config.MODEL.SEMANTIC_ENCODER.is_thda:
                    batch["semantic"] = batch["semantic"] - 1

            prev_actions = torch.zeros(
                self.envs.num_envs, 1, device=self.device, dtype=torch.long
            ) - 1
            # initialize at the begning of each episode
            history_embeds = None
            last_gps, last_compass = observations[0]['gps'], observations[0]['compass']
            action_masks = torch.zeros(self.envs.num_envs, 6).bool().to(self.device)
            current_episode_reward = 0

            done = False
            nav_step = 0
            is_collide = False
            while not done:
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(
                        {"rgb": batch["rgb"][0]}, {}
                    )
                    rgb_frames.append(frame)

                if config.MODEL.enc_collide_steps or (not is_collide):
                    logits, history_embeds = self.policy(batch, history_embeds, prev_actions, nav_step)
                # print('step', nav_step, history_embeds.size())
                logits = logits.masked_fill(action_masks, -float('inf'))
                # print(nav_step, history_embeds.size(), action_masks, is_collide)

                actions = torch.argmax(logits, dim=1)
                step_data = [a.item() for a in actions.to(device="cpu")]

                outputs = self.envs.step(step_data)
                observations, rewards_l, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]
                done = dones[0]
                current_episode_reward += rewards_l[0]
                
                # if (all(observations[0]['gps'] == last_gps)) and \
                #    (all(observations[0]['compass'] == last_compass)) and \
                #    (step_data[0] in [1, 2, 3]):
                if (all(np.isclose(observations[0]['gps'], last_gps))) and \
                    (all(np.isclose(observations[0]['compass'], last_compass))) and \
                    (step_data[0] in [1, 2, 3]):
                # if infos[0]['collisions']['is_collision']:
                    is_collide = True
                else:
                    is_collide = False
                # print(nav_step, is_collide, infos[0]['collisions'])
                # print(last_gps, last_compass)
                # print(observations[0]['gps'], observations[0]['compass'])
                last_gps = observations[0]['gps']
                last_compass = observations[0]['compass']

                # action_masks = torch.zeros(len(step_data), 6).bool()
                if is_collide:
                    action_masks[0, step_data[0]] = True
                else:
                    action_masks[0, :] = False
                pred_trajectories['actions'].append(step_data[0])
                pred_trajectories['infos'].append(infos[0])

                if config.MODEL.enc_collide_steps or (not is_collide):
                    batch = batch_obs(observations, device=self.device)
                    batch = process_batch_gps_compass(batch, gpscompass_noise_type)
                    batch = apply_obs_transforms_batch(batch, self.obs_transforms)
                    if self.semantic_predictor is not None:
                        batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
                        if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                            batch["semantic"] = batch["semantic"] - 1

                    nav_step += 1
                    prev_actions.copy_(actions.unsqueeze(1))  # type: ignore
                
            # print(current_episodes[0].object_category)
            # print(pred_trajectories['actions'])
            # print(pred_trajectories['infos'][0])
            # print(infos[0])
            # print()
            episode_stats = {}
            episode_stats["reward"] = current_episode_reward
            episode_stats.update(
                self._extract_scalars_from_info(infos[0])
            )
            stats_episodes[
                (
                    current_episodes[0].scene_id,
                    current_episodes[0].episode_id,
                )
            ] = episode_stats

            pred_trajectories.update({'episode_id': current_episodes[0].episode_id})
            write_to_jsonlines(pred_outfile, pred_trajectories)
            if len(self.config.VIDEO_OPTION) > 0:
                generate_video(
                    video_option=self.config.VIDEO_OPTION,
                    video_dir=self.config.VIDEO_DIR,
                    images=rgb_frames,
                    episode_id=current_episodes[0].episode_id,
                    checkpoint_idx=checkpoint_index,
                    metrics=self._extract_scalars_from_info(infos[0]),
                    tb_writer=writer,
                )

        return stats_episodes

    def _eval_checkpoint_recursive_transformer(
        self, config, number_of_eval_episodes, pred_outfile, writer, checkpoint_index
    ):
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        model_config = self.config.MODEL
        gpscompass_noise_type = getattr(self.config.MODEL, 'gpscompass_noise_type', None)
        for _ in tqdm.trange(number_of_eval_episodes):
            pred_trajectories = {
                'actions': [], 'infos': []
            }
            rgb_frames = []
            
            observations = self.envs.reset()
            current_episodes = self.envs.current_episodes()

            batch = batch_obs(observations, device=self.device)
            batch = process_batch_gps_compass(batch, gpscompass_noise_type)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            if self.semantic_predictor is not None:
                batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
                if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                    batch["semantic"] = batch["semantic"] - 1

            if model_config.USE_PRETRAINED_SEGMENTATION:
                batch['obj_areas'], batch['obj_dists'] = [], []
                for ib in range(len(batch['rgb'])):
                    obj_area, obj_dist = self.segmentation_model.predict_target(
                        batch['rgb'][ib], batch['depth'][ib], 
                        batch['objectgoal'][ib]
                    )
                    batch['obj_areas'].append(obj_area)
                    batch['obj_dists'].append(obj_dist)

            recursive_states = None
            if config.INIT_MAP_EMBED_DIR is not None:
                init_map_embeds = np.load(os.path.join(config.INIT_MAP_EMBED_DIR, '%s.npy'%current_episodes[0].episode_id))
                recursive_states = (torch.from_numpy(init_map_embeds).to(self.device), None)
            prev_actions = torch.zeros(
                self.envs.num_envs, 1, device=self.device, dtype=torch.long
            ) - 1
            # initialize at the begning of each episode
            last_gps, last_compass = observations[0]['gps'], observations[0]['compass']
            action_masks = torch.zeros(self.envs.num_envs, 6).bool().to(self.device)
            current_episode_reward = 0

            done = False
            is_collide = False
            nav_step = 0
            while not done:
                if len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(
                        {"rgb": batch["rgb"][0]}, {}
                    )
                    rgb_frames.append(frame)

                if config.MODEL.enc_collide_steps or (not is_collide):
                    logits, recursive_states = self.policy(batch, recursive_states, prev_actions, nav_step)
                logits = logits.masked_fill(action_masks, -float('inf'))

                actions = torch.argmax(logits, dim=1)
                step_data = [a.item() for a in actions.to(device="cpu")]

                if model_config.USE_PRETRAINED_SEGMENTATION:
                    step_data = [0 if batch['obj_areas'][ib] > 0.1 and batch['obj_dists'][ib] < 0.9 \
                                 else a for ib, a in enumerate(step_data)]

                outputs = self.envs.step(step_data)
                observations, rewards_l, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]
                done = dones[0]
                current_episode_reward += rewards_l[0]

                if model_config.USE_PRETRAINED_SEGMENTATION:
                    infos[0]['obj_area'] = batch['obj_areas'][0]
                    infos[0]['obj_dist'] = batch['obj_dists'][0]
                
                if (all(np.isclose(observations[0]['gps'],  last_gps))) and \
                   (all(np.isclose(observations[0]['compass'], last_compass))) and \
                   (step_data[0] in [1, 2, 3]):
                    is_collide = True
                else:
                    is_collide = False
                # print(nav_step, is_collide, infos[0]['collisions'])
                # print(last_gps, last_compass)
                # print(observations[0]['gps'], observations[0]['compass'])
                last_gps = observations[0]['gps']
                last_compass = observations[0]['compass']

                if is_collide:
                    action_masks[0, step_data[0]] = True
                else:
                    action_masks[0, :] = False
                pred_trajectories['actions'].append(step_data[0])
                pred_trajectories['infos'].append(infos[0])

                if config.MODEL.enc_collide_steps or (not is_collide):
                    batch = batch_obs(observations, device=self.device)
                    batch = process_batch_gps_compass(batch, gpscompass_noise_type)
                    batch = apply_obs_transforms_batch(batch, self.obs_transforms)
                    if self.semantic_predictor is not None:
                        batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
                        if self.config.MODEL.SEMANTIC_ENCODER.is_thda:
                            batch["semantic"] = batch["semantic"] - 1
                    if model_config.USE_PRETRAINED_SEGMENTATION:
                        batch['obj_areas'], batch['obj_dists'] = [], []
                        for ib in range(len(batch['rgb'])):
                            obj_area, obj_dist = self.segmentation_model.predict_target(
                                batch['rgb'][ib], batch['depth'][ib], batch['objectgoal'][ib],
                                # outfile=os.path.join('../notebooks/segmasks', '%04d.png'%(nav_step+1))
                            )
                            # import cv2
                            # cv2.imwrite('../notebooks/segmasks/rgb_%04d.png'%(nav_step+1), batch['rgb'][ib].data.cpu().numpy())
                            # print(nav_step, batch['objectgoal'][ib], step_data[ib], obj_area, obj_dist)
                            batch['obj_areas'].append(obj_area)
                            batch['obj_dists'].append(obj_dist)

                    nav_step += 1
                    prev_actions.copy_(actions.unsqueeze(1))  # type: ignore

            # print(current_episodes[0].object_category)
            # print(pred_trajectories['actions'])
            # print(pred_trajectories['infos'][0])
            # print(infos[0])
            # print()
            if config.SAVE_RECURSIVE_STATE:
                state_outdir = os.path.join(config.EVAL_RESULTS_DIR, 'recursive_states')
                os.makedirs(state_outdir, exist_ok=True)
                np.save(os.path.join(state_outdir, '%s.npy'%(current_episodes[0].episode_id)), recursive_states[0].data.cpu().numpy())

            episode_stats = {}
            episode_stats["reward"] = current_episode_reward
            episode_stats.update(
                self._extract_scalars_from_info(infos[0])
            )
            stats_episodes[
                (
                    current_episodes[0].scene_id,
                    current_episodes[0].episode_id,
                )
            ] = episode_stats

            pred_trajectories.update({'episode_id': current_episodes[0].episode_id})
            write_to_jsonlines(pred_outfile, pred_trajectories)
            if len(self.config.VIDEO_OPTION) > 0:
                generate_video(
                    video_option=self.config.VIDEO_OPTION,
                    video_dir=self.config.VIDEO_DIR,
                    images=rgb_frames,
                    episode_id=current_episodes[0].episode_id,
                    checkpoint_idx=checkpoint_index,
                    metrics=self._extract_scalars_from_info(infos[0]),
                    tb_writer=writer,
                )

        return stats_episodes


def get_episodic_gps(agent_state, start_position, start_rotation):
    origin = np.array(start_position, dtype=np.float32)
    rotation_world_start = quaternion_from_coeff(start_rotation)

    agent_position = agent_state.position
    agent_position = quaternion_rotate_vector(
        rotation_world_start.inverse(), agent_position - origin
    )

    return np.array(
        [-agent_position[2], agent_position[0]], dtype=np.float32
    )

def get_episodic_compass(agent_state, start_rotation):
    rotation_world_agent = agent_state.rotation
    rotation_world_start = quaternion_from_coeff(start_rotation)

    quat = rotation_world_agent.inverse() * rotation_world_start
    direction_vector = np.array([0, 0, -1])
    heading_vector = quaternion_rotate_vector(quat, direction_vector)
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return np.array([phi], dtype=np.float32)
