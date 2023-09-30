#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
from tarfile import is_tarfile

import numpy as np

import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

from habitat_baselines.config.default import get_config
from habitat_baselines.common.environments import NavRLEnv

cv2 = try_cv2_import()


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def viz_human_demos(scene_id, output_dir, num_episodes=10, shortest_path=False):
    config = get_config(
        config_paths="habitat_baselines/config/objectnav/il_objectnav.yaml"
    )
    print(config)
    config.defrost()
    config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene_id]
    config.freeze()        

    with NavRLEnv(config=config) as env:
        print("Environment creation successful")
        for i in range(num_episodes):
            env.reset()
            current_episode = env.current_episode

            dirname = os.path.join(output_dir, scene_id)
            os.makedirs(dirname, exist_ok=True)
            print("Agent stepping around inside environment.")

            if shortest_path:
                if current_episode.is_thda:
                    goal_radius = 1.0
                    follower = ShortestPathFollower(
                        env.habitat_env.sim, goal_radius, False
                    )
                else:
                    actions = [x.action if x.action is not None else 0 for x in current_episode.shortest_paths[0]]

            else:
                actions = [x.action for x in current_episode.reference_replay]
                assert actions[0] == 'STOP'
                actions = actions[1:]

            images = []
            t = 0
            while True:
                if shortest_path and current_episode.is_thda:
                    action = follower.get_next_action(
                        env.habitat_env.current_episode.goals[0].position
                    )   # int
                else:
                    action = actions[t]

                if action == 'STOP' or action == 0:
                    break
                observations, reward, done, info = env.step(action=action)
                im = observations["rgb"]
                top_down_map = draw_top_down_map(info, im.shape[0])
                output_im = np.concatenate((im, top_down_map), axis=1)
                images.append(output_im)
                t += 1

            images_to_video(
                images, dirname, 
                f'{current_episode.object_category}-{current_episode.episode_id}'
            )
            print(f"{i} Episode finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--shortest_path', action='store_true', default=False)
    args = parser.parse_args()

    scene_ids = [
        x.split('.')[0] for x in \
            os.listdir('data/datasets/objectnav/objectnav_mp3d_70k/train/content')
    ]
    for scene_id in scene_ids[1:10]:
        viz_human_demos(
            scene_id, args.output_dir, 
            num_episodes=args.num_episodes,
            shortest_path=args.shortest_path,
        )


if __name__ == "__main__":
    main()
