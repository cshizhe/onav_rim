'''
For each demo episode, extract:
- RGB images: T x (224, 224, 3)
- Depth features extracted by a pretrained Resnet50 (DDPPO): (T, 2048)
- Predicted semantics by a pretrained Rednet segmentor: T x (224, 224, 28)
'''

import os
import argparse
import json
import jsonlines
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm, trange
import collections

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from habitat_baselines.config.default import get_config
from habitat_baselines.common.environments import NavRLEnv

import torch
import torch.nn as nn

import torchvision.transforms as T

from offline_bc.preprocess.encoders import (
    CLIPEncoder,
    ResnetEncoders,
    DepthEncoder,
    SemanticPredictor,
)

ACTION_MAPS = {
    'STOP': 0, 'MOVE_FORWARD': 1, 'TURN_LEFT': 2, 'TURN_RIGHT': 3, 'LOOK_UP': 4, 'LOOK_DOWN': 5
}

category_to_task_category_id = {
    'chair': 0,
    'table': 1,
    'picture': 2,
    'cabinet': 3,
    'cushion': 4,
    'sofa': 5,
    'bed': 6,
    'chest_of_drawers': 7,
    'plant': 8,
    'sink': 9,
    'toilet': 10,
    'stool': 11,
    'towel': 12,
    'tv_monitor': 13,
    'shower': 14,
    'bathtub': 15,
    'counter': 16,
    'fireplace': 17,
    'gym_equipment': 18,
    'seating': 19,
    'clothes': 20
}
category_to_mp3d_category_id = {
    'chair': 3,
    'table': 5,
    'picture': 6,
    'cabinet': 7,
    'cushion': 8,
    'sofa': 10,
    'bed': 11,
    'chest_of_drawers': 13,
    'plant': 14,
    'sink': 15,
    'toilet': 18,
    'stool': 19,
    'towel': 20,
    'tv_monitor': 22,
    'shower': 23,
    'bathtub': 25,
    'counter': 26,
    'fireplace': 27,
    'gym_equipment': 33,
    'seating': 34,
    'clothes': 38
}


LABEL_MAP = {}
for k, v in category_to_mp3d_category_id.items():
    LABEL_MAP[v] = category_to_task_category_id[k]


def extract_demo_obs_and_fts(args):
    # build env
    config = get_config(args.cfg_file)
    config.defrost()

    cfg = config.TASK_CONFIG
    cfg.DATASET.CONTENT_SCENES = [args.scene_id]

    cfg.DATASET.MAX_REPLAY_STEPS = 2000
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = 2000

    cfg.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1

    orientation = np.deg2rad(args.camera_init_pitch)
    cfg.SIMULATOR.DEPTH_SENSOR.HFOV = args.camera_hfov
    cfg.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]
    cfg.SIMULATOR.DEPTH_SENSOR.ORIENTATION = [orientation, 0, 0]
    cfg.SIMULATOR.RGB_SENSOR.HFOV = args.camera_hfov
    cfg.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]
    cfg.SIMULATOR.RGB_SENSOR.ORIENTATION = [orientation, 0, 0]
    cfg.SIMULATOR.SEMANTIC_SENSOR.HFOV = args.camera_hfov
    cfg.SIMULATOR.SEMANTIC_SENSOR.POSITION = [0, args.camera_height, 0]
    cfg.SIMULATOR.SEMANTIC_SENSOR.ORIENTATION = [orientation, 0, 0]
    cfg.SIMULATOR.AGENT_0.SENSORS.append('SEMANTIC_SENSOR')

    config.freeze()

    env = NavRLEnv(config)
    num_episodes = len(env.episodes)
    print('num episodes:', num_episodes)

    # build encoders
    device = torch.device('cuda:0')
    torch.set_grad_enabled(False)

    rgb_preprocess = T.Compose([
        T.Resize(args.image_size),
        T.CenterCrop(size=(args.image_size, args.image_size))
    ])

    if args.encode_rgb_clip:
        rgb_clip_encoder = CLIPEncoder(device, batch_size=args.batch_size)
    if args.encode_rgb_resnet:
        rgb_resnet_encoders = ResnetEncoders(device, ['resnet18', 'resnet50'], batch_size=args.batch_size)
    if args.encode_depth:
        depth_encoder = DepthEncoder(
            device, env.observation_space,
            config.MODEL.DEPTH_ENCODER.ddppo_checkpoint,
            config.MODEL.DEPTH_ENCODER.backbone,
            batch_size=args.batch_size
        )
    if args.predict_semantic:
        semantic_predictor = SemanticPredictor(
            device, 
            config.MODEL.SEMANTIC_ENCODER.rednet_ckpt,
            config.MODEL.SEMANTIC_ENCODER.num_classes,
            batch_size=args.batch_size
        )

    # extract observations and features
    if args.save_rgb:
        os.makedirs(os.path.join(args.outdir, 'rgb_images'), exist_ok=True)
        rgb_lmdb_env = lmdb.open(
            os.path.join(args.outdir, 'rgb_images', args.scene_id),
            map_size=int(1024**4)
        )
    if args.save_semantic:
        os.makedirs(os.path.join(args.outdir, 'sem_images'), exist_ok=True)
        sem_lmdb_env = lmdb.open(
            os.path.join(args.outdir, 'sem_images', args.scene_id),
            map_size=int(1024**4)
        )
    if args.save_semantic_fts:
        os.makedirs(os.path.join(args.outdir, 'sem_fts'), exist_ok=True)
        sem_ft_lmdb_env = lmdb.open(
            os.path.join(args.outdir, 'sem_fts', args.scene_id),
            map_size=int(1024**4)
        )

    rgb_encoder_names = []
    rgb_ft_lmdb_envs = {}
    if args.encode_rgb_clip:
        rgb_encoder_names.append('clip')
    if args.encode_rgb_resnet:
        rgb_encoder_names.extend(['resnet18', 'resnet50'])
    for name in rgb_encoder_names:
        os.makedirs(os.path.join(args.outdir, 'rgb_fts', name), exist_ok=True)
        rgb_ft_lmdb_envs[name] = lmdb.open(
            os.path.join(args.outdir, 'rgb_fts', name, args.scene_id),
            map_size=int(1024**4)
        )

    if args.encode_depth:
        os.makedirs(os.path.join(args.outdir, 'depth_fts'), exist_ok=True)
        depth_ft_lmdb_env = lmdb.open(
            os.path.join(args.outdir, 'depth_fts', args.scene_id),
            map_size=int(1024**4)
        )

    if args.predict_semantic:
        os.makedirs(os.path.join(args.outdir, 'semantic_preds'), exist_ok=True)
        sem_preds_lmdb_env = lmdb.open(
            os.path.join(args.outdir, 'semantic_preds', args.scene_id),
            map_size=int(1024**4)
        )

    os.makedirs(os.path.join(args.outdir, 'meta_infos_lmdb'), exist_ok=True)
    meta_info_lmdb_env = lmdb.open(
        os.path.join(args.outdir, 'meta_infos_lmdb', args.scene_id),
        map_size=int(1024**4)
    )

    for _ in trange(num_episodes):
        observations = env.reset()
        episode_id = env.current_episode.episode_id
        # rgb <class 'numpy.ndarray'> (480, 640, 3) uint8 0 255
        # depth <class 'numpy.ndarray'> (480, 640, 1) float32 0.0 1.0
        # objectgoal <class 'numpy.ndarray'> (1,) int64 5 5
        # compass <class 'numpy.ndarray'> (1,) float32 1.0365224e-09 1.0365224e-09
        # gps <class 'numpy.ndarray'> (2,) float32 -0.0 -0.0
        # demonstration <class 'int'> 3
        # inflection_weight <class 'float'> 1.0

        demo_actions = [
            x.action if x.action is not None else 'STOP' \
                for x in env.current_episode.reference_replay[1:]
        ]
        demo_actions = [ACTION_MAPS[x] for x in demo_actions]

        episode_obs = {'reward': [], 'info': []}

        for action in demo_actions:
            for k, v in observations.items():
                episode_obs.setdefault(k, [])
                episode_obs[k].append(v)
            observations, reward, done, info = env.step(action=action)
            episode_obs['reward'].append(reward)
            episode_obs['info'].append(info)
            if action == 'STOP':
                break

        # remove collision steps
        if not args.keep_collision_steps:
            collide_step_ids = set()
            gps = episode_obs['gps']
            compass = episode_obs['compass']
            for t in range(len(gps)-1):
                # look up and down also result in the same gps and compass
                if all(np.isclose(gps[t], gps[t+1])) and all(np.isclose(compass[t], compass[t+1])) and demo_actions[t] in [1, 2, 3]:
                    collide_step_ids.add(t)
                    
            for k, v in episode_obs.items():
                episode_obs[k] = [v[idx] for idx in range(len(v)) if idx not in collide_step_ids]
            demo_actions = [x for idx, x in enumerate(demo_actions) if idx not in collide_step_ids]

        resized_rgb_images = [rgb_preprocess(Image.fromarray(x)) for x in episode_obs['rgb']]

        rgb_fts = {}
        if args.encode_rgb_clip:
            rgb_fts['clip'] = rgb_clip_encoder.extract_fts(resized_rgb_images)
        if args.encode_rgb_resnet:
            rgb_fts.update(rgb_resnet_encoders.extrat_fts(resized_rgb_images))

        if args.encode_depth:
            depth_images = torch.stack([torch.from_numpy(x) for x in episode_obs['depth']]).to(device)
            depth_fts = depth_encoder.extract_fts(depth_images) # (batch, dim_ft)

        if args.predict_semantic:
            rgb_images = torch.from_numpy(np.stack(episode_obs['rgb'], 0)).to(device)
            sem_preds = semantic_predictor.predict(rgb_images, depth_images) # (batch, height, width)
        
        # save to files
        episode_key = episode_id.encode('ascii')

        if args.save_rgb:  
            for t, x in enumerate(resized_rgb_images):
                x = np.array(x)
                _, x = cv2.imencode('.png', x, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                txn = rgb_lmdb_env.begin(write=True)
                txn.put(('%s:%04d'%(episode_id, t)).encode('ascii'), x)
                txn.commit()

        if args.save_semantic:
            for t, x in enumerate(episode_obs['semantic']):
                _, x = cv2.imencode('.png', x, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                txn = sem_lmdb_env.begin(write=True)
                txn.put(('%s:%04d'%(episode_id, t)).encode('ascii'), x)
                txn.commit()

        if args.save_semantic_fts:
            sem_fts = np.zeros((len(episode_obs['semantic']), len(LABEL_MAP)), dtype=np.float32)
            for t, x in enumerate(episode_obs['semantic']):
                x = x.flatten()
                label_counter = collections.Counter(x)
                npixels = len(x)
                for label, count in label_counter.most_common():
                    if label in LABEL_MAP:
                        sem_fts[t, LABEL_MAP[label]] = count / npixels
            txn = sem_ft_lmdb_env.begin(write=True)
            txn.put(episode_key, msgpack.packb(sem_fts))
            txn.commit()
        
        if args.encode_rgb_clip or args.encode_rgb_resnet:
            for k, v in rgb_fts.items():
                txn = rgb_ft_lmdb_envs[k].begin(write=True)
                txn.put(episode_key, msgpack.packb(v))
                txn.commit()

        if args.encode_depth:
            txn = depth_ft_lmdb_env.begin(write=True)
            txn.put(episode_key, msgpack.packb(depth_fts))
            txn.commit()

        if args.predict_semantic:            
            sem_preds = np.concatenate([
                np.array(x) for x in sem_preds
            ], 0)
            _, sem_preds = cv2.imencode(
                '.png', sem_preds, 
                [cv2.IMWRITE_PNG_COMPRESSION, 3]
            )
            txn = sem_preds_lmdb_env.begin(write=True)
            txn.put(episode_key, sem_preds)
            txn.commit()

        meta_info = {
            'episode_id': episode_key.decode('ascii'),
            'objectgoal': episode_obs['objectgoal'][0].item(),
            'object_category': env.current_episode.object_category,
            'compass': np.stack(episode_obs['compass'], 0).tolist(),
            'gps': np.stack(episode_obs['gps'], 0).tolist(),
            'demonstration': demo_actions,
            # 'demonstration': np.array(episode_obs['demonstration']).tolist(),
            # 'inflection_weight': np.array(episode_obs['inflection_weight']).tolist(),
            'reward': np.array(episode_obs['reward']).tolist(),
            'info': episode_obs['info'],
        }
        txn = meta_info_lmdb_env.begin(write=True)
        txn.put(episode_key, msgpack.packb(meta_info))
        txn.commit()

    if args.save_rgb:
        rgb_lmdb_env.close()
    if args.save_semantic:
        sem_lmdb_env.close()
    if args.save_semantic_fts:
        sem_ft_lmdb_env.close()
    if args.encode_rgb_clip or args.encode_rgb_resnet:
        for k, v in rgb_ft_lmdb_envs.items():
            v.close()
    if args.encode_depth:
        depth_ft_lmdb_env.close()
    if args.predict_semantic:
        sem_preds_lmdb_env.close()
    meta_info_lmdb_env.close()

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg_file', help='habitat_baselines config file', 
        default='habitat_baselines/config/objectnav/il_ddp_objectnav.yaml'
    )
    parser.add_argument('--scene_id', help='scene id in mp3d', default='gZ6f7yhEvPG')

    parser.add_argument('--camera_hfov', type=int, default=79, help='degree')
    parser.add_argument('--camera_height', type=float, default=0.88, help='meter')
    parser.add_argument('--camera_init_pitch', type=int, default=0,
                        help='+lookup, -lookdown (degrees)')
    # parser.add_argument('--agent_height', type=float, default=0.88)
    # parser.add_argument('--agent_radius', type=float, default=0.18)

    parser.add_argument('--save_rgb', action='store_true', default=False)
    parser.add_argument('--save_semantic', action='store_true', default=False)
    parser.add_argument('--save_semantic_fts', action='store_true', default=False)

    parser.add_argument('--encode_depth', action='store_true', default=False)
    parser.add_argument('--encode_rgb_clip', action='store_true', default=False)
    parser.add_argument('--encode_rgb_resnet', action='store_true', default=False)
    parser.add_argument('--predict_semantic', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--keep_collision_steps', action='store_true', default=False)

    parser.add_argument('--outdir', help='output directory')
    
    args = parser.parse_args()

    extract_demo_obs_and_fts(args)


if __name__ == '__main__':
    main()
