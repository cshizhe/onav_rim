import os
import json
import jsonlines
import numpy as np
import cv2
import time
import math
import random
import collections

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from offline_bc.utils.logger import LOGGER
from offline_bc.utils.ops import pad_tensors, gen_seq_masks
THDA_OBJECTS = set(['foodstuff', 'fruit', 'plaything', 'game_equipment', 'hand_tool', 'kitchenware', 'stationery'])

MAP_IMAGE_SIZE = 960

def crop_centered_local_map(
    agent_loc, global_map, local_sizes, 
    keep_centered_when_out_of_boundary=False,
    channel_before_size=False
):
    '''Crop a local map with local_sizes in the global_map centered at the agent's location.
    keep_centered_when_out_of_boundary: if True, we will pad zeros in the local map
    '''
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    if channel_before_size:
        n_channel, full_w, full_h = global_map.shape
    else:
        full_w, full_h, n_channel = global_map.shape

    assert (local_w <= full_w) & (local_h <= full_h)

    gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
    gx2, gy2 = gx1 + local_w, gy1 + local_h
    
    if keep_centered_when_out_of_boundary:
        if isinstance(global_map, np.ndarray):
            if channel_before_size:
                local_map = np.zeros((n_channel, local_w, local_h), dtype=global_map.dtype)
            else:
                local_map = np.zeros((local_w, local_h, n_channel), dtype=global_map.dtype)
        elif isinstance(global_map, torch.Tensor):
            if channel_before_size:
                local_map = torch.zeros(
                    n_channel, local_w, local_h,
                    dtype=global_map.dtype, device=global_map.device
                )
            else:
                local_map = torch.zeros(
                    local_w, local_h, n_channel, 
                    dtype=global_map.dtype, device=global_map.device
                )
    
        lx1 = max(- gx1, 0)
        ly1 = max(- gy1, 0)
        if lx1 >= local_w or ly1 >= local_h:
            return local_map
        lx2 = lx1 + local_w - max(0, gx2 - full_w)
        ly2 = ly1 + local_h - max(0, gy2 - full_h)
        if lx2 <= 0 or ly2 <= 0:
            return local_map

        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(full_w, gx2), min(full_h, gy2)
       
        if channel_before_size:
            local_map[:, lx1: lx2, ly1: ly2] = global_map[:, gx1: gx2, gy1: gy2]
        else:
            local_map[lx1: lx2, ly1: ly2] = global_map[gx1: gx2, gy1: gy2]

    else:
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h

        if channel_before_size:
            local_map = global_map[:, gx1: gx2, gy1: gy2]
        else:
            local_map = global_map[gx1: gx2, gy1: gy2]

    return local_map

class NavDemoDataset(Dataset):
    def __init__(
        self, trn_scene_ids, rgb_image_dir=None, rgb_ft_dir=None, 
        depth_ft_dir=None, sem_ft_dir=None, meta_dir=None, semseg_dir=None,
        rgb_image_size=224, max_steps=500, inflection_weight=None,
        num_ft_views=1, num_history_fts=None, use_thda=True, 
        val_scene_ids=[], validation=False, map_dir=None, 
        map_image_size=240, **kwargs
    ) -> None:
        if validation:
            self.scene_ids = val_scene_ids
        else:
            if trn_scene_ids == '*':
                self.scene_ids = [os.path.splitext(x)[0] for x in os.listdir(meta_dir)]
            else:
                self.scene_ids = trn_scene_ids
            # if len(val_scene_ids) > 0:
            #     self.scene_ids = [x for x in self.scene_ids if x not in val_scene_ids]

        self.rgb_image_dir = rgb_image_dir
        self.rgb_ft_dir = rgb_ft_dir
        self.depth_ft_dir = depth_ft_dir
        self.semseg_dir = semseg_dir
        self.sem_ft_dir = sem_ft_dir
        self.meta_dir = meta_dir
        self.rgb_image_size = rgb_image_size
        self.max_steps = max_steps
        self.num_history_fts = num_history_fts if num_history_fts is not None else max_steps
        self.inflection_weight = inflection_weight
        self.num_ft_views = num_ft_views
        self.use_thda = use_thda
        self.kwargs = kwargs

        self._all_lmdb_envs = []

        self.meta_info_envs, self.meta_info_txns = self.build_lmdb_envs(
            self.meta_dir, self.scene_ids
        )
        if rgb_image_dir is not None:
            self.rgb_image_envs, self.rgb_image_txns = self.build_lmdb_envs(rgb_image_dir, self.scene_ids)
        if rgb_ft_dir is not None:
            if isinstance(rgb_ft_dir, str):
                rgb_ft_dir = [rgb_ft_dir]
            self.rgb_ft_envs = collections.defaultdict(list)
            self.rgb_ft_txns = collections.defaultdict(list)
            for x in rgb_ft_dir:
                rgb_ft_envs, rgb_ft_txns = self.build_lmdb_envs(x, self.scene_ids)
                for k in rgb_ft_envs.keys():
                    self.rgb_ft_envs[k].append(rgb_ft_envs[k])
                    self.rgb_ft_txns[k].append(rgb_ft_txns[k])
        if depth_ft_dir is not None:
            if isinstance(depth_ft_dir, str):
                depth_ft_dir = [depth_ft_dir]
            self.depth_ft_envs = collections.defaultdict(list)
            self.depth_ft_txns = collections.defaultdict(list)
            for x in depth_ft_dir:
                depth_ft_envs, depth_ft_txns = self.build_lmdb_envs(x, self.scene_ids)
                for k in depth_ft_envs.keys():
                    self.depth_ft_envs[k].append(depth_ft_envs[k])
                    self.depth_ft_txns[k].append(depth_ft_txns[k])
        if semseg_dir is not None:
            self.semseg_envs, self.semseg_txns = self.build_lmdb_envs(semseg_dir, self.scene_ids)
        if sem_ft_dir is not None:
            self.sem_ft_envs, self.sem_ft_txns = self.build_lmdb_envs(sem_ft_dir, self.scene_ids)

        self.load_dataset_meta_infos()

        self.map_dir = map_dir
        self.map_image_size = map_image_size
        if self.map_dir is not None:
            self.data_idxs, self.num_episodes = [], 0
            self.visited_locs_in_full_map = {}
            for scene_id in self.scene_ids:
                self.visited_locs_in_full_map[scene_id] = {}
                with jsonlines.open(os.path.join(map_dir, 'meta_infos', f'{scene_id}.jsonl'), 'r') as f:
                    for item in f:
                        self.visited_locs_in_full_map[scene_id][item['episode_id']] = item['visited_locs_in_full_map']
                        self.data_idxs.append((scene_id, item['episode_id']))
                        self.num_episodes += 1
            resized_map_dir = os.path.join(self.map_dir, 'infer_map_%s_size_%d_pool_%d'%(
                '_'.join(self.kwargs['map_types']), self.map_image_size, 
                self.kwargs['map_image_max_pool'])
            )
            if os.path.exists(resized_map_dir):
                self.resized_map_dir = resized_map_dir
                map_env_dir = resized_map_dir
            else:
                self.resized_map_dir = None
                map_env_dir = os.path.join(map_dir, 'maps')
            self.map_envs, self.map_txns = self.build_lmdb_envs(map_env_dir, self.scene_ids)
            LOGGER.info(f'MAP: #scenes: {len(self.scene_ids)}, #episodes: {self.num_episodes}')

    def load_dataset_meta_infos(self):
        self.data_idxs, self.num_episodes = [], 0

        for scene_id in self.scene_ids:
            for episode_id, item in self.meta_info_txns[scene_id].cursor():
                episode_id = episode_id.decode('ascii')
                item = msgpack.unpackb(item)
                if (not self.use_thda) and (item['object_category'] in THDA_OBJECTS):
                    continue
                if len(item['demonstration']) > self.kwargs.get('max_episode_steps', float('inf')):
                    continue
                self.data_idxs.append((scene_id, episode_id))
                self.num_episodes += 1

        LOGGER.info(f'#scenes: {len(self.scene_ids)}, #episodes: {self.num_episodes}')

    def build_lmdb_envs(self, lmdb_dir, scene_ids):
        lmdb_envs, lmdb_txns = {}, {}
        for scene_id in scene_ids:
            lmdb_envs[scene_id] = lmdb.open(
                os.path.join(lmdb_dir, scene_id), 
                readonly=True, lock=False, readahead=False, meminit=False
            )
            lmdb_txns[scene_id] = lmdb_envs[scene_id].begin()
            self._all_lmdb_envs.append(lmdb_envs[scene_id])
        return lmdb_envs, lmdb_txns

    def load_map_at_step(self, txn, episode_id, step, map_names=['obs', 'exp']):
        imgs = []
        for map_name in map_names:
            key = f'{episode_id}_{step}_{map_name}'
            value = txn.get(key.encode('ascii'))
            if value is None:
                img = np.zeros((MAP_IMAGE_SIZE, MAP_IMAGE_SIZE), dtype=np.uint8)
                LOGGER.info(f'DATASET ERROR: {key} not exists')
            else:
                img = np.frombuffer(value, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)   # (MAP_IMAGE_SIZE, MAP_IMAGE_SIZE)
            imgs.append(img)
        imgs = np.stack(imgs, axis=2)
        return imgs
    
    def __del__(self):
        for lmdb_env in self._all_lmdb_envs:
            lmdb_env.close()

    def __len__(self):
        return self.num_episodes

    def load_episode_meta_info(self, txn, episode_id):
        key = episode_id.encode('ascii')
        value = txn.get(key)
        value = msgpack.unpackb(value)
        return value

    def load_episode_images(self, txn, episode_id, im_height, im_width, im_channel):
        key = episode_id.encode('ascii')
        value = txn.get(key)
        assert value is not None, episode_id
        images_flt = np.frombuffer(value, dtype=np.uint8)
        images_flt = cv2.imdecode(
            images_flt, 
            cv2.IMREAD_GRAYSCALE if im_channel == 1 else cv2.IMREAD_COLOR
        )
        images = images_flt.reshape(-1, im_height, im_width, im_channel)
        return images

    def load_episode_fts(self, txn, episode_id):
        key = episode_id.encode('ascii')
        value = txn.get(key)
        assert value is not None, episode_id
        fts = msgpack.unpackb(value)
        if len(fts.shape) == 3: # (nsteps, num_views, dim)
            if self.num_ft_views == 1:
                fts = fts[:, 0]
            elif self.num_ft_views == 12: # horizontal pano
                fts = fts[:, :12]
            elif self.num_ft_views == 3: # up and down
                fts = np.concatenate([fts[:, :1], fts[:, -2:]], axis=1)
            elif self.num_ft_views == 2: # down
                fts = np.concatenate([fts[:, :1], fts[:, -1:]], axis=1)
            elif self.num_ft_views == '2_up':
                fts = np.concatenate([fts[:, :1], fts[:, -2:-1]], axis=1)
        # BUG: The given NumPy array is not writeable if without np.array()
        fts = np.array(fts, dtype=np.float32)
        return fts

    def get_full_episode(self, scene_id, episode_id, meta):
        outs = {
            'episode_ids': episode_id,
            'scene_ids': scene_id,
            'objectgoal': meta['objectgoal'],
            'object_category': meta['object_category'],
            'compass': np.array(meta['compass'], dtype=np.float32), # (nsteps, 1)
            'gps': np.array(meta['gps'], dtype=np.float32), # (nsteps, 2)
            'demonstration': np.array(meta['demonstration'], dtype=np.int64), # (nsteps, )
        }

        if self.rgb_image_dir is not None:
            outs['rgb'] = self.load_episode_images(
                self.rgb_image_txns[scene_id], episode_id,
                self.rgb_image_size, self.rgb_image_size, 3
            )
        if self.rgb_ft_dir is not None:
            ridx = np.random.randint(len(self.rgb_ft_txns[scene_id]))
            outs['rgb_features'] = self.load_episode_fts(
                self.rgb_ft_txns[scene_id][ridx], episode_id
            )
        if self.depth_ft_dir is not None:
            ridx = np.random.randint(len(self.depth_ft_txns[scene_id]))
            outs['depth_features'] = self.load_episode_fts(
                self.depth_ft_txns[scene_id][ridx], episode_id
            )
        if self.sem_ft_dir is not None:
            sem_fts = self.load_episode_fts(
                self.sem_ft_txns[scene_id], episode_id
            )
            if meta['objectgoal'] < 21:
                sem_goal_fts = sem_fts[:, meta['objectgoal']][:, None]
            else:
                sem_goal_fts = np.zeros((sem_fts.shape[0], 1), dtype=np.float32)
            outs['sem_features'] = np.concatenate(
                [sem_fts, sem_goal_fts], 1
            )
        if self.semseg_dir is not None:
            outs['semantic'] = self.load_episode_images(
                self.semseg_txns[scene_id], episode_id,
                480, 640, 1
            ) - 1 # [1, 29] -> [0, 28]

        num_steps = len(meta['compass'])

        outs['inflection_weight'] = np.ones((num_steps, ), dtype=np.float32)
        if self.inflection_weight is not None:
            prev_actions = np.array([0] + meta['demonstration'][:-1])
            actions = np.array(meta['demonstration'])
            outs['inflection_weight'][prev_actions != actions] = self.inflection_weight
        stop_weight = self.kwargs.get('stop_weight', None)
        if stop_weight is not None:
            outs['inflection_weight'][-1] = stop_weight

        return outs

    def __getitem__(self, data_idx):
        scene_id, episode_id = self.data_idxs[data_idx]
        meta = self.load_episode_meta_info(self.meta_info_txns[scene_id], episode_id)

        outs = self.get_full_episode(scene_id, episode_id, meta)
        num_steps = len(meta['compass'])

        if num_steps > self.num_history_fts:
            even_start_idxs = np.arange(0, num_steps-self.num_history_fts+1, self.num_history_fts).tolist() + \
                              np.arange(num_steps-self.num_history_fts, 0, -self.num_history_fts).tolist()
            even_start_idxs = list(set(even_start_idxs))
            start_step = np.random.choice(even_start_idxs)
            if self.kwargs.get('no_random_start_step', False):
                start_step = 0
            end_step = start_step + self.num_history_fts
        else:
            start_step, end_step = 0, num_steps

        if self.kwargs.get('infer_visual_feature_task', False):
            max_future_step_size = self.kwargs.get('max_future_step_size', 100)
            infer_time_ids = np.array([
                np.random.randint(start_step, min(num_steps, t+max_future_step_size)) \
                    for t in range(start_step, end_step)
            ])
            outs['infer_gps'] = outs['gps'][infer_time_ids]
            outs['infer_compass'] = outs['compass'][infer_time_ids]
            outs['infer_visual_features'] = outs['rgb_features'][infer_time_ids]
            if self.kwargs.get('infer_depth_feature', False):
                outs['infer_visual_features'] = np.concatenate(
                    [outs['infer_visual_features'], outs['depth_features'][infer_time_ids]],
                    axis=-1
                )

        if self.kwargs.get('infer_local_map_task', False):
            map_types = self.kwargs.get('map_types', ['obs'])
            max_pool_size = self.kwargs.get('map_image_max_pool', 1)
            visited_locs_in_full_map = self.visited_locs_in_full_map[scene_id][episode_id]
            if self.resized_map_dir is None:
                local_maps = []
                for t in range(start_step, end_step):
                    local_maps.append(
                        crop_centered_local_map(
                            visited_locs_in_full_map[t], 
                            self.load_map_at_step(self.map_txns[scene_id], episode_id, t, map_types),
                            (self.map_image_size, self.map_image_size),
                            keep_centered_when_out_of_boundary=True
                        )
                    )
                local_maps = np.stack(local_maps, 0)
                local_maps = torch.from_numpy(local_maps) # (nsteps, height, width, nchannels)
                local_maps = F.max_pool2d(
                    local_maps.permute(0, 3, 1, 2).float(), max_pool_size, max_pool_size
                ).bool().view(end_step - start_step, len(map_types), -1)
            else:
                local_maps = msgpack.unpackb(self.map_txns[scene_id].get(episode_id.encode('ascii')))
                local_maps = local_maps[start_step: end_step]
                local_maps = torch.from_numpy(local_maps) # (nsteps, height, width, nchannels)
            outs['infer_local_maps'] = local_maps

        if num_steps > self.num_history_fts:
            for k, v in outs.items():
                if isinstance(v, np.ndarray) and (not k.startswith('infer')):
                    outs[k] = v[start_step: end_step]

        outs['step_ids'] = np.arange(start_step, end_step)
        if outs['step_ids'][-1] >= self.max_steps:
            outs['step_ids'] = outs['step_ids'] - start_step
        outs['ori_start_step'] = start_step
        outs['num_steps'] = min(self.num_history_fts, num_steps)

        gpscompass_noise_type = self.kwargs.get('gpscompass_noise_type', None)
        if gpscompass_noise_type == 'zero':
            outs['gps'][:] = 0
            outs['compass'][:] = 0
        elif gpscompass_noise_type == 'gaussian':
            outs['gps'] = outs['gps'] + np.random.normal(0, 1, size=outs['gps'].shape)
            outs['compass'] = outs['compass'] + np.random.normal(0, 1, size=outs['compass'].shape)

        return outs


def collate_fn(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }

    max_steps = np.max(batch['num_steps'])
    # batch['num_steps'] = torch.LongTensor(batch['num_steps'])

    if 'rgb' in batch:
        # (batch x nsteps, 224, 224, 3)
        batch['rgb'] = np.concatenate(batch['rgb'], 0)
    if 'semantic' in batch:
        # (batch x nsteps, 480, 640, 1)
        batch['semantic'] = np.concatenate(batch['semantic'], 0)
    
    # for key in ['rgb_features', 'depth_features', 'compass', 'gps', \
    #             'infer_gps', 'infer_compass', 'infer_visual_features']:
    #     if key in batch:
    #         # (batch, max_steps, dim_ft)
    #         batch[key] = pad_tensors(
    #             [torch.FloatTensor(x) for x in batch[key]]
    #         )
    # if batch['infer_gps'].size(1) != batch['rgb_features'].size(1):
    #     print(batch['infer_gps'].size(), batch['rgb_features'].size())

    batch['objectgoal'] = torch.LongTensor(batch['objectgoal'])
    batch['demonstration'] = pad_sequence(
        [torch.LongTensor(x) for x in batch['demonstration']], 
        batch_first=True, padding_value=-100
    )
    batch['inflection_weight'] = pad_sequence(
        [torch.FloatTensor(x) for x in batch['inflection_weight']], 
        batch_first=True, padding_value=0
    )
    batch['step_ids'] = pad_sequence(
        [torch.from_numpy(x) for x in batch['step_ids']],
        batch_first=True, padding_value=0
    )
    return batch
