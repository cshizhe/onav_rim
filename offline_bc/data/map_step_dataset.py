import os
import jsonlines
import numpy as np
import cv2
import time

import torch

from torch.nn.utils.rnn import pad_sequence
from offline_bc.utils.ops import pad_tensors

from offline_bc.utils.logger import LOGGER
from offline_bc.data.dataset import NavDemoDataset


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

class NavMapStepDemoDataset(NavDemoDataset):
    def __init__(
        self, scene_ids, rgb_image_dir=None, rgb_ft_dir=None, 
        depth_ft_dir=None, meta_dir=None, semseg_dir=None,
        rgb_image_size=224, max_steps=500, inflection_weight=None,
        num_ft_views=1, use_thda=True, map_dir=None, map_image_size=240,
        num_history_fts=None, num_history_ft_gap=1, num_actions=6, **kwargs
    ) -> None:
        super().__init__(
            scene_ids, rgb_image_dir=rgb_image_dir, rgb_ft_dir=rgb_ft_dir,
            depth_ft_dir=depth_ft_dir, meta_dir=meta_dir, semseg_dir=semseg_dir,
            rgb_image_size=rgb_image_size, max_steps=max_steps, 
            inflection_weight=inflection_weight, num_history_fts=num_history_fts,
            num_ft_views=num_ft_views, use_thda=use_thda, save_step_idxs=True, 
            **kwargs
        )
        self.map_dir = map_dir
        self.map_image_size = map_image_size
        self.num_history_ft_gap = num_history_ft_gap
        self.num_actions = num_actions

        self.visited_locs_in_full_map = {}
        for scene_id in self.scene_ids:
            self.visited_locs_in_full_map[scene_id] = {}
            with jsonlines.open(os.path.join(map_dir, 'meta_infos', f'{scene_id}.jsonl'), 'r') as f:
                for item in f:
                    self.visited_locs_in_full_map[scene_id][item['episode_id']] = item['visited_locs_in_full_map']

        self.map_envs, self.map_txns = self.build_lmdb_envs(
            os.path.join(map_dir, 'maps'), self.scene_ids
        ) 

    def load_dataset_meta_infos(self):
        super().load_dataset_meta_infos()
        del self.data_idxs

    def __len__(self):
        return self.num_steps

    def load_map_at_step(self, txn, episode_id, step):
        imgs = []
        for map_name in ['obs', 'exp']:
            key = f'{episode_id}_{step}_{map_name}'
            value = txn.get(key.encode('ascii'))
            if value is None:
                img = np.zeros((960, 960), dtype=np.uint8)
                LOGGER.info(f'DATASET ERROR: {key} not exists')
            else:
                img = np.frombuffer(value, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
            imgs.append(img)
        imgs = np.stack(imgs, axis=2)
        return imgs

    def plot_traj_image(self, traj_img, visited_locs_in_full_map, start_step, end_step=None):
        nrows, ncols = traj_img.shape
        if end_step is None:
            end_step = start_step + 1
        for t in range(start_step, end_step):
            r, c = visited_locs_in_full_map[t]
            r = min(max(0, r), nrows - 1)
            c = min(max(0, c), ncols - 1)
            traj_img[max(0, r-1):r+2, max(0, c-1):c+2] = 1 # t + 1
        return traj_img

    def _sample_steps(self, num_steps):
        step_ids = np.arange(num_steps, dtype=np.int32)
        # reverse the order to ensure the last step is selected
        step_ids = step_ids[::-self.num_history_ft_gap]
        step_ids = step_ids[:self.num_history_fts]
        step_ids = step_ids[::-1]
        return step_ids

    def __getitem__(self, item_idx):
        scene_id, data_idx, cur_step = self.step_idxs[item_idx]

        meta = self.load_episode_meta_info(self.meta_info_txns[scene_id], data_idx)
        episode_id = meta['episode_id']

        num_steps = len(meta['compass'])
        start_step = 0
        if num_steps > self.max_steps:
            start_step = num_steps - self.max_steps

        outs = {
            # 'episode_ids': episode_id,
            # 'scene_ids': scene_id,
            'objectgoal': meta['objectgoal'],
            'object_category': meta['object_category'],
        }

        step_ids = self._sample_steps(cur_step + 1)
         # for episode > max_steps, just use the last max_steps steps
        step_ids = step_ids[step_ids >= start_step]
        outs['step_ids'] = step_ids.copy() - start_step
        outs['num_history_fts'] = len(step_ids)

        # relative to the start orientation
        compass = np.array(meta['compass'], dtype=np.float32)[step_ids] # (nsteps, 1)
        # relative to the start position using the start orientation as coordinates
        # habitat-sim axis [-agent_position[2] -z, agent_position[0] x]
        gps = np.array(meta['gps'], dtype=np.float32)[step_ids] # (nsteps, 2)
        # add the <bos> token
        prev_actions = np.array([self.num_actions] + meta['demonstration'], dtype=np.int32)[step_ids] # (nsteps, )
        outs['prev_actions'] = prev_actions

        if self.rgb_ft_dir is not None:
            rgb_fts = self.load_episode_fts(
                self.rgb_ft_txns[scene_id], episode_id
            )
            outs['rgb_features'] = rgb_fts[step_ids]
        if self.depth_ft_dir is not None:
            depth_fts = self.load_episode_fts(
                self.depth_ft_txns[scene_id], episode_id
            )
            outs['depth_features'] = depth_fts[step_ids]

        visited_locs_in_full_map = self.visited_locs_in_full_map[scene_id][episode_id]
        map_imgs = self.load_map_at_step(
            self.map_txns[scene_id], episode_id, cur_step, 
        )
        traj_img = np.zeros(map_imgs.shape[:-1], dtype=np.uint8)
        traj_img = self.plot_traj_image(traj_img, visited_locs_in_full_map, 0, cur_step+1)
        map_imgs = np.concatenate([map_imgs, traj_img[:, :, None]], axis=2)
        outs['maps'] = crop_centered_local_map(
            visited_locs_in_full_map[cur_step], map_imgs, 
            (self.map_image_size, self.map_image_size), 
            keep_centered_when_out_of_boundary=True
        )

        outs['compass_features'] = np.concatenate([np.sin(compass), np.cos(compass)], axis=1)
        agent_loc = np.array(visited_locs_in_full_map[cur_step]) # (x, y)
        agent_loc = 0.05 * (agent_loc - 480) # each grid is 5cm
        outs['gps_features'] = np.stack([gps[:, 0], -gps[:, 1]], axis=1) # relative to the start position (meters)
        # print('agent_loc', agent_loc, 'gps', outs['gps_features'][-1]) # should be similar
        outs['gps_features'] = outs['gps_features'] - agent_loc # relative to the agent location

        outs['demonstration'] = meta['demonstration'][cur_step]

        outs['inflection_weight'] = 1
        if self.inflection_weight is not None:
            if cur_step == 0 or meta['demonstration'][cur_step] != meta['demonstration'][cur_step - 1]:
                outs['inflection_weight'] = self.inflection_weight
        if self.kwargs.get('stop_weight', None) is not None and outs['demonstration'] == 0:
            outs['inflection_weight'] = self.kwargs['stop_weight']
            
        return outs


def map_step_collate_fn(inputs):
    batch = {
        k: [x[k] for x in inputs] for k in inputs[0].keys()
    }

    # max_steps = np.max(batch['num_history_fts'])
    # batch['num_history_fts'] = torch.LongTensor(batch['num_history_fts'])
    
    # Do not pad the features here to save batch memory
    # for key in ['rgb_features', 'depth_features', 'compass_features', 'gps_features']:
    #     if key in batch:
    #         # (batch, max_steps, dim_ft)
    #         batch[key] = pad_tensors(
    #             [torch.FloatTensor(x) for x in batch[key]]
    #         )
    # batch['step_ids'] = pad_sequence(
    #     [torch.LongTensor(x) for x in batch['step_ids']], 
    #     batch_first=True, padding_value=0
    # )
    # batch['prev_actions]

    batch['maps'] = torch.from_numpy(np.stack(batch['maps'], 0)) # (batch, h, w, 3) uint8

    batch['objectgoal'] = torch.LongTensor(batch['objectgoal'])
    batch['demonstration'] = torch.LongTensor(batch['demonstration'])
    batch['inflection_weight'] = torch.FloatTensor(batch['inflection_weight'])
    return batch


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    input_dir = 'data/datasets/objectnav/mp3d_70k_demos'
    map_dir = 'data/datasets/objectnav/mp3d_70k_demo_maps_48m'
    scene_ids = ['17DRP5sb8fy']

    dataset = NavMapStepDemoDataset(
        scene_ids, 
        rgb_ft_dir=os.path.join(input_dir, 'rgb_fts', 'clip'), 
        depth_ft_dir=os.path.join(input_dir, 'depth_fts'), 
        meta_dir=os.path.join(input_dir, 'meta_infos'), 
        map_dir=map_dir, map_image_size=240,
        num_history_fts=100, num_history_ft_gap=1, 
        inflection_weight=3,
    )

    # st = time.time()
    # for i in range(len(dataset)):
    #     outs = dataset[i]
    #     if i in range(3):
    #         for k, v in outs.items():
    #             if isinstance(v, np.ndarray):
    #                 print(k, v.dtype, v.shape)
    #                 print('\t', np.min(v), np.mean(v), np.max(v))
    #             else:
    #                 print(k, type(v), v)
    # print('cost time: %.2fmin' % ((time.time() - st) / 60.))  


    # multi-workers BUG:
    # if the batch_size is large, there is:
    # struct.error: 'i' format requires -2147483648 <= number <= 2147483647
    dataloader = DataLoader(
        dataset, batch_size=256, shuffle=True, 
        num_workers=4, collate_fn=map_step_collate_fn, 
        pin_memory=True, drop_last=False
    )
    print('num batches: %d' % (len(dataloader)))
    st = time.time()
    for n, batch in enumerate(dataloader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.dtype, v.size())
            else:
                print(k, type(v), len(v), type(v[0]))
        if n < 1:
            np.save('batch%02d.npy'%n, batch)
        break
        print()
        
    print('cost time: %.2fmin' % ((time.time() - st) / 60.))    # ~10min for all with 4 workers

