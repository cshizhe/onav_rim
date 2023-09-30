import os
import jsonlines
import numpy as np
import cv2
import time

import torch

from offline_bc.utils.logger import LOGGER
from offline_bc.data.dataset import (
    NavDemoDataset, collate_fn
)
from habitat_baselines.common.nav_mapping import get_local_map_boundaries

MAP_IMAGE_SIZE = 480


class NavMapDemoDataset(NavDemoDataset):
    def __init__(
        self, scene_ids, rgb_image_dir=None, rgb_ft_dir=None, 
        depth_ft_dir=None, meta_dir=None, semseg_dir=None,
        rgb_image_size=224, max_steps=500, inflection_weight=None,
        num_ft_views=1, use_thda=True, map_dir=None, map_image_size=240,
        **kwargs
    ) -> None:
        super().__init__(
            scene_ids, rgb_image_dir=rgb_image_dir, rgb_ft_dir=rgb_ft_dir,
            depth_ft_dir=depth_ft_dir, meta_dir=meta_dir, semseg_dir=semseg_dir,
            rgb_image_size=rgb_image_size, max_steps=max_steps, 
            inflection_weight=inflection_weight,
            num_ft_views=num_ft_views, use_thda=use_thda, **kwargs
        )
        self.map_dir = map_dir
        self.map_image_size = map_image_size

        self.visited_locs_in_full_map = {}
        for scene_id in self.scene_ids:
            self.visited_locs_in_full_map[scene_id] = {}
            with jsonlines.open(os.path.join(map_dir, 'meta_infos', f'{scene_id}.jsonl'), 'r') as f:
                for item in f:
                    self.visited_locs_in_full_map[scene_id][item['episode_id']] = item['visited_locs_in_full_map']

        self.map_envs, self.map_txns = self.build_lmdb_envs(
            os.path.join(map_dir, 'maps'), self.scene_ids
        ) 

    def load_map_at_step(self, txn, episode_id, step):
        imgs = []
        for map_name in ['obs', 'exp']:
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

    def plot_traj_image(self, traj_img, visited_locs_in_full_map, start_step, end_step=None):
        if end_step is None:
            end_step = start_step + 1
        for t in range(start_step, end_step):
            r, c = visited_locs_in_full_map[t]
            r = min(max(0, r), MAP_IMAGE_SIZE - 1)
            c = min(max(0, c), MAP_IMAGE_SIZE - 1)
            traj_img[max(0, r-1):r+2, max(0, c-1):c+2] = t + 1
        return traj_img

    def __getitem__(self, data_idx):
        outs = super().__getitem__(data_idx)

        scene_id, idx = self.data_idxs[data_idx]
        meta = self.meta_infos[scene_id][idx]
        episode_id = meta['episode_id']

        visited_locs_in_full_map = self.visited_locs_in_full_map[scene_id][episode_id]
        start_step, end_step = outs['start_step'], outs['end_step']

        # traj_img = np.zeros((MAP_IMAGE_SIZE, MAP_IMAGE_SIZE), dtype=np.float32) # memory costly
        # traj_img = self.plot_traj_image(traj_img, visited_locs_in_full_map, 0, start_step)

        outs['maps'] = []
        for step in range(start_step, end_step):
            map_imgs = self.load_map_at_step(
                self.map_txns[scene_id], episode_id, step, 
            )
            x1, x2, y1, y2 = get_local_map_boundaries(
                visited_locs_in_full_map[step], (self.map_image_size, self.map_image_size),
                (MAP_IMAGE_SIZE, MAP_IMAGE_SIZE), global_downscaling=2
            )

            # traj_img = self.plot_traj_image(traj_img, visited_locs_in_full_map, start_step)
            # normed_traj_img = traj_img / (step + 1)
            # img = np.concatenate([map_imgs, normed_traj_img[:, :, None]], axis=2)[x1:x2, y1:y2]

            img = map_imgs[x1:x2, y1:y2]
            outs['maps'].append(img)
        outs['maps'] = np.stack(outs['maps'], 0)
        outs['visited_locs_in_full_map'] = visited_locs_in_full_map[:end_step]

        return outs


def map_collate_fn(inputs):
    batch = collate_fn(inputs)
    
    # (batch x nsteps, 240, 240, 3)
    batch['maps'] = np.concatenate(batch['maps'], 0)
    
    return batch


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    input_dir = 'data/datasets/objectnav/mp3d_70k_demos'
    map_dir = 'data/datasets/objectnav/mp3d_70k_demo_maps'
    scene_ids = ['vyrNrziPKCB']

    dataset = NavMapDemoDataset(
        scene_ids, 
        rgb_ft_dir=os.path.join(input_dir, 'rgb_fts', 'resnet50'), 
        depth_ft_dir=os.path.join(input_dir, 'depth_fts'), 
        meta_dir=os.path.join(input_dir, 'meta_infos'), 
        rgb_image_size=224, map_dir=map_dir, map_image_size=240,
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
    # print('cost time: %.2fmin' % ((time.time() - st) / 60.))    # ~6min for all


    # multi-workers BUG:
    # if the batch_size is large, there is:
    # struct.error: 'i' format requires -2147483648 <= number <= 2147483647
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, 
        num_workers=4, collate_fn=map_collate_fn, 
        pin_memory=True, drop_last=False
    )
    st = time.time()
    for n, batch in enumerate(dataloader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.dtype, v.size())
            else:
                print(k, type(v), len(v), type(v[0]))
        # if n < 3:
        #     np.save('batch%02d.npy'%n, batch)
        print()
    print('cost time: %.2fmin' % ((time.time() - st) / 60.))    # ~2min for all with 4 workers

