import sys
import numpy as np
import torch
import cv2

sys.path.append('/home/shichen/codes/objgoalnav/ViT-Adapter/segmentation//')

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes

MP3D_CATEGORIES = [
    'chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa', 'bed', 'chest_of_drawers',
    'plant', 'sink', 'toilet', 'stool', 'towel', 'tv_monitor', 'shower', 'bathtub',
    'counter', 'fireplace', 'gym_equipment', 'seating', 'clothes', 
    # 'foodstuff', 'stationery', 'fruit', 'plaything', 
    # 'hand_tool', 'game_equipment', 'kitchenware',
]

mp3d_to_ade20k = {
    'chair': ['chair', 'armchair', 'swivel chair'], 
    'table': ['table', 'desk', 'pool table', 'coffee table'],
    'picture': ['painting'], 
    'cabinet': ['cabinet'], 
    'cushion': ['cushion', 'pillow'], 
    'sofa': ['sofa'], 
    'bed': ['bed '], 
    'chest_of_drawers': ['wardrobe', 'chest of drawers'],
    'plant': ['plant', 'flower'], 
    'sink': ['sink'], 
    'toilet': ['toilet'], 
    'stool': ['stool'], 
    'towel': ['towel'], 
    'tv_monitor': ['screen', 'crt screen', 'monitor'], 
    'shower': ['shower'], 
    'bathtub': ['bathtub'],
    'counter': ['counter', 'countertop'], 
    'fireplace': ['fireplace'], 
    'gym_equipment': ['minibike', 'bicycle', ], 
    'seating': ['seat', 'bench'], 
    'clothes': ['apparel'],
}
ade20k_to_mp3d = {}
for k, vs in mp3d_to_ade20k.items():
    for v in vs:
        ade20k_to_mp3d[v] = k

class SegmentationModel(object):
    def __init__(self, device) -> None:
        self.load_pretrained_semantic_models(device)

        self.category_mapping = np.zeros((len(self.classes), ), dtype=np.int64) - 1
        for i, label in enumerate(self.classes):
            if label in ade20k_to_mp3d:
                self.category_mapping[i] = MP3D_CATEGORIES.index(
                    ade20k_to_mp3d[label]
                )
        
    def load_pretrained_semantic_models(self, device='cuda:0'):
        config_file = '/home/shichen/codes/objgoalnav/ViT-Adapter/segmentation/configs/ade20k/mask2former_beit_adapter_large_640_160k_ade20k_ss.py'
        checkpoint_file = '/home/shichen/codes/objgoalnav/ViT-Adapter/segmentation/released/mask2former_beit_adapter_large_640_160k_ade20k.pth.tar'

        palette = 'ade20k'

        model = init_segmentor(config_file, checkpoint=None, device=device)
        checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = get_classes(palette)

        self.model = model
        self.classes = model.CLASSES

    def predict_target(self, img, depth_img, target_obj, outfile=None):
        # should convert image from RGB to BGR
        img = torch.flip(img, (2, ))
        result = inference_segmentor(self.model, img.data.cpu().numpy())
    
        seg_masks = self.category_mapping[result[0]]
        seg_tgt_masks = seg_masks == target_obj.item()

        area, med_dist = 0, np.inf
        
        # print(np.unique(seg_masks))
        if np.sum(seg_tgt_masks) > 0:
            area = np.sum(seg_tgt_masks) / np.prod(seg_masks.shape)
            dists = depth_img[seg_tgt_masks]
            # print('dists', dists)
            med_dist = torch.median(dists).item()
            med_dist = med_dist * 4.5 + 0.5

        if outfile is not None:
            cv2.imwrite(outfile, result[0])

        return area, med_dist