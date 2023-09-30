import numpy as np

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as T

import clip
from habitat_baselines.il.common.encoders.resnet_encoders import VlnResnetDepthEncoder
from habitat_baselines.il.env_based.policy.rednet import load_rednet


class CLIPEncoder(object):
    def __init__(self, device, batch_size=64) -> None:
        # load clip model
        clip_model, clip_preprocess = clip.load('RN50', device=device)
        clip_model = clip_model.visual
        clip_model.attnpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        clip_model.eval()
        clip_preprocess = T.Compose([
            T.ToTensor(),
            clip_preprocess.transforms[4] # normalize
        ])
        self.model = clip_model
        self.preprocess = clip_preprocess
        self.device = device
        self.batch_size = batch_size

    def extract_fts(self, rgb_images):
        '''
        images: list of PIL.Image (preprocessed 224 x 224 x 3)
        '''
        fts = []
        for i in range(0, len(rgb_images), self.batch_size):
            # (batch, 3, 224, 224)
            inputs = torch.stack(
                [self.preprocess(x) for x in rgb_images[i: i+self.batch_size]], 0
            ).to(self.device)
            # (batch, 2048)
            fts.append(self.model(inputs).data.cpu().numpy())
        fts = np.concatenate(fts, 0)
        return fts

class ResnetEncoders(object):
    def __init__(self, device, backbones=['resnet50'], batch_size=64) -> None:
        self.device = device
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.models = {
            backbone: getattr(torchvision.models, backbone)(pretrained=True).to(self.device) \
                for backbone in backbones
        }
        for name, model in self.models.items():
            model.fc = nn.Identity()
            model.eval()
        self.batch_size = batch_size

    def extrat_fts(self, rgb_images):
        '''
        images: list of PIL.Image (preprocessed 224 x 224 x 3)
        '''
        fts = {name: [] for name in self.models.keys()}
        for name, model in self.models.items():
            for i in range(0, len(rgb_images), self.batch_size):
                # (batch, 3, 224, 224)
                inputs = torch.stack(
                    [self.preprocess(x) for x in rgb_images[i: i+self.batch_size]], 0
                ).to(self.device)
                # (batch, 2048)
                fts[name].append(model(inputs).data.cpu().numpy())
        fts = {name: np.concatenate(value, 0) for name, value in fts.items()}
        return fts

class DepthEncoder(object):
    def __init__(self, device, observation_space, ckpt_file, backbone, batch_size=64) -> None:
        depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=256, # useless
            checkpoint=ckpt_file,
            backbone=backbone,
            trainable=False,
        )
        depth_encoder.eval()
        self.model = depth_encoder.to(device)
        self.device = device
        self.batch_size = batch_size
    
    def extract_fts(self, depth_images):
        '''
        depth_images: Tensor (batch, height, width, 1)
        '''
        fts = []
        for i in range(0, len(depth_images), self.batch_size):
            ft = self.model.visual_encoder(
                {'depth': depth_images[i: i+self.batch_size]}
            ) # (batch, 128, 4, 4)
            fts.append(ft.flatten(start_dim=1).data.cpu().numpy())
        fts = np.concatenate(fts, 0)
        return fts

class SemanticPredictor(object):
    def __init__(self, device, ckpt_file, num_classes, batch_size=64) -> None:
        semantic_predictor = load_rednet(
            device,
            ckpt=ckpt_file,
            resize=True, # since we train on half-vision
            num_classes=num_classes
        )
        semantic_predictor.eval()
        self.model = semantic_predictor
        self.device = device
        self.batch_size = batch_size

    def predict(self, rgb_images, depth_images):
        preds = []
        for i in range(0, len(rgb_images), self.batch_size):
            preds.append(self.model(
                rgb_images[i: i+self.batch_size], depth_images[i: i+self.batch_size]
            ).data.cpu().numpy())
        preds = np.concatenate(preds, 0)
        return preds

