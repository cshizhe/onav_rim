from typing import Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from gym import spaces
from habitat import logger
from habitat_baselines.utils.common import Flatten
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder

import clip

class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=128,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint, map_location=torch.device('cpu'))

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                # Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations, no_fc_layer=False):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            obs_depth = observations["depth"]
            if len(obs_depth.size()) == 5:
                observations["depth"] = obs_depth.contiguous().view(
                    -1, obs_depth.size(2), obs_depth.size(3), obs_depth.size(4)
                )
            x = self.visual_encoder(observations)
            x = x.flatten(start_dim=1)

        x = x.detach()
        if no_fc_layer:
            return x

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)

class ResnetRGBEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=256,
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()

        backbone_split = backbone.split("_")
        logger.info("backbone: {}".format(backbone_split))
        make_backbone = getattr(resnet, backbone_split[0])

        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"rgb": observation_space.spaces["rgb"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=make_backbone,
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)
    
    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_rgb = observations["rgb"]
        if len(obs_rgb.size()) == 5:
            observations["rgb"] = obs_rgb.contiguous().view(
                -1, obs_rgb.size(2), obs_rgb.size(3), obs_rgb.size(4)
            )

        if "rgb_features" in observations:
            x = observations["rgb_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)

class ResNetCLIPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        output_size=256,
        pooling='attnpool',
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()

        model, preprocess = clip.load("RN50", device=device)

        # expected input: C x H x W (np.uint8 in [0-255])
        self.preprocess = T.Compose([
            T.Resize(224),
            T.CenterCrop(size=(224, 224)),
            T.ToTensor(),
            # preprocess.transforms[0],  
            # preprocess.transforms[1],
            # # already tensor, but want float
            # T.ConvertImageDtype(torch.float), # if the input is uint8, will convert [0, 255] -> [0, 1]
            # T.Normalize(mean=(0., 0., 0.), std=(255., 255., 255.)),   # [0, 255] -> convert to [0, 1]
            # # normalize with CLIP mean, std
            preprocess.transforms[4],
        ])
        # expected output: C x H x W (np.float32)

        self.backbone = model.visual

        if pooling == 'none':
            self.backbone.attnpool = nn.Identity()
            self.output_shape = (2048, 7, 7)
        elif pooling == 'avgpool':
            self.backbone.attnpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1,1)),
                nn.Flatten()
            )
            self.output_shape = (2048,)
        else:
            self.output_shape = (1024,)

        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if "BatchNorm" in type(module).__name__:
                module.momentum = 0.0
        self.backbone.eval()

        self.visual_fc = nn.Sequential(
            nn.Linear(np.prod(self.output_shape), output_size),
            nn.ReLU(True),
        )
        self.output_shape = (output_size, )

    def forward(self, observations: Dict[str, torch.Tensor], no_fc_layer=False) -> torch.Tensor:  # type: ignore
        for module in self.backbone.modules():
            if "BatchNorm" in type(module).__name__:
                module.eval()

        if "rgb_features" in observations:
            x = observations["rgb_features"]
        else:
            device = observations["rgb"].device
            rgb_observations = observations["rgb"]  # (batch, height, width, channel)

            rgb_observations = rgb_observations.data.cpu().numpy().astype(np.uint8)
            rgb_observations = [self.preprocess(Image.fromarray(rgb_image)) for rgb_image in rgb_observations]
            rgb_observations = torch.stack(rgb_observations, 0).to(device)

            # rgb_observations = rgb_observations.permute(0, 3, 1, 2) # BATCH x CHANNEL x HEIGHT X WIDTH
            # rgb_observations = torch.stack(
            #     [self.preprocess(rgb_image) for rgb_image in rgb_observations]
            # )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            x = self.backbone(rgb_observations).float()

        x = x.detach()

        if no_fc_layer:
            return x

        x = self.visual_fc(x)
        return x

class ResNetImageNetEncoder(nn.Module):
    def __init__(self, observation_space: spaces.Dict, output_size=256, backbone='resnet50'):
        super().__init__()

        self.preprocess = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(size=(224, 224)),
                # T.ConvertImageDtype(torch.float),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.backbone = getattr(models, backbone)(pretrained=True)
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if "BatchNorm" in type(module).__name__:
                module.momentum = 0.0
        self.backbone.eval()

        self.output_shape = (2048,) if backbone != 'resnet18' else 512

        self.visual_fc = nn.Sequential(
            nn.Linear(np.prod(self.output_shape), output_size),
            nn.ReLU(True),
        )
        self.output_shape = (output_size, )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        for module in self.backbone.modules():
            if "BatchNorm" in type(module).__name__:
                module.eval()

        if "rgb_features" in observations:
            x = observations["rgb_features"]
        else:
            device = observations["rgb"].device
            rgb_observations = observations["rgb"]  # (batch, height, width, channel)

            rgb_observations = rgb_observations.data.cpu().numpy().astype(np.uint8)
            rgb_observations = [self.preprocess(Image.fromarray(rgb_image)) for rgb_image in rgb_observations]
            rgb_observations = torch.stack(rgb_observations, 0).to(device)

            # rgb_observations = rgb_observations.permute(0, 3, 1, 2) # BATCH x CHANNEL x HEIGHT X WIDTH
            # rgb_observations = torch.stack(
            #     [self.preprocess(rgb_image) for rgb_image in rgb_observations]
            # )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32

            x = self.backbone(rgb_observations).float()            

        x = self.visual_fc(x)
        return x


class ResnetSemSeqEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=256,
        backbone="resnet18",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
        semantic_embedding_size=4,
        use_pred_semantics=False,
        use_goal_seg=False,
        is_thda=False,
    ):
        super().__init__()
        if not use_goal_seg:
            sem_input_size = 40 + 2
            self.semantic_embedder = nn.Embedding(sem_input_size, semantic_embedding_size)

        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"semantic": observation_space.spaces["semantic"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            sem_embedding_size=semantic_embedding_size,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        self.spatial_output = spatial_output
        self.use_goal_seg = use_goal_seg

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)
    
    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_semantic = observations["semantic"]
        if len(obs_semantic.size()) == 5:
            observations["semantic"] = obs_semantic.contiguous().view(
                -1, obs_semantic.size(2), obs_semantic.size(3), obs_semantic.size(4)
            )

        if "semantic_features" in observations:
            x = observations["semantic_features"]
        else:
            # Embed input when using all object categories
            if not self.use_goal_seg:
                categories = observations["semantic"].long() + 1
                observations["semantic"] = self.semantic_embedder(categories)
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class ResnetEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=256,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=True,
        spatial_output: bool = False,
        sem_embedding_size=4,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            sem_embedding_size=sem_embedding_size,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        obs_rgb = observations["rgb"]
        if len(obs_rgb.size()) == 5:
            observations["rgb"] = obs_rgb.contiguous().view(
                -1, obs_rgb.size(2), obs_rgb.size(3), obs_rgb.size(4)
            )
        obs_depth = observations["depth"]
        if len(obs_rgb.size()) == 5:
            observations["depth"] = obs_depth.contiguous().view(
                -1, obs_depth.size(2), obs_depth.size(3), obs_depth.size(4)
            )
        obs_semantic = observations["semantic"]
        if len(obs_rgb.size()) == 5:
            observations["semantic"] = obs_semantic.contiguous().view(
                -1, obs_semantic.size(2), obs_semantic.size(3), obs_semantic.size(4)
            )

        if "rgb_features" in observations:
            x = observations["rgb_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)
