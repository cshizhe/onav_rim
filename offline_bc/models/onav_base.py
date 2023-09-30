import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from gym import spaces

from habitat.tasks.nav.object_nav_task import (
    task_cat2mpcat40,
    mapping_mpcat40_to_goal21
)
from habitat_baselines.il.common.encoders.resnet_encoders import (
    VlnResnetDepthEncoder,
    ResnetRGBEncoder,
    ResnetSemSeqEncoder,
    ResNetImageNetEncoder,
    ResNetCLIPEncoder,
)

from offline_bc.utils.logger import LOGGER
from offline_bc.utils.ops import pad_tensors_wgrad
from offline_bc.utils.ops import pad_tensors, gen_seq_masks


class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, output_size, input_size=None, dropout_rate=0):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 nn.LayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.net(x)


class NavILBaseModel(nn.Module):
    r"""Reimplementation of habitat_baselines.il.env_based.policy.resnet_policy.ObjectNavILPolicy
    """

    def __init__(self, model_config, device):
        super().__init__()
        self.model_config = model_config
        self.device = device

        self.step_input_sizes = {}
        self.build_model(model_config)

    def build_model(self, model_config):
        # encoders
        if model_config.encoder_type.startswith('concat'):
            self.build_encoders_concat(model_config)
        elif model_config.encoder_type == 'add':
            self.build_encoders_add(model_config)
        else:
            raise NotImplementedError('unsupported encoder type: %s' % (model_config.encoder_type))

        # action predictor
        if model_config.action_clf_class == 'ClsPrediction':
            self.action_distribution = ClsPrediction(
                self.output_size, model_config.num_actions, 
                dropout_rate=model_config.dropout_rate
            )
        else:
            self.action_distribution = nn.Sequential(
                nn.Dropout(model_config.dropout_rate),
                nn.Linear(self.output_size, model_config.num_actions),
            )

    def build_visual_encoders(self, model_config):

        observation_space = spaces.Dict({
            'depth': spaces.Box(low=0., high=1., shape=(480, 640, 1), dtype=np.float32),
            'rgb': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            'semantic': spaces.Box(low=0, high=4294967295, shape=(480, 640), dtype=np.uint32),
        })

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder",
            "None",
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        if model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
                trainable=model_config.DEPTH_ENCODER.trainable,
            )
            self.step_input_sizes['depth'] = model_config.DEPTH_ENCODER.output_size * model_config.num_ft_views
        else:
            self.depth_encoder = None

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "ResnetRGBEncoder",
            "resnet50_imagenet",
            "resnet18_imagenet",
            "resnet50_clip",
            "None",
        ], "RGB_ENCODER.cnn_type must be 'ResnetRGBEncoder'."

        if model_config.RGB_ENCODER.cnn_type == "ResnetRGBEncoder":
            self.rgb_encoder = ResnetRGBEncoder(
                observation_space,
                output_size=model_config.RGB_ENCODER.output_size,
                backbone=model_config.RGB_ENCODER.backbone,
                trainable=model_config.RGB_ENCODER.train_encoder,
                normalize_visual_inputs=model_config.RGB_ENCODER.normalize_visual_inputs,
            )
            self.step_input_sizes['rgb'] = model_config.RGB_ENCODER.output_size * model_config.num_ft_views
        elif model_config.RGB_ENCODER.cnn_type in ["resnet50_imagenet", 'resnet18_imagenet']:
            self.rgb_encoder = ResNetImageNetEncoder(
                observation_space,
                output_size=model_config.RGB_ENCODER.output_size,
                backbone=model_config.RGB_ENCODER.cnn_type.split('_')[0],
            )
            self.step_input_sizes['rgb'] = model_config.RGB_ENCODER.output_size * model_config.num_ft_views
        elif model_config.RGB_ENCODER.cnn_type.startswith("resnet50_clip"):
            self.rgb_encoder = ResNetCLIPEncoder(
                observation_space,
                output_size=model_config.RGB_ENCODER.output_size,
                pooling='avgpool',
                device=self.device
            )
            self.step_input_sizes['rgb'] = model_config.RGB_ENCODER.output_size * model_config.num_ft_views
        else:
            self.rgb_encoder = None
            LOGGER.info("RGB encoder is none")

        sem_seg_output_size = 0
        self.semantic_predictor = None
        self.is_thda = False
        if model_config.USE_SEMANTICS:
            sem_embedding_size = model_config.SEMANTIC_ENCODER.embedding_size

            self.is_thda = model_config.SEMANTIC_ENCODER.is_thda
            sem_spaces = {
                "semantic": spaces.Box(
                    low=0,
                    high=255,
                    shape=(480, 640, sem_embedding_size),
                    dtype=np.uint8,
                ),
            }
            sem_obs_space = spaces.Dict(sem_spaces)
            self.sem_seg_encoder = ResnetSemSeqEncoder(
                sem_obs_space,
                output_size=model_config.SEMANTIC_ENCODER.output_size,
                backbone=model_config.SEMANTIC_ENCODER.backbone,
                trainable=model_config.SEMANTIC_ENCODER.train_encoder,
                semantic_embedding_size=sem_embedding_size,
                is_thda=self.is_thda
            )
            sem_seg_output_size = model_config.SEMANTIC_ENCODER.output_size
            LOGGER.info("Setting up Sem Seg model")
            self.step_input_sizes['sem'] = sem_seg_output_size

            self.embed_sge = model_config.embed_sge
            if self.embed_sge:
                self.task_cat2mpcat40 = torch.tensor(task_cat2mpcat40)
                self.mapping_mpcat40_to_goal = np.zeros(
                    max(
                        max(mapping_mpcat40_to_goal21.keys()) + 1,
                        50,
                    ),
                    dtype=np.int8,
                )

                for key, value in mapping_mpcat40_to_goal21.items():
                    self.mapping_mpcat40_to_goal[key] = value
                self.mapping_mpcat40_to_goal = torch.tensor(self.mapping_mpcat40_to_goal)
                self.step_input_sizes['sem_sge'] = 1
        
    def build_encoders_concat(self, model_config):
        '''Habitat-web baseline: concatenate the embeded features
        '''
        self.build_visual_encoders(model_config)

        if hasattr(model_config, 'SEM_FT_ENCODER') and model_config.SEM_FT_ENCODER.type is not None:
            self.sem_ft_layer = nn.Linear(
                model_config.SEM_FT_ENCODER.input_size,
                model_config.SEM_FT_ENCODER.output_size
            )
            self.step_input_sizes['sem'] = model_config.SEM_FT_ENCODER.output_size
            LOGGER.info("\n\nSetting up Semantic sensor")
        else:
            self.sem_ft_layer = None

        if model_config.USE_GPS:
            self.gps_embedding = nn.Linear(2, 32)
            self.step_input_sizes['gps'] = 32
            LOGGER.info("\n\nSetting up GPS sensor")
        
        if model_config.USE_COMPASS:
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(input_compass_dim, self.compass_embedding_dim)
            self.step_input_sizes['compass'] = 32
            LOGGER.info("\n\nSetting up Compass sensor")

        self._n_object_categories = 28
        LOGGER.info("Object categories: {}".format(self._n_object_categories))
        if self.model_config.obj_embed_file is None:
            self.obj_category_to_embeds = None
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
        else:
            obj_embeds = np.load(self.model_config.obj_embed_file, allow_pickle=True).item()
            self.obj_category_to_embeds = {
                k: torch.from_numpy(v.astype(np.float32)).to(self.device) \
                    for k, v in obj_embeds.items()
            }
            self.obj_categories_embedding = nn.Linear(
                list(obj_embeds.values())[0].shape[0], 32
            )
        self.step_input_sizes['objectgoal'] = 32
        LOGGER.info("\n\nSetting up Object Goal sensor")

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(model_config.num_actions + 1, 32)
            self.step_input_sizes['prev_action'] = self.prev_action_embedding.embedding_dim

        LOGGER.info('step input size: %d' % (np.sum(list(self.step_input_sizes.values()))))

        if model_config.encoder_type == 'concat_linear':
            step_input_size = np.sum(list(self.step_input_sizes.values()))
            self.ft_fusion_layer = nn.Linear(step_input_size, self.output_size)
        else:
            self.ft_fusion_layer = None

    def build_encoders_add(self, model_config):
        '''Typical approach in transformer: add all the embeded features
        '''
        hidden_size = model_config.hidden_size

        model_config.defrost()
        model_config.DEPTH_ENCODER.output_size = hidden_size
        model_config.RGB_ENCODER.output_size = hidden_size
        model_config.freeze()

        # RGB and depth encoder
        self.build_visual_encoders(model_config)
        self.rgb_layer_norm = nn.LayerNorm(hidden_size)
        self.depth_layer_norm = nn.LayerNorm(hidden_size)

        if hasattr(model_config, 'SEM_FT_ENCODER') and model_config.SEM_FT_ENCODER.type is not None:
            self.sem_ft_layer = nn.Sequential(
                nn.Linear(
                    model_config.SEM_FT_ENCODER.input_size,
                    model_config.SEM_FT_ENCODER.hidden_size
                ),
                nn.LayerNorm(hidden_size)
            )
            self.step_input_sizes['sem'] = hidden_size
        else:
            self.sem_ft_layer = None

        if model_config.USE_GPS:
            self.gps_embedding = nn.Sequential(
                nn.Linear(2, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            self.step_input_sizes['gps'] = hidden_size
        
        if model_config.USE_COMPASS:
            # cos and sin of the heading angle
            self.compass_embedding = nn.Sequential(
                nn.Linear(2, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            self.step_input_sizes['compass'] = hidden_size

        self._n_object_categories = 28
        LOGGER.info("Object categories: {}".format(self._n_object_categories))
        if self.model_config.obj_embed_file is None:
            self.obj_category_to_embeds = None
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, hidden_size
            )
        else:
            obj_embeds = np.load(self.model_config.obj_embed_file, allow_pickle=True).item()
            self.obj_category_to_embeds = {
                k: torch.from_numpy(v.astype(np.float32)).to(self.device) \
                    for k, v in obj_embeds.items()
            }
            self.obj_categories_embedding = nn.Linear(
                iter(obj_embeds.values()).shape[0], hidden_size
            )
        self.step_input_sizes['objectgoal'] = hidden_size

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(model_config.num_actions + 1, hidden_size)
            self.step_input_sizes['prev_action'] = hidden_size

        self.ft_fusion_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(model_config.dropout_rate),
        )

    @property
    def output_size(self):
        return self.model_config.hidden_size

    @property
    def num_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            nweights += np.prod(v.size())
            nparams += 1
        return nweights, nparams

    @property
    def num_trainable_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            if v.requires_grad:
                nweights += np.prod(v.size())
                nparams += 1
        return nweights, nparams

    def _extract_sge(self, batch):
        # recalculating to keep this self-contained instead of depending on training infra
        if "semantic" in batch and "objectgoal" in batch:
            # (batch x nsteps, height x width x 1)
            obj_semantic = batch["semantic"].contiguous().flatten(start_dim=1)
            
            object_goals = []
            for i, og in enumerate(batch['objectgoal'].data.cpu().numpy()):
                object_goals.extend([og] * batch['num_steps'][i])
            object_goals = torch.from_numpy(np.array(object_goals))

            idx = self.task_cat2mpcat40[object_goals]
            if self.is_thda:
                idx = self.mapping_mpcat40_to_goal[idx].long()
            idx = idx.to(obj_semantic.device).unsqueeze(1)

            goal_visible_pixels = (obj_semantic == idx).sum(dim=1)
            goal_visible_area = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1)).float()
            return goal_visible_area.unsqueeze(-1)

    def prepare_batch(self, batch):
        for key in ['rgb_features', 'depth_features', 'sem_features', \
                    'compass', 'gps', 
                    'infer_gps', 'infer_compass', 'infer_visual_features']:
            if key in batch:
                # (batch, max_steps, dim_ft)
                batch[key] = pad_tensors(
                    [torch.FloatTensor(x) for x in batch[key]]
                )
        
        if 'infer_local_maps' in batch:
            batch['infer_local_maps'] = pad_tensors(
                [x.float() for x in batch['infer_local_maps']]
            )

        for k, v in batch.items():
            if k in ['rgb']:
                batch[k] = torch.from_numpy(v).to(self.device).float()
            elif k in ['semantic']:
                batch[k] = torch.from_numpy(v).to(self.device)
            else:
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
        return batch

    def encode_step_obs_concat(self, batch, step_embeddings=None):
        batch_size, num_steps, _ = batch['gps'].size()

        x = []
        if self.depth_encoder is not None:
            depth_embedding = self.depth_encoder(batch)
            if self.model_config.num_ft_views > 1: # (batch, nsteps, nviews, dim)
                depth_embedding = depth_embedding.view(batch_size, num_steps, -1)
            x.append(depth_embedding)

        if self.rgb_encoder is not None:
            rgb_embedding = self.rgb_encoder(batch)
            if len(rgb_embedding.size()) == 2:  # (batch x nsteps, dim)
                rgb_embedding = torch.split(rgb_embedding, batch['num_steps'], dim=0)
                rgb_embedding = pad_tensors_wgrad(rgb_embedding)
            if self.model_config.num_ft_views > 1: # (batch, nsteps, nviews, dim)
                rgb_embedding = rgb_embedding.view(batch_size, num_steps, -1)
            x.append(rgb_embedding)

        if self.sem_ft_layer is not None:
            sem_embedding = self.sem_ft_layer(batch['sem_features'])
            x.append(sem_embedding)

        if self.model_config.USE_SEMANTICS:
            # batch["semantic"]: (num_steps x num_envs, h, w, 1)
            if self.embed_sge:
                sge_embedding = self._extract_sge(batch)
                if len(sge_embedding.size()) == 2:
                    sge_embedding = torch.split(sge_embedding, batch['num_steps'], dim=0)
                    sge_embedding = pad_tensors_wgrad(sge_embedding)
                x.append(sge_embedding)

            batch['semantic'] = batch['semantic'].squeeze(dim=3)
            sem_seg_embedding = self.sem_seg_encoder(batch)
            if len(sem_seg_embedding.size()) == 2:  # (batch x nsteps, dim)
                sem_seg_embedding = torch.split(sem_seg_embedding, batch['num_steps'], dim=0)
                sem_seg_embedding = pad_tensors_wgrad(sem_seg_embedding)
            x.append(sem_seg_embedding)

        if self.model_config.USE_GPS:
            x.append(self.gps_embedding(batch['gps']))
        
        if self.model_config.USE_COMPASS:
            compass_observations = torch.concat(
                [torch.cos(batch['compass']), torch.sin(batch['compass'])],
                -1,
            )
            compass_embedding = self.compass_embedding(compass_observations)
            x.append(compass_embedding)

        if self.obj_category_to_embeds is None:
            obj_embedding = self.obj_categories_embedding(batch['objectgoal'])
        else:
            obj_embedding = torch.stack(
                [self.obj_category_to_embeds[k] for k in batch['object_category']], 0
            )
            obj_embedding = self.obj_categories_embedding(obj_embedding)

        x.append(
            einops.repeat(
                obj_embedding, 'b d -> b t d', t=num_steps
            )
        )

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions = torch.zeros(batch_size, num_steps, dtype=torch.long).to(self.device)
            prev_actions[:, 0] = -1
            prev_actions[:, 1:] = batch['demonstration'][:, :-1]
            prev_actions[prev_actions == -100] = -1
            prev_actions_embedding = self.prev_action_embedding(
                prev_actions + 1
            )
            x.append(prev_actions_embedding)
        
        x = torch.cat(x, dim=2)
        if self.ft_fusion_layer is not None:
            x = self.ft_fusion_layer(x)

        if step_embeddings is not None:
            x = x + step_embeddings
        
        return x

    def encode_step_obs_add(self, batch, step_embeddings=None):
        x = {}

        if self.depth_encoder is not None:
            depth_embedding = self.depth_encoder(batch)
            depth_embedding = self.depth_layer_norm(depth_embedding)
            x['depth'] = depth_embedding

        if self.rgb_encoder is not None:
            rgb_embedding = self.rgb_encoder(batch)
            rgb_embedding = self.rgb_layer_norm(rgb_embedding)
            x['rgb'] = rgb_embedding

        if self.sem_ft_layer is not None:
            sem_embedding = self.sem_ft_layer(batch['sem_features'])
            x['sem'] = sem_embedding

        if self.model_config.USE_GPS:
            x['gps'] = self.gps_embedding(batch['gps'])
        
        if self.model_config.USE_COMPASS:
            compass_observations = torch.concat(
                [torch.cos(batch['compass']), torch.sin(batch['compass'])],
                -1,
            )
            compass_embedding = self.compass_embedding(compass_observations)
            x['compass'] = compass_embedding

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions = torch.zeros_like(batch['demonstration'])
            prev_actions[:, 0] = -1
            prev_actions[:, 1:] = batch['demonstration'][:, :-1]
            prev_actions[prev_actions == -100] = -1
            prev_actions_embedding = self.prev_action_embedding(
                prev_actions + 1
            )
            x['prev_action'] = prev_actions_embedding

        # (batch_size, nsteps, hidden_size)
        fused_embeds = torch.sum(torch.stack(list(x.values()), 0), 0)

        if self.obj_category_to_embeds is None:
            obj_embedding = self.obj_categories_embedding(batch['objectgoal'])
        else:
            obj_embedding = torch.stack(
                [self.obj_category_to_embeds[k] for k in batch['object_category']], 0
            )
            obj_embedding = self.obj_categories_embedding(obj_embedding)
        x['objectgoal'] = obj_embedding

        add_objgoal = self.model_config.get('encoder_add_objgoal', True)
        if add_objgoal:
            fused_embeds = fused_embeds + x['objectgoal'].unsqueeze(1)
        
        if step_embeddings is not None:
            fused_embeds = fused_embeds + step_embeddings
        fused_embeds = self.ft_fusion_layer(fused_embeds)
        x['fused_embeds'] = fused_embeds

        return x

    def encode_step_obs(self, batch, **kwargs):
        if self.model_config.encoder_type.startswith('concat'):
            return self.encode_step_obs_concat(batch, **kwargs)
        elif self.model_config.encoder_type == 'add':
            return self.encode_step_obs_add(batch, **kwargs)

    def compute_loss(self, logits, batch):
        losses = F.cross_entropy(
            logits.permute(0, 2, 1),
            batch['demonstration'],
            reduction='none', ignore_index=-100
        )   # (N, T)
        loss = torch.mean(
            torch.sum(losses * batch['inflection_weight'], 1) / \
                torch.sum(batch['inflection_weight'], 1)
        )
        return loss

    def forward(self, batch, compute_loss=False):
        raise NotImplementedError

