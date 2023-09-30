import numpy as np
import torch
import torch.nn as nn

from gym import Space
from gym import spaces
from habitat import Config, logger

from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
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
from habitat_baselines.rl.ppo import Net


class ObjectNavBase(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions, device=None):
        super().__init__()
        self.model_config = model_config
        self.device = device
        
        if isinstance(model_config.num_ft_views, str):
            self.model_config.defrost()
            self.model_config.num_ft_views = int(self.model_config.num_ft_views.split('_')[0]) # 2_up
            self.model_config.freeze()

        self.step_input_sizes = {}
        self.build_model(observation_space, model_config, num_actions)

        self.train()

    def build_visual_encoders(self, observation_space, model_config):
        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder",
            "None", None
        ], "DEPTH_ENCODER.cnn_type %s must be VlnResnetDepthEncoder" % (model_config.DEPTH_ENCODER.cnn_type)
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
                normalize_visual_inputs=model_config.normalize_visual_inputs,
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
            logger.info("RGB encoder is none")

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
            logger.info("Setting up Sem Seg model")
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
        
    def build_object_category_encoder(self, model_config, hidden_size):
        self._n_object_categories = 28
        logger.info("Object categories: {}".format(self._n_object_categories))
        obj_embed_file = getattr(model_config, 'obj_embed_file', None)
        if obj_embed_file is None:
            self.obj_category_to_embeds = None
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, hidden_size
            )
        else:
            obj_embeds = np.load(obj_embed_file, allow_pickle=True).item()
            category_to_task_category_id = {'chair': 0, 'table': 1, 'picture': 2, 'cabinet': 3, 'cushion': 4, 'sofa': 5, 'bed': 6, 'chest_of_drawers': 7, 'plant': 8, 'sink': 9, 'toilet': 10, 'stool': 11, 'towel': 12, 'tv_monitor': 13, 'shower': 14, 'bathtub': 15, 'counter': 16, 'fireplace': 17, 'gym_equipment': 18, 'seating': 19, 'clothes': 20}

            self.obj_category_to_embeds = {
                category_to_task_category_id[k]: torch.from_numpy(v.astype(np.float32)) \
                    for k, v in obj_embeds.items() if k in category_to_task_category_id
            }
            self.obj_categories_embedding = nn.Linear(
                list(obj_embeds.values())[0].shape[0], hidden_size
            )
        self.step_input_sizes['objectgoal'] = hidden_size

    def build_encoders_concat(self, observation_space, model_config, num_actions):
        '''Habitat-web baseline: concatenate the embeded features
        '''
        self.build_visual_encoders(observation_space, model_config)

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            self.step_input_sizes['gps'] = 32
            logger.info("\n\nSetting up GPS sensor")
        
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(input_compass_dim, self.compass_embedding_dim)
            self.step_input_sizes['compass'] = 32
            logger.info("\n\nSetting up Compass sensor")

        self.build_object_category_encoder(model_config, 32)

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            self.step_input_sizes['prev_action'] = self.prev_action_embedding.embedding_dim

        logger.info('step input size: %d' % (np.sum(list(self.step_input_sizes.values()))))

        if model_config.encoder_type == 'concat_linear':
            step_input_size = np.sum(list(self.step_input_sizes.values()))
            self.ft_fusion_layer = nn.Linear(step_input_size, self.output_size)
        else:
            self.ft_fusion_layer = None

    def build_encoders_add(self, observation_space, model_config, num_actions):
        '''Typical approach in transformer: add all the embeded features
        '''
        hidden_size = model_config.hidden_size

        model_config.defrost()
        model_config.DEPTH_ENCODER.output_size = hidden_size
        model_config.RGB_ENCODER.output_size = hidden_size
        model_config.freeze()

        # RGB and depth encoder
        self.build_visual_encoders(observation_space, model_config)
        self.rgb_layer_norm = nn.LayerNorm(hidden_size)
        self.depth_layer_norm = nn.LayerNorm(hidden_size)

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Sequential(
                nn.Linear(input_gps_dim, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            self.step_input_sizes['gps'] = hidden_size
            logger.info("\n\nSetting up GPS sensor")
        
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Sequential(
                nn.Linear(input_compass_dim, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            self.step_input_sizes['compass'] = hidden_size
            logger.info("\n\nSetting up Compass sensor")

        self.build_object_category_encoder(model_config, hidden_size)

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, hidden_size)
            self.step_input_sizes['prev_action'] = hidden_size

        self.ft_fusion_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(model_config.dropout_rate),
        )

    def build_encoders(self, observation_space, model_config, num_actions):
        # encoders
        if model_config.encoder_type.startswith('concat'):
            self.build_encoders_concat(observation_space, model_config, num_actions)
        elif model_config.encoder_type == 'add':
            self.build_encoders_add(observation_space, model_config, num_actions)

    def build_model(self, observation_space, model_config, num_actions):
        raise NotImplementedError

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind and self.depth_encoder.is_blind

    def _extract_sge(self, observations):
        # recalculating to keep this self-contained instead of depending on training infra
        if "semantic" in observations and "objectgoal" in observations:
            obj_semantic = observations["semantic"].contiguous().flatten(start_dim=1)
            
            if len(observations["objectgoal"].size()) == 3:
                observations["objectgoal"] = observations["objectgoal"].contiguous().view(
                    -1, observations["objectgoal"].size(2)
                )

            idx = self.task_cat2mpcat40[
                observations["objectgoal"].long()
            ]
            if self.is_thda:
                idx = self.mapping_mpcat40_to_goal[idx].long()
            idx = idx.to(obj_semantic.device)

            if len(idx.size()) == 3:
                idx = idx.squeeze(1)

            goal_visible_pixels = (obj_semantic == idx).sum(dim=1)
            goal_visible_area = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1)).float()
            return goal_visible_area.unsqueeze(-1)

    def encode_visual_obs(self, observations):
        rgb_obs = observations.get("rgb", None)
        depth_obs = observations.get("depth", None)

        outs = {}

        if self.depth_encoder is not None:
            if depth_obs is not None and len(depth_obs.size()) == 5:  # (num_steps, num_envs, h, w, c)
                observations["depth"] = depth_obs.contiguous().view(
                    -1, depth_obs.size(2), depth_obs.size(3), depth_obs.size(4)
                )
            depth_embedding = self.depth_encoder(observations)
            if self.model_config.num_ft_views > 1: # multiple cameras
                depth_embedding = [depth_embedding]
                for k in range(1, self.model_config.num_ft_views):
                    depth_embedding.append(self.depth_encoder({'depth': observations[f'depth_{k}']}))
                depth_embedding = torch.cat(depth_embedding, -1)
            outs['depth'] = depth_embedding

        if self.rgb_encoder is not None:
            if rgb_obs is not None and len(rgb_obs.size()) == 5:    # (num_steps, num_envs, h, w, c)
                observations["rgb"] = rgb_obs.contiguous().view(
                    -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
                )
            rgb_embedding = self.rgb_encoder(observations)
            if self.model_config.num_ft_views > 1: # multiple cameras
                rgb_embedding = [rgb_embedding]
                for k in range(1, self.model_config.num_ft_views):
                    rgb_embedding.append(self.rgb_encoder({'rgb': observations[f'rgb_{k}']}))
                rgb_embedding = torch.cat(rgb_embedding, -1)
            outs['rgb'] = rgb_embedding

        if self.model_config.USE_SEMANTICS:
            semantic_obs = observations["semantic"] # (num_steps, num_envs, h, w)
            if len(semantic_obs.size()) == 4:
                observations["semantic"] = semantic_obs.contiguous().view(
                    -1, semantic_obs.size(2), semantic_obs.size(3)
                )
            if self.embed_sge:
                sge_embedding = self._extract_sge(observations)
                outs['sem_sge'] = sge_embedding

            sem_seg_embedding = self.sem_seg_encoder(observations)
            outs['sem'] = sem_seg_embedding

        return outs

    def encode_step_obs_concat(self, observations, prev_actions=None, step_embeddings=None, offline=False):
        outs = self.encode_visual_obs(observations)
        embeds = []
        for key in ['depth', 'rgb', 'sem', 'sem_sge']:
            if key in outs:
                embeds.append(outs[key])
        device = embeds[-1].device

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]  # (num_steps, num_envs, 2)
            if len(obs_gps.size()) == 3 and (not offline):
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            outs['gps'] = self.gps_embedding(obs_gps)
            embeds.append(outs['gps'])
        
        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]  # (num_steps, num_envs, 1)
            if len(obs_compass.size()) == 3 and (not offline):
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            if offline:
                compass_observations = torch.concat(
                    [torch.cos(obs_compass), torch.sin(obs_compass)], -1,
                )
                compass_embedding = self.compass_embedding(compass_observations)
            else:
                compass_observations = torch.stack(
                    [torch.cos(obs_compass), torch.sin(obs_compass)], -1,
                )
                compass_embedding = self.compass_embedding(compass_observations.squeeze(dim=1))
            outs['compass'] = compass_embedding
            embeds.append(outs['compass'])

        object_goal = observations[ObjectGoalSensor.cls_uuid].long()  # (num_steps, num_envs, 1)
        if len(object_goal.size()) == 3 and (not offline):
            object_goal = object_goal.contiguous().view(-1, object_goal.size(2))
        if self.obj_category_to_embeds is None:
            if offline:
                embeds.append(self.obj_categories_embedding(object_goal))
            else:
                embeds.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))
        else:
            obj_embedding = torch.stack(
                [self.obj_category_to_embeds[k] for k in object_goal[..., 0].data.cpu().numpy()], 0
            )
            embeds.append(self.obj_categories_embedding(obj_embedding.to(device)))

        if self.model_config.SEQ2SEQ.use_prev_action:
            if offline:
                prev_actions_embedding = self.prev_action_embedding(
                    prev_actions + 1
                )   # (batch_size, num_steps)
            else:
                prev_actions_embedding = self.prev_action_embedding(
                    (prev_actions + 1).view(-1)
                )  # prev_actions: (num_steps, num_envs, 1)
            embeds.append(prev_actions_embedding)
        
        if offline:
            for i, x in enumerate(embeds):
                if len(x.size()) != 3: # (3, (4, 32))
                    print(i, x.size())
        embeds = torch.cat(embeds, dim=-1)
        if self.ft_fusion_layer is not None:
            embeds = self.ft_fusion_layer(embeds)

        if step_embeddings is not None:
            embeds = embeds + step_embeddings

        outs['fused_embeds'] = embeds
        
        return outs

    def encode_step_obs_add(self, observations, prev_actions=None, step_embeddings=None):
        outs = self.encode_visual_obs(observations)

        if self.depth_encoder is not None:
            outs['depth'] = self.depth_layer_norm(outs['depth'])

        if self.rgb_encoder is not None:
            outs['rgb'] = self.rgb_layer_norm(outs['rgb'])

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]  # (num_steps, num_envs, 2)
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            outs['gps'] = self.gps_embedding(obs_gps)
        
        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]  # (num_steps, num_envs, 1)
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            compass_observations = torch.stack(
                [torch.cos(obs_compass), torch.sin(obs_compass)], -1,
            )
            compass_embedding = self.compass_embedding(compass_observations.squeeze(dim=1))
            outs['compass'] = compass_embedding

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                (prev_actions + 1).view(-1)
            )  # prev_actions: (num_steps, num_envs, 1)
            outs['prev_action'] = prev_actions_embedding

        # (batch_size, hidden_size)
        fused_embeds = torch.sum(torch.stack(list(outs.values()), 0), 0)

        object_goal = observations[ObjectGoalSensor.cls_uuid].long()  # (num_steps, num_envs, 1)
        if len(object_goal.size()) == 3:
            object_goal = object_goal.contiguous().view(-1, object_goal.size(2))
        outs['objectgoal'] = self.obj_categories_embedding(object_goal).squeeze(dim=1)

        add_objgoal = self.model_config.get('encoder_add_objgoal', True)
        if add_objgoal:
            fused_embeds = fused_embeds + outs['objectgoal']

        if step_embeddings is not None:
            fused_embeds = fused_embeds + step_embeddings

        fused_embeds = self.ft_fusion_layer(fused_embeds)
        outs['fused_embeds'] = fused_embeds

        return outs

    def encode_step_obs(self, observations, **kwargs):
        if self.model_config.encoder_type.startswith('concat'):
            return self.encode_step_obs_concat(observations, **kwargs)
        elif self.model_config.encoder_type == 'add':
            return self.encode_step_obs_add(observations, **kwargs)

    def forward(self, observations, hidden_states, prev_actions, nav_step):
        raise NotImplementedError

