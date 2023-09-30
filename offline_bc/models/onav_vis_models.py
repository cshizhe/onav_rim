import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder

from offline_bc.models.transformer import (
    triangular_mask, PositionalEncoding,
    TransformerEncoderLayer,
    TransformerEncoder,
)
from offline_bc.models.onav_base import NavILBaseModel
from offline_bc.utils.logger import LOGGER



class NavILRecurrentNet(NavILBaseModel):
    r"""Reimplementation of habitat_baselines.il.env_based.policy.resnet_policy.ObjectNavILPolicy
    """

    def build_model(self, model_config):
        super().build_model(model_config)
        self.state_encoder = RNNStateEncoder(
            input_size=model_config.hidden_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.STATE_ENCODER.num_recurrent_layers,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, batch, compute_loss=False):
        batch = self.prepare_batch(batch)
        batch_size, num_steps, _ = batch['gps'].size()

        inputs = self.encode_step_obs(batch)
        if isinstance(inputs, dict):
            input_embeds = inputs['fused_embeds']
        else:
            input_embeds = inputs

        init_hidden_states = torch.zeros(
            self.num_recurrent_layers,
            batch_size,
            self.model_config.hidden_size
        ).to(self.device)

        x, _ = self.state_encoder(input_embeds, init_hidden_states, seq_single_episode=True)

        add_objgoal = self.model_config.get('encoder_add_objgoal', True)
        if not add_objgoal:
            x = x + inputs['objectgoal'].unsqueeze(1)
        logits = self.action_distribution(x)    # (N, T, D)

        if compute_loss:
            loss = self.compute_loss(logits, batch)
            return loss, logits

        return logits


class NavILTransformer(NavILBaseModel):
    r"""Reimplementation of habitat_baselines.il.env_based.policy.resnet_policy.ObjectNavILPolicy
    """
    def build_model(self, model_config):
        super().build_model(model_config)

        if model_config.STATE_ENCODER.learnable_step_embedding:
            self.step_embedding = nn.Embedding(
                model_config.STATE_ENCODER.max_steps,
                model_config.hidden_size,
            )
        else:
            self.step_embedding = PositionalEncoding(
                model_config.hidden_size,
                max_len=model_config.STATE_ENCODER.max_steps,
            )

        encoder_layer = TransformerEncoderLayer(
            model_config.hidden_size, 
            model_config.STATE_ENCODER.num_attention_heads, 
            dim_feedforward=model_config.STATE_ENCODER.intermediate_size, 
            dropout=model_config.STATE_ENCODER.dropout_prob,
            activation=model_config.STATE_ENCODER.hidden_act, 
            normalize_before=False
        )
        self.state_encoder = TransformerEncoder(
            encoder_layer, 
            model_config.STATE_ENCODER.num_hidden_layers, 
            norm=None, batch_first=True
        )

    def forward(self, batch, compute_loss=False):
        batch = self.prepare_batch(batch)
        batch_size, num_steps, _ = batch['gps'].size()

        # (batch, num_steps, dim)
        stepid_embeds = self.step_embedding(batch['step_ids'])
        inputs = self.encode_step_obs(batch, step_embeddings=stepid_embeds)
        if isinstance(inputs, dict):
            input_embeds = inputs['fused_embeds']
        else:
            input_embeds = inputs

        # (len_target, len_source): True denotes disallowing attention
        causal_attn_masks = triangular_mask(num_steps, self.device, diagonal_shift=1)
        hiddens = self.state_encoder(
            input_embeds, mask=causal_attn_masks, 
        )

        add_objgoal = self.model_config.get('encoder_add_objgoal', True)
        if not add_objgoal:
            hiddens = hiddens + inputs['objectgoal'].unsqueeze(1)
        logits = self.action_distribution(hiddens)    # (N, T, D)

        if compute_loss:
            loss = self.compute_loss(logits, batch)
            return loss, logits

        return logits
