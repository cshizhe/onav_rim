import torch
import torch.nn as nn


from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.il.env_based.policy.onav_base import ObjectNavBase

from offline_bc.models.transformer import (
    triangular_mask, PositionalEncoding,
    TransformerEncoderLayer,
    TransformerEncoder,
)


class ObjectNavRNN(ObjectNavBase):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """
    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def output_size(self):
        return self.model_config.hidden_size

    def build_model(self, observation_space, model_config, num_actions):
        self.build_encoders(observation_space, model_config, num_actions)
        
        self.state_encoder = RNNStateEncoder(
            input_size=self.output_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.STATE_ENCODER.num_recurrent_layers,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        x = self.encode_step_obs(observations, prev_actions=prev_actions)['fused_embeds']
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


class ObjectNavTransformer(ObjectNavBase):

    @property
    def output_size(self):
        return self.model_config.hidden_size

    def build_model(self, observation_space, model_config, num_actions):
        self.build_encoders(observation_space, model_config, num_actions)

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

    def forward(self, batch, history_embeds, prev_actions, step_id):
        batch_size = batch['gps'].size(0)
        device = batch['gps'].device

        step_ids = torch.zeros(batch_size).long().to(device) + step_id
        step_embeddings = self.step_embedding(step_ids)

        # (batch_size, hidden_size)
        inputs = self.encode_step_obs(
            batch, prev_actions=prev_actions, 
            step_embeddings=step_embeddings
        )
        embeds = inputs['fused_embeds']

        if history_embeds is None:
            history_embeds = embeds.unsqueeze(1)
        else:
            history_embeds = torch.cat([history_embeds, embeds.unsqueeze(1)], dim=1)
        
        causal_attn_masks = triangular_mask(
            history_embeds.size(1), device, diagonal_shift=1
        )
        hiddens = self.state_encoder(
            history_embeds, mask=causal_attn_masks, 
        )

        hiddens = hiddens[:, -1]
        add_objgoal = self.model_config.get('encoder_add_objgoal', True)
        if not add_objgoal:
            hiddens = hiddens + inputs['objectgoal']

        return hiddens, history_embeds

