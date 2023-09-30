import torch
import torch.nn as nn
import einops

from habitat_baselines.il.env_based.policy.onav_base import ObjectNavBase
from habitat_baselines.il.env_based.policy.onav_vis_models import ObjectNavRNN

from offline_bc.models.transformer import (
    PositionalEncoding, 
    TransformerEncoderLayer,
    TransformerEncoder
)
from offline_bc.models.onav_imap_models import (
    ImapEmbedding, GpsCompassEmbedding
)


class ObjectNavImapSingleTransformer(ObjectNavRNN):

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
                max_len=2000 #model_config.STATE_ENCODER.max_steps,
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
        if model_config.STATE_ENCODER.add_pos_attn:
            self.obs_pos_layer = GpsCompassEmbedding(model_config.hidden_size)

        self.imap_embedding = ImapEmbedding(model_config)

        subgoal_pred_type = getattr(model_config, 'subgoal_pred_type', None)
        if subgoal_pred_type == 'coord':
            self.subgoal_pred_layer = nn.Sequential(
                nn.Dropout(model_config.dropout_rate),
                nn.Linear(model_config.hidden_size, 3 + 22 + 2),
            )
        elif subgoal_pred_type == 'heatmap':
            x, y = torch.meshgrid(
                torch.arange(model_config.subgoal_goto_heatmap_size), 
                torch.arange(model_config.subgoal_goto_heatmap_size)
            )
            xy = torch.stack([x, y], dim=2)
            xy = (xy + 0.5 - model_config.subgoal_goto_heatmap_size / 2).float() # relative distance to the center
            xy = xy.view(-1, 2) # (heatmapsize**2, 2)
            self.subgoal_goto_heatmap_locs = xy * model_config.subgoal_goto_heatmap_grid_meter
            self.subgoal_pred_layer = nn.Sequential(
                nn.Dropout(model_config.dropout_rate),
                nn.Linear(model_config.hidden_size, 3 + 22 + model_config.subgoal_goto_heatmap_size**2),
            )    
        else:
            self.subgoal_pred_layer = None        

    def forward(self, batch, recursive_states, prev_actions, step_id, pred_subgoal=False):
        batch_size = batch['gps'].size(0)
        device = batch['gps'].device

        step_ids = torch.zeros(batch_size).long().to(device) + step_id
        stepid_embeds = self.step_embedding(step_ids)

        # (batch_size, hidden_size)
        t_inputs = self.encode_step_obs(
            batch, prev_actions=prev_actions, 
            step_embeddings=stepid_embeds
        )
        t_embeds = t_inputs['fused_embeds']

        if recursive_states is None:
            imap_embeds, imap_pos_embeds = self.imap_embedding(batch_size)
        elif recursive_states[1] is None:
            _, imap_pos_embeds = self.imap_embedding(batch_size)
            imap_embeds = recursive_states[0]
        else:
            imap_embeds, imap_pos_embeds = recursive_states
        
        if self.model_config.STATE_ENCODER.add_pos_attn:
            obs_pos_embeds = self.obs_pos_layer(batch['gps'], batch['compass'])
            t_pos_embeds = torch.cat(
                [imap_pos_embeds, obs_pos_embeds.unsqueeze(1)], dim=1
            )
        else:
            t_pos_embeds = None

        hiddens, layer_attn_weights = self.state_encoder(
            torch.cat([imap_embeds, t_embeds.unsqueeze(1)], dim=1), 
            pos=t_pos_embeds, return_attn_weights=True, 
        )

        imap_embeds = hiddens[:, :-1]
        obs_hiddens = hiddens[:, -1]
        add_objgoal = self.model_config.get('encoder_add_objgoal', True)
        if not add_objgoal:
            obs_hiddens = obs_hiddens + t_inputs['objectgoal']

        return obs_hiddens, (imap_embeds, imap_pos_embeds)
