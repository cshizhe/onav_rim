import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from offline_bc.models.transformer import (
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerEncoder,
)
from offline_bc.models.onav_base import NavILBaseModel
from offline_bc.utils.logger import LOGGER


class ImapEmbedding(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()

        map_cfg = model_config.MAP_ENCODER
        if map_cfg.token_embed_type == 'single':
            self.imap_token_embedding = nn.Embedding(1, model_config.hidden_size)
        elif map_cfg.token_embed_type == 'multi':
            self.imap_token_embedding = nn.Embedding(map_cfg.imap_size**2, model_config.hidden_size)
        else:
            raise NotImplementedError(f'unsupported imap token embed {map_cfg.token_embed_type}')
        
        if map_cfg.encode_position:
            self.imap_pos_fts = self._create_imap_pos_features(map_cfg.imap_size)
            self.imap_pos_layer = nn.Sequential(
                nn.Linear(2, model_config.hidden_size), # (location: x, y)
                nn.LayerNorm(model_config.hidden_size)
            )
            
        self.ft_fusion_layer = nn.Sequential(
            nn.LayerNorm(model_config.hidden_size),
            nn.Dropout(model_config.dropout_rate),
        )

        self.imap_token_type = map_cfg.token_embed_type
        self.imap_num_tokens = map_cfg.imap_size**2
        self.imap_size = map_cfg.imap_size
        self.encode_position = map_cfg.encode_position

    def _create_imap_pos_features(self, imap_size):
        x, y = torch.meshgrid(torch.arange(imap_size), torch.arange(imap_size))
        xy = torch.stack([x, y], dim=2)
        xy = (xy + 0.5 - imap_size / 2).float() # relative distance to the center
        xy = xy.view(-1, 2)
        return xy

    def forward(self, batch_size):
        '''Get the initialized imap embedding
        '''
        device = self.imap_token_embedding.weight.device

        if self.imap_token_type == 'multi':
            token_types = torch.arange(self.imap_num_tokens, dtype=torch.long, device=device)
        else:
            token_types = torch.zeros(self.imap_num_tokens, dtype=torch.long, device=device)

        embeds = self.imap_token_embedding(token_types)
        if self.encode_position:
            pos_embeds = self.imap_pos_layer(self.imap_pos_fts.to(device))
            embeds = embeds + pos_embeds
        else:
            pos_embeds = None
        embeds = self.ft_fusion_layer(embeds)

        embeds = einops.repeat(embeds, 'n d -> b n d', b=batch_size)
        if pos_embeds is not None:
            pos_embeds = einops.repeat(pos_embeds, 'n d -> b n d', b=batch_size)

        return embeds, pos_embeds

class GpsCompassEmbedding(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.gps_layer = nn.Sequential(
            nn.Linear(2, hidden_size), # (location: x, y)
            nn.LayerNorm(hidden_size)
        )
        self.compass_layer = nn.Sequential(
            nn.Linear(2, hidden_size), # (heading: sin, cos)
            nn.LayerNorm(hidden_size)
        )

    def forward(self, gps, compass):
        gps_embeds = self.gps_layer(gps)
        compass_fts = torch.cat(
            [torch.cos(compass), torch.sin(compass)], -1,
        )
        compass_embeds = self.compass_layer(compass_fts)
        embeds = gps_embeds + compass_embeds
        return embeds


class NavImapSingleTransformer(NavILBaseModel):
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
        if model_config.STATE_ENCODER.add_pos_attn:
            self.obs_pos_layer = GpsCompassEmbedding(model_config.hidden_size)

        self.imap_embedding = ImapEmbedding(model_config)

        if model_config.infer_visual_feature_loss > 0:
            dim_vis = 2048  # TODO: CLIP resnet50
            if model_config.infer_depth_feature:
                dim_vis += 2048 # depth resnet50
            self.vis_pred_layer = nn.Sequential(
                nn.Dropout(model_config.dropout_rate),
                nn.Linear(model_config.hidden_size, dim_vis),
            )
        
        if model_config.infer_local_map_loss > 0:
            output_size = model_config.pred_map_nchannels * (model_config.pred_map_image_size**2)
            input_size = model_config.pred_map_input_token_dim * (model_config.MAP_ENCODER.imap_size ** 2)
            self.map_pred_layer = nn.Sequential(
                nn.Linear(model_config.hidden_size, model_config.pred_map_input_token_dim),
                nn.ReLU(),
                nn.Flatten(start_dim=1),
                nn.Dropout(model_config.dropout_rate),
                nn.Linear(input_size, output_size),
                nn.Sigmoid(),
            )

        if 'infer_sem_label_loss' in self.model_config and self.model_config.infer_sem_label_loss > 0:
            # TODO: only valid for concat encoder
            input_size = model_config.RGB_ENCODER.output_size + model_config.DEPTH_ENCODER.output_size
            self.sem_pred_layer = nn.Sequential(
                nn.Dropout(model_config.dropout_rate),
                nn.Linear(input_size, 21*2+1)
            )     

    def forward(self, batch, compute_loss=False, return_imap_embeds=False):
        batch = self.prepare_batch(batch)
        batch_size, num_steps, _ = batch['gps'].size()

        # (batch, num_steps, dim)
        stepid_embeds = self.step_embedding(batch['step_ids'])
        inputs = self.encode_step_obs(batch, step_embeddings=stepid_embeds)
        if isinstance(inputs, dict):
            input_embeds = inputs['fused_embeds']
        else:
            input_embeds = inputs

        imap_embeds, imap_pos_embeds = self.imap_embedding(batch_size)
        if 'init_map_embeds' in batch:
            imap_embeds = batch['init_map_embeds']
            
        if self.model_config.STATE_ENCODER.add_pos_attn:
            obs_pos_embeds = self.obs_pos_layer(batch['gps'], batch['compass'])     

        if compute_loss and self.model_config.infer_visual_feature_loss > 0:
            if self.model_config.encoder_type == 'concat':
                masked_input_embeds = torch.zeros_like(input_embeds)
                dim_vis = self.step_input_sizes['depth'] + self.step_input_sizes['rgb']
                masked_input_embeds[:, :, dim_vis: dim_vis+32] = self.gps_embedding(batch['infer_gps'])
                compass_fts = torch.concat(
                    [torch.cos(batch['infer_compass']), torch.sin(batch['infer_compass'])],
                    -1,
                )
                masked_input_embeds[:, :, dim_vis+32: dim_vis+64] = self.compass_embedding(compass_fts)
            elif self.model_config.encoder_type == 'add':
                masked_input_embeds = self.gps_embedding(batch['infer_gps']) + \
                                      self.compass_embedding(torch.concat([torch.cos(batch['infer_compass']),
                                          torch.sin(batch['infer_compass'])], -1))
                masked_input_embeds = self.ft_fusion_layer(masked_input_embeds)

            if self.model_config.STATE_ENCODER.add_pos_attn:
                masked_obs_pos_embeds = self.obs_pos_layer(
                    batch['infer_gps'], batch['infer_compass']
                )

            target_visual_features = batch['infer_visual_features']
            seq_visual_features = batch['rgb_features']
            if self.model_config.infer_depth_feature:
                seq_visual_features = torch.cat([seq_visual_features, batch['depth_features']], dim=-1)
            seq_visual_features = F.normalize(seq_visual_features, p=2, dim=-1)

            ntokens = imap_embeds.size(1) + 2
            attn_masks = torch.zeros(ntokens, ntokens).bool().to(imap_embeds.device)
            attn_masks[:-1, -1] = True  # imap and obs tokens should not see the masked token
            attn_masks[-1, -2] = True   # the masked token should not see the current observation

        logits = []
        vis_pred_losses, map_pred_losses, sem_pred_losses = [], [], []
        if return_imap_embeds:
            last_imap_embeds = [None for _ in range(batch_size)]
        for t in range(num_steps):
            t_input_embeds = torch.cat(
                [imap_embeds, input_embeds[:, t:t+1]], dim=1
            )
            if self.model_config.STATE_ENCODER.add_pos_attn:
                t_pos_embeds = torch.cat(
                    [imap_pos_embeds, obs_pos_embeds[:, t:t+1]], dim=1
                )
                if compute_loss and self.model_config.infer_visual_feature_loss > 0:
                    t_pos_embeds = torch.cat(
                        [t_pos_embeds, masked_obs_pos_embeds[:, t:t+1]], dim=1
                    )
            else:
                t_pos_embeds = None

            if compute_loss and self.model_config.infer_visual_feature_loss > 0:
                hiddens = self.state_encoder(
                    torch.cat([t_input_embeds, masked_input_embeds[:, t:t+1]], dim=1),
                    pos=t_pos_embeds, mask=attn_masks,
                )
                imap_embeds = hiddens[:, :-2]
                obs_hiddens = hiddens[:, -2]
                masked_obs_hiddens = hiddens[:, -1]
                # (batch_size, dim)
                masked_visual_preds = self.vis_pred_layer(masked_obs_hiddens)
                if self.model_config.infer_visual_feature_loss_type == 'mse':
                    vis_pred_losses.append(F.mse_loss(
                        masked_visual_preds,
                        target_visual_features[:, t], reduction='none'
                    ).mean(dim=1))
                elif self.model_config.infer_visual_feature_loss_type == 'nce':
                    masked_visual_preds = F.normalize(masked_visual_preds, p=2, dim=-1)
                    pos_sim_scores = torch.einsum(
                        'bd,bd->b', F.normalize(target_visual_features[:, t], p=2, dim=-1), 
                        masked_visual_preds
                    )
                    neg_sim_scores = torch.einsum(
                        'btd,bd->bt', seq_visual_features, masked_visual_preds
                    )
                    neg_sim_scores.masked_fill_(batch['demonstration'] == -100, -float('inf'))
                    sim_scores = torch.cat([pos_sim_scores.unsqueeze(1), neg_sim_scores], 1)
                    sim_scores = sim_scores / 0.1
                    # print(t, sim_scores)
                    vis_pred_losses.append(F.cross_entropy(
                        sim_scores, 
                        torch.zeros(sim_scores.size(0), dtype=torch.long, device=sim_scores.device),
                        reduction='none'
                    ))
                    # print(vis_pred_losses[-1])
            else:
                hiddens, layer_attn_weights = self.state_encoder(
                    t_input_embeds, pos=t_pos_embeds, return_attn_weights=True
                )
                imap_embeds = hiddens[:, :-1]
                obs_hiddens = hiddens[:, -1]

            if self.model_config.infer_local_map_loss > 0:
                pred_maps = self.map_pred_layer(imap_embeds).view(
                    batch_size, self.model_config.pred_map_nchannels, -1
                )
                if self.model_config.infer_local_map_loss_type == 'mse':
                    t_map_pred_loss = F.mse_loss(
                        pred_maps, batch['infer_local_maps'][:, t], reduction='none'
                    ).view(batch_size, -1).mean(1)
                elif self.model_config.infer_local_map_loss_type == 'clf':
                    t_map_pred_loss = F.binary_cross_entropy(
                        pred_maps, batch['infer_local_maps'][:, t], reduction='none'
                    ).view(batch_size, -1).mean(1)
                map_pred_losses.append(t_map_pred_loss)

            if 'infer_sem_label_loss' in self.model_config and self.model_config.infer_sem_label_loss > 0:
                # TODO: only valid for concat encoder
                sem_input_hiddens = inputs[:, t, :self.model_config.DEPTH_ENCODER.output_size+self.model_config.RGB_ENCODER.output_size]
                pred_semantics = torch.sigmoid(self.sem_pred_layer(sem_input_hiddens))
                sem_labels = batch['sem_features'][:, t, :-1]
                sem_clf_loss = F.binary_cross_entropy(
                    pred_semantics[:, :21], (sem_labels > 0).float(), reduction='none'
                ).mean(1)
                sem_mse_loss = F.mse_loss(
                    pred_semantics[:, 21:], batch['sem_features'][:, t], reduction='none'
                ).mean(1)
                sem_pred_losses.append(sem_clf_loss + sem_mse_loss)

            add_objgoal = self.model_config.get('encoder_add_objgoal', True)
            if not add_objgoal:
                obs_hiddens = obs_hiddens + inputs['objectgoal']
            step_logits = self.action_distribution(obs_hiddens)
            logits.append(step_logits)

            if return_imap_embeds:
                for ib in range(batch_size):
                    if last_imap_embeds[ib] is None and batch['demonstration'][ib, t] == 0:
                        last_imap_embeds[ib] = imap_embeds[ib].data.cpu().numpy()

        logits = torch.stack(logits, 1)
        
        if compute_loss:
            act_loss = self.compute_loss(logits, batch)
            loss_dict = {'overall': act_loss, 'action_loss': act_loss}
            
            if self.model_config.infer_visual_feature_loss > 0:
                vis_pred_losses = torch.stack(vis_pred_losses, 1) # (batch, nsteps)
                vis_pred_masks = (batch['inflection_weight'] > 0).float()
                vis_pred_loss = torch.mean(
                    torch.sum(vis_pred_losses * vis_pred_masks, 1) / \
                        torch.sum(vis_pred_masks, 1)
                )
                # print(torch.sum(vis_pred_losses * vis_pred_masks, 0))
                loss_dict['overall'] = loss_dict['overall'] + vis_pred_loss * self.model_config.infer_visual_feature_loss
                loss_dict['vis_pred_loss'] = vis_pred_loss

            if self.model_config.infer_local_map_loss > 0:
                map_pred_masks = (batch['inflection_weight'] > 0).float()
                map_pred_losses = torch.stack(map_pred_losses, 1) # (batch, nsteps)
                map_pred_loss = torch.mean(
                    torch.sum(map_pred_losses * map_pred_masks, 1) / \
                        torch.sum(map_pred_masks, 1)
                )
                loss_dict['overall'] = loss_dict['overall'] + map_pred_loss * self.model_config.infer_local_map_loss
                loss_dict['map_pred_loss'] = map_pred_loss

            if 'infer_sem_label_loss' in self.model_config and self.model_config.infer_sem_label_loss > 0:
                sem_pred_losses = torch.stack(sem_pred_losses, 1)
                sem_pred_masks = (batch['inflection_weight'] > 0).float()
                sem_pred_loss = torch.mean(
                    torch.sum(sem_pred_losses * sem_pred_masks, 1) / torch.sum(sem_pred_masks, 1)
                )
                loss_dict['overall'] = loss_dict['overall'] + sem_pred_loss * self.model_config.infer_sem_label_loss
                loss_dict['sem_pred_loss'] = sem_pred_loss

            # for k, v in loss_dict.items():
            #     print(k, v.item())
            return loss_dict, logits

        if return_imap_embeds:
            for ib in range(batch_size):
                if last_imap_embeds[ib] is None:
                    last_imap_embeds[ib] = imap_embeds[ib].data.cpu().numpy()

            return last_imap_embeds

        return logits
