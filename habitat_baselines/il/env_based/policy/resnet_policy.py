import torch
import torch.nn as nn
import einops

from gym import Space
from habitat import Config, logger

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Policy

from habitat_baselines.il.env_based.policy.onav_vis_models import (
    ObjectNavRNN,
    ObjectNavTransformer,
)
from habitat_baselines.il.env_based.policy.onav_imap_models import (
    ObjectNavImapSingleTransformer
)
from offline_bc.models.onav_base import ClsPrediction


model_factory = {
    'ObjectNavRNN': ObjectNavRNN,
    'ObjectNavTransformer': ObjectNavTransformer,
    'ObjectNavImapSingleTransformer': ObjectNavImapSingleTransformer,
}

@baseline_registry.register_policy
class ObjectNavILPolicy(Policy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config,
        no_critic=True
    ):
        model_class = model_factory[model_config.model_class]
        model = model_class(
            observation_space=observation_space,
            model_config=model_config,
            num_actions=action_space.n,
        )

        if model_config.action_clf_class == 'ClsPrediction':
            clf_net = ClsPrediction(model.output_size, action_space.n)
        elif model_config.action_clf_class == 'linear':
            clf_net = nn.Sequential(
                nn.Dropout(model_config.dropout_rate),
                nn.Linear(model.output_size, action_space.n),
            )
        else:
            clf_net = None
        
        super().__init__(
            model,
            action_space.n,
            no_critic=no_critic,
            clf_net=clf_net
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space, action_space, no_critic=True
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,         
            no_critic=no_critic,   
        )

