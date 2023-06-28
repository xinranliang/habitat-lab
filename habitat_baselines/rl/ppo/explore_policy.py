#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
import torchvision
import torch.nn as nn
import torchvision.models as models

from habitat_baselines.rl.models.resnet import DetectronResNet50
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.common.utils import CategoricalNet


class ExplorePolicy(nn.Module):
    def __init__(self, policy_net, dim_actions):
        super().__init__()

        # CNN encoder to extract features from Policy network, NOT Visual encoder
        self.policy_net = policy_net

        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.policy_net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.policy_net.output_size)

    def forward(self, *x):
        raise NotImplementedError
    
    def forward_visual(self, observations):
        return self.visual_net(observations)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.policy_net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states


    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.policy_net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        if isinstance(observations, dict):
            observations = observations['rgb']
        features, rnn_hidden_states = self.policy_net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action.argmax(dim=1))
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class ExploreBaselinePolicy(ExplorePolicy):
    def __init__(self, 
                observation_space, 
                action_space, 
                hidden_size=512, 
                num_recurrent_layers=1, 
                rnn_type="LSTM", 
                device=torch.device("cpu")):
        super().__init__(
            ExplorePolicyNet(
                observation_space, hidden_size, num_recurrent_layers, rnn_type, device
            ),
            action_space.n
        )

class ExploreBaselinePolicyRollout(ExplorePolicy):
    def __init__(self, 
                action_dim, 
                hidden_size=512, 
                num_recurrent_layers=1, 
                rnn_type="LSTM", 
                device=torch.device("cpu")):
        super().__init__(
            ExplorePolicyNet(
                None, hidden_size, num_recurrent_layers, rnn_type, device
            ),
            action_dim
        )


class ExploreNet(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass


class ExplorePolicyNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, num_recurrent_layers, rnn_type, device):
        super().__init__()

        self._hidden_size = hidden_size

        self.policy_encoder = DetectronResNet50(pretrained=False, downsample=True, device_id=torch.cuda.current_device())
        self.state_encoder = RNNStateEncoder(
            self.policy_encoder.output_size,
            self._hidden_size,
            num_recurrent_layers,
            rnn_type
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        perception_embed = self.policy_encoder(observations)
        x, rnn_hidden_states = self.state_encoder(perception_embed, rnn_hidden_states, masks)

        return x, rnn_hidden_states
