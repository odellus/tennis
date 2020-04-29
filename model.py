# -*- coding: utf-8

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import device

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1./ np.sqrt(fan_in)
    return (-lim, lim)

def norm_init(layer):
    fan_in = layer.weight.data.size()[0]
    return 1./np.sqrt(fan_in)

class Actor(nn.Module):
    """Actor Policy model."""

    def __init__(self, state_size, action_size, seed, cfg):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each state
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        # Set configuration
        fc1_units = cfg["Model"]["fc1_size_actor"]
        self.weight_init_lim = cfg["Model"]["weight_init_lim"]

        # Seed RNG
        self.seed = torch.manual_seed(seed)

        # Create layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-self.weight_init_lim, self.weight_init_lim)

    def forward(self, state):
        """Build an actor policy that maps states to actions."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        return torch.tanh(self.fc2(x))

class ActorNoise(Actor):
    """Parameter noise for Actor policy model."""

    def __init__(self, state_size, action_size, seed, cfg):
        super(ActorNoise, self).__init__(state_size, action_size, seed, cfg)

    def reset_parameters(self):
        self.fc1.weight.data.normal_(std=norm_init(self.fc1))
        self.fc2.weight.data.normal_(std=self.weight_init_lim)

class Critic(nn.Module):
    """Critic value model."""

    def __init__(self, state_size, action_size, seed, cfg):
        """Initialize parameters and build model.
        """
        super(Critic, self).__init__()

        # Set up configuration
        fcs1_units = cfg["Model"]["fcs1_size_critic"]
        fc2_units = cfg["Model"]["fc2_size_critic"]
        fc3_units = cfg["Model"]["fc3_size_critic"]
        self.weight_init_lim = cfg["Model"]["weight_init_lim"]

        # Seed the RNG
        self.seed = torch.manual_seed(seed)

        # Create layers
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        # Mapping onto a single scalar value.
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-self.weight_init_lim, self.weight_init_lim)


    def forward(self, state, action):
        """Build a critic value network that maps state-action pairs to Q-values."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
