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
            fc2_units = cfg["Model"]["fc2_size_actor"]
            self.weight_init_lim = cfg["Model"]["weight_init_lim"]

            # Seed RNG
            self.seed = torch.manual_seed(seed)

            # Create layers
            self.bn1 = nn.BatchNorm1d(state_size)
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)
            self.reset_parameters()

        def reset_parameters(self):
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-self.weight_init_lim, self.weight_init_lim)
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()
            self.bn3.reset_parameters()

        def forward(self, state):
            """Build an actor policy that maps states to actions."""
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            x = F.relu(self.fc1(self.bn1(state)))
            x = F.relu(self.fc2(self.bn2(x)))
            return torch.tanh(self.fc3(self.bn3(x)))

class Critic(nn.Module):
    """Critic value model."""

    def __init__(self, state_size, action_size, seed, cfg):
        """Initialize parameters and build model.
        """
        super(Critic, self).__init__()

        # Set up configuration
        fcs1_units = cfg["Model"]["fcs1_size_critic"]
        fc2_units = cfg["Model"]["fc2_size_critic"]
        self.weight_init_lim = cfg["Model"]["weight_init_lim"]

        # Seed the RNG
        self.seed = torch.manual_seed(seed)

        # Create layers
        self.bns1 = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn2 = nn.BatchNorm1d(fcs1_units + action_size)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.bn3 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1) # Mapping onto a single scalar value.
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-self.weight_init_lim, self.weight_init_lim)
        self.bns1.reset_parameters()
        self.bn2.reset_parameters()
        self.bn3.reset_parameters()

    def forward(self, state, action):
        """Build a critic value network that maps state-action pairs to Q-values."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        xs = F.relu(self.fcs1(self.bns1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(self.bn2(x)))
        return self.fc3(self.bn3(x))
