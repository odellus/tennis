import torch
import torch.nn as nn
import torch.nn.function as F

from utils import load_cfg, device

class Policy(nn.Module):

    def __init__(self, state_size, action_size, cfg):
        super(Policy, self).__init_()

        self.state_size = state_size
        self.action_size = action_size
        self.fc1_size = cfg["Policy"]["fc1_size"]
        self.fc2_size = cfg["Policy"]["fc2_size"]

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(x)
