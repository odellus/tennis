# -*- coding: utf-8

import random
import copy
import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, cfg):
        """Initialize parameters and noise process."""
        # Unpack configuration.
        mu = cfg["Noise"]["Mu"]
        theta = cfg["Noise"]["Theta"]
        sigma = cfg["Noise"]["Sigma"]
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma

        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """"Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=x.shape)
        # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
