"""
taken from https://github.com/rlcode/per/blob/master/prioritized_memory.py
"""
import random
import numpy as np
from sum_tree import SumTree

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, seed):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        random.seed(seed)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        # print(self.tree.total())
        sampling_probabilities = priorities / self.tree.total()
        # print(sampling_probabilities)
        # print(self.tree.n_entries)
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight_max = max(is_weight.max(), 1e-10)
        if is_weight.max() <= 1e-10:
            print(is_weight.max(), is_weight_max)
        is_weight /= is_weight_max
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
