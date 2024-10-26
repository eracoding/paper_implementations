"""
The code in this file is based on https://github.com/ml-jku/hopfield-layers.
This repo accompanies Ramsauer et al. (2020) (https://arxiv.org/abs/2008.02217).
"""
import math
import torch
import torch.nn as nn
import numpy as np
from itertools import product
from scipy.special import softmax
from tqdm import tqdm


class ClassicalHopfield:
    """class providing the classical hopfield model which can learn patterns and retrieve them"""

    def __init__(self, pat_size):
        """Constructor of classical Hopfield network

        Args:
            pat_size (int): the dimension d of a single pattern vector
        """
        self.size = pat_size
        self.W = torch.zeros((self.size, self.size))
        self.b = torch.zeros((self.size, 1))

    def learn(self, patterns):
        """learns the weight matrix by applying hebbian learning rule to the provided patterns, without self reference

        Args:
            patterns ([torch.Tensor]): list of dx1 torch tensors, column-wise
        """
        self.num_pat = len(patterns)
        assert all(isinstance(x, torch.Tensor) for x in patterns), 'not all input patterns are torch tensors'
        assert all(len(x.shape) == 2 for x in patterns), 'not all input patterns have dimension 2'
        assert all(1 == x.shape[1] for x in patterns), f'not all input patterns have shape ({self.size},1)'

        n = len(patterns)
        W = torch.zeros(self.W.shape)

        for pat in tqdm(patterns):
            W += torch.matmul(pat, pat.T)  # w_ij += pat_i * pat_j

        W /= n
        W.fill_diagonal_(0)
        self.W = W

    def retrieve(self, test_pat, reps=1):
        """Retrieves a memorized pattern from the provided one

        Args:
            test_pat (torch.Tensor): the partial/masked test pattern which should be retrieved
            reps (int, optional): number of times the retrieval update should be applied. Defaults to 1.

        Returns:
            torch.Tensor: the retrieved pattern of shape dx1
        """
        assert isinstance(test_pat, torch.Tensor), 'input pattern is not a torch tensor'
        assert test_pat.shape == (self.size, 1), f'input pattern has wrong shape {test_pat.shape}'

        for _ in tqdm(range(reps)):
            # synchronous update, aka use old state for all updates 
            reconst = torch.matmul(self.W, test_pat)  # reconst_i = \sum_j w_ij * partial_pattern_j
            test_pat = torch.where(reconst > self.b, 1, -1)

        return test_pat

    def energy(self, pattern):
        """calculates energy for a pattern according to hopfield model"""
        assert isinstance(pattern, torch.Tensor)
        return -0.5 * torch.matmul(torch.matmul(pattern.T, self.W), pattern) + torch.matmul(self.b.T, pattern)

    def energy_landscape(self):
        """print out all vectors of the input space with their energy"""
        for pat in product([1, -1], repeat=self.size):
            pat = torch.tensor(pat, dtype=torch.float32).view(-1, 1)
            print(f"energy({pat.tolist()})={self.energy(pat).item():.3f}")


class DenseHopfield:
    def __init__(self, pat_size, beta=1, normalization_option=1):
        self.size = pat_size
        self.beta = beta
        self.max_norm = math.sqrt(self.size)
        if normalization_option == 0:
            self.energy = self.energy_unnormalized
        elif normalization_option == 1:
            self.energy = self.energy_normalized
        elif normalization_option == 2:
            self.energy = self.energy_normalized2
        else:
            raise ValueError(f'Unknown option for normalization: {normalization_option}')

    def learn(self, patterns):
        """expects patterns as torch tensors and stores them col-wise in pattern matrix"""
        self.num_pat = len(patterns)
        assert all(isinstance(x, torch.Tensor) for x in patterns), 'not all input patterns are torch tensors'
        assert all(len(x.shape) == 2 for x in patterns), 'not all input patterns have dimension 2'
        assert all(1 == x.shape[1] for x in patterns), 'not all input patterns have shape (-1,1)'
        self.patterns = torch.stack([p.squeeze(-1) for p in patterns], dim=1)
        self.max_pat_norm = max(torch.norm(p).item() for p in patterns)

    def retrieve(self, partial_pattern, max_iter=float('inf'), thresh=0.5):
        if partial_pattern.size != self.size:
            raise ValueError(f"Input pattern {partial_pattern} does not match state size: {len(partial_pattern)} vs {self.size}")

        if None in partial_pattern:
            raise NotImplementedError("None elements not supported")

        assert isinstance(partial_pattern, torch.Tensor), 'test pattern was not torch tensor'
        assert len(partial_pattern.shape) <= 2 and 1 == partial_pattern.shape[1], f'test pattern with shape {partial_pattern.shape} is not a col-vector'

        pat_old = partial_pattern.clone()
        iters = 0

        for _ in tqdm(range(int(max_iter))):
            pat_new = torch.zeros_like(pat_old)
            for jj in range(self.size):
                E = 0
                temp = pat_old[jj].clone()
                pat_old[jj] = +1
                E -= self.energy(pat_old)
                pat_old[jj] = -1
                E += self.energy(pat_old)

                pat_old[jj] = temp
                pat_new[jj] = torch.where(E > 0, 1, -1)

            if torch.count_nonzero(pat_old != pat_new) <= thresh:
                break
            else:
                pat_old = pat_new

        return pat_new

    @staticmethod
    def _lse(z, beta):
        return 1 / beta * torch.logsumexp(beta * z, dim=0)

    def energy_unnormalized(self, pattern):
        return -1 * torch.sum(torch.exp(torch.matmul(self.patterns.T, pattern)))

    def energy_normalized(self, pattern):
        return -1 * torch.sum(torch.exp(torch.matmul(self.patterns.T, pattern) / self.max_norm))

    def energy_normalized2(self, pattern):
        exponents = torch.matmul(self.patterns.T, pattern)
        norm_exponents = exponents - self.max_pat_norm
        norm_exponents[norm_exponents < -73] = -float('inf')
        return -1 * torch.sum(torch.exp(norm_exponents))

    def energy_landscape(self):
        for pat in product([1, -1], repeat=self.size):
            pat = torch.tensor(pat, dtype=torch.float32).view(-1, 1)
            print(f"energy({pat.tolist()})={self.energy(pat).item():.3f}")


class ContinuousHopfield:
    def __init__(self, pat_size, beta=1, do_normalization=True):
        self.size = pat_size
        self.beta = beta
        self.max_norm = math.sqrt(self.size)
        if do_normalization:
            self.softmax = self.softmax_normalized
            self.energy = self.energy_normalized
        else:
            self.softmax = self.softmax_unnormalized
            self.energy = self.energy_unnormalized

    def learn(self, patterns):
        self.num_pat = len(patterns)
        assert all(isinstance(x, torch.Tensor) for x in patterns), 'not all input patterns are torch tensors'
        assert all(len(x.shape) == 2 for x in patterns), 'not all input patterns have dimension 2'
        assert all(1 == x.shape[1] for x in patterns), 'not all input patterns have shape (-1,1)'
        self.patterns = torch.stack([p.squeeze(-1) for p in patterns], dim=1)
        self.M = max(torch.norm(vec).item() for vec in patterns)

    def retrieve(self, partial_pattern, max_iter=float('inf'), thresh=0.5):
        if partial_pattern.size != self.size:
            raise ValueError(f"Input pattern {partial_pattern} does not match state size: {len(partial_pattern)} vs {self.size}")

        assert isinstance(partial_pattern, torch.Tensor), 'test pattern was not torch tensor'
        assert len(partial_pattern.shape) == 2 and 1 == partial_pattern.shape[1], f'test pattern with shape {partial_pattern.shape} is not a col-vector'

        pat_old = partial_pattern.clone()
        iters = 0

        while iters < max_iter:
            pat_new = torch.matmul(self.patterns, self.softmax(self.beta * torch.matmul(self.patterns.T, pat_old)))
            if torch.count_nonzero(pat_old != pat_new) <= thresh:
                break
            else:
                pat_old = pat_new
            iters += 1

        return pat_new

    def softmax_unnormalized(self, z):
        return torch.softmax(z, dim=0)

    def softmax_normalized(self, z):
        return torch.softmax(z / self.max_norm, dim=0)

    @staticmethod
    def _lse(z, beta):
        return 1 / beta * torch.logsumexp(beta * z, dim=0)

    def energy_unnormalized(self, pattern):
        return -1 * self._lse(torch.matmul(self.patterns.T, pattern), 1) + 0.5 * torch.matmul(pattern.T, pattern) \
            + 1 / self.beta * math.log(self.num_pat) + 0.5 * self.M ** 2

    def energy_normalized(self, pattern):
        return -1 * self._lse((torch.matmul(self.patterns.T, pattern)) / self.max_norm, 1) + 0.5 * torch.matmul(pattern.T, pattern) \
            + 1 / self.beta * math.log(self.num_pat) + 0.5 * self.M ** 2

    def energy_landscape(self):
        for pat in product([1, -1], repeat=self.size):
            pat = torch.tensor(pat, dtype=torch.float32).view(-1, 1)
            print(f"energy({pat.tolist()})={self.energy(pat).item():.3f}")
