# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Img2Real module."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ravens_torch.utils import utils, MeanMetrics, to_device
from ravens_torch.utils.text import bold
from ravens_torch.utils.utils import apply_rotations_to_tensor

from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
# REMOVE BELOW
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


class Img2Real:
    """Image coordinates to real-world position module."""

    def __init__(self, verbose=False):
        self.model = torch.nn.Sequential(
                    nn.Linear(2, 4),
                    nn.ReLU(),
                    nn.Linear(4, 2)
                )
        self.device = to_device([self.model], "MLP", verbose=verbose)

        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        self.metric = MeanMetrics()

        self.loss = nn.MSELoss()

    def forward(self, uv):
        """Forward pass."""

        in_tensor = torch.tensor(
            uv, dtype=torch.float32).to(self.device)
        output = self.model(in_tensor)
        return output

    def train_block(self, uv, p):
        output = self.forward(uv)
        label = torch.tensor(p, dtype=torch.float32).to(self.device)
        print(output, label)
        loss = self.loss(output, label)
        return loss

    def train(self, uv, p):
        """Train."""
        self.metric.reset()
        self.train_mode()
        self.optimizer.zero_grad()

        loss = self.train_block(uv, p)
        loss.backward()
        self.optimizer.step()
        self.metric(loss)

        return np.float32(loss.detach().cpu().numpy())

    def test_single_action(self,uv):
         """Test."""
         self.eval_mode()

         with torch.no_grad():
            p = self.forward(uv)

         return p

    def test(self, uv, p):
        """Test."""
        self.eval_mode()
        with torch.no_grad():
            loss = self.train_block(uv, p)

        return np.float32(loss.detach().cpu().numpy())

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def load(self, path, verbose=False):
        if verbose:
            device = "GPU" if self.device.type == "cuda" else "CPU"
            print(
                f"Loading {bold('img2real')} model on {bold(device)} from {bold(path)}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, filename, verbose=False):
        if verbose:
            print(f"Saving img2real model to {bold(filename)}")
        torch.save(self.model.state_dict(), filename)
