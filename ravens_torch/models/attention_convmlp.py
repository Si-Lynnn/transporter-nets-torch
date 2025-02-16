# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""AttentionConvMLP module."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ravens_torch.utils import utils, MeanMetrics, to_device
from ravens_torch.utils.text import bold
from ravens_torch.utils.utils import apply_rotations_to_tensor
from ravens_torch.models.resnet import ResNet43_8s, ResNet36_4s

from ravens_torch.models import mdn_utils
from ravens_torch.models.conv_mlp import ConvMLP, DeepConvMLP

from einops.layers.torch import Rearrange

# REMOVE BELOW
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


class AttentionConvMLP:
    """AttentionConvMLP module."""

    def __init__(self, in_shape, preprocess, use_mdn=False, verbose=False):
        self.preprocess = preprocess

        resnet = False

        if resnet:
            self.model = DeepConvMLP(in_shape, d_action=6, use_mdn=use_mdn)
        else:
            self.model = ConvMLP(d_action=2, use_mdn=use_mdn)
        self.device = to_device([self.model], "Regression", verbose=verbose)

        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        self.metric = MeanMetrics()
        self.val_metric = MeanMetrics()

        self.loss = nn.MSELoss() if not use_mdn else mdn_utils.mdn_loss

    def forward(self, in_img):
        """Forward pass."""
        input_data = self.preprocess(in_img)
        in_tensor = torch.tensor(
            input_data, dtype=torch.float32).to(self.device)
        output = self.model(in_tensor)
        return output

    def train_block(self, in_img, p):
        output = self.forward(in_img)
        label = torch.tensor(p, dtype=torch.float32).to(self.device)
        loss = self.loss(output, label)
        return loss

    def train(self, in_img, p):
        """Train."""
        self.metric.reset()
        self.train_mode()
        self.optimizer.zero_grad()

        loss = self.train_block(in_img, p)
        loss.backward()
        self.optimizer.step()
        self.metric(loss)

        return np.float32(loss.detach().cpu().numpy())

    def test(self, in_img, p):
        """Test."""
        self.eval_mode()

        with torch.no_grad():
            loss = self.train_block(in_img, p)

        return np.float32(loss.detach().cpu().numpy())

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def load(self, path, verbose=False):
        if verbose:
            device = "GPU" if self.device.type == "cuda" else "CPU"
            print(
                f"Loading {bold('attention')} model on {bold(device)} from {bold(path)}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, filename, verbose=False):
        if verbose:
            print(f"Saving attention model to {bold(filename)}")
        torch.save(self.model.state_dict(), filename)
