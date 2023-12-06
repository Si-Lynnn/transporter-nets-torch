# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transport module."""


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import torchvision.models as models
from ravens_torch.models.resnet import ResNet43_8s
from ravens_torch.utils import utils, MeanMetrics, to_device
from ravens_torch.utils.text import bold
from ravens_torch.utils.utils import apply_rotations_to_tensor


class Transport:
    """Transport module."""

    def __init__(self, n_rotations, crop_size, preprocess, verbose=False, name="Transport"):
        """Transport module for placing.

        Args:
          n_rotations: number of rotations of convolving kernel.
          crop_size: crop size around pick argmax used as convolving kernel.
          preprocess: function to preprocess input images.
        """
        self.iters = 0
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        # 2 fully convolutional ResNets
        resnet_model = models.resnet18(pretrained=True)

        self.model_query = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.model_key = torch.nn.Sequential(*(list(model.children())[:-1]))
        for param in self.model_query.parameters():
            param.requires_grad = True
        for param in self.model_key.parameters():
            param.requires_grad = True

        self.model_mlp = torch.nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(1000, 128), ## 1000 might be wrong!
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2)
                )

        self.device = to_device(
            [self.model_query, self.model_key, self.model_mlp], name, verbose=verbose)

        self.optimizer_query = optim.Adam(self.model_query.parameters(), lr=1e-4)
        self.optimizer_key = optim.Adam(self.model_key.parameters(), lr=1e-4)
        self.optimizer_mlp = optim.Adam(self.model_mlp.parameters(), lr=1e-4)
        self.loss = nn.MSELoss()
        self.metric = MeanMetrics()

    def correlate(self, in0, in1):
        """Correlate two input tensors."""
        in0 = Rearrange('b h w c -> b c h w')(in0)
        in1 = Rearrange('b h w c -> b c h w')(in1)

        output = F.conv2d(in0, in1)
        output = self.mlp(output)
        return output

    def forward(self, in_img, p):
        """Forward pass."""
        img_unprocessed = np.pad(in_img, self.padding, mode='constant')
        input_data = self.preprocess(img_unprocessed.copy())
        input_data = Rearrange('h w c -> 1 h w c')(input_data)
        in_tensor = torch.tensor(
            input_data, dtype=torch.float32
        ).to(self.device)

        # Rotate crop.
        pivot = list(np.array([p[1], p[0]]) + self.pad_size)

        # Crop before network (default for Transporters in CoRL submission).
        crop = apply_rotations_to_tensor(
            in_tensor, self.n_rotations, center=pivot)
        crop = crop[:, p[0]:(p[0] + self.crop_size),
                    p[1]:(p[1] + self.crop_size), :]

        logits = self.model_query(in_tensor)
        kernel_raw = self.model_key(crop)

        # Obtain kernels for cross-convolution.
        # Padding of one on right and bottom of (h, w)
        kernel_paddings = nn.ConstantPad2d((0, 0, 0, 1, 0, 1, 0, 0), 0)
        kernel = kernel_paddings(kernel_raw)

        return self.correlate(logits, kernel)

    def train_block(self, in_img, p, q):
        output = self.forward(in_img, p)
        label = torch.tensor(q, dtype=torch.float32).to(self.device)
        print(output, label)
        loss = self.loss(output, label)

        return loss

    def train(self, in_img, p, q):
        """Transport pixel p to pixel q.

        Args:
          in_img: input image.
          p: pixel (y, x)
          q: pixel (y, x)

        Returns:
          loss: training loss.
        """

        self.metric.reset()
        self.train_mode()
        self.optimizer_query.zero_grad()
        self.optimizer_key.zero_grad()
        self.optimizer_mlp.zero_grad()

        loss = self.train_block(in_img, p, q)
        loss.backward()
        self.optimizer_query.step()
        self.optimizer_key.step()
        self.optimizer_mlp.step()
        self.metric(loss)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())

    def test(self, in_img, p, q):
        """Test."""
        self.eval_mode()

        with torch.no_grad():
            loss = self.train_block(in_img, p, q)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())

    def train_mode(self):
        self.model_query.train()
        self.model_key.train()
        self.model_mlp.train()

    def eval_mode(self):
        self.model_query.eval()
        self.model_key.eval()
        self.model_mlp.eval()

    def format_fname(self, fname, suffix):
        return fname.split('.pth')[0] + f'_{suffix}.pth'

    def load(self, fname, verbose):
        query_name = self.format_fname(fname, 'query')
        key_name = self.format_fname(fname, 'key')
        mlp_name = self.format_fname(fname, 'mlp')

        if verbose:
            device = "GPU" if self.device.type == "cuda" else "CPU"
            print(
                f"Loading {bold('transport query')} model on {bold(device)} from {bold(query_name)}")
            print(
                f"Loading {bold('transport key')}   model on {bold(device)} from {bold(key_name)}")
            print(
                f"Loading {bold('transport mlp')}   model on {bold(device)} from {bold(mlp_name)}")

        self.model_query.load_state_dict(
            torch.load(query_name, map_location=self.device))
        self.model_key.load_state_dict(
            torch.load(key_name, map_location=self.device))
        self.model_mlp.load_state_dict(
            torch.load(mlp_name, map_location=self.device))

    def save(self, fname, verbose=False):
        query_name = self.format_fname(fname, 'query')
        key_name = self.format_fname(fname, 'key')
        mlp_name = self.format_fname(fname, 'mlp')

        if verbose:
            print(
                f"Saving {bold('transport query')} model to {bold(query_name)}")
            print(
                f"Saving {bold('transport key')}   model to {bold(key_name)}")
            print(
                f"Saving {bold('transport mlp')}   model to {bold(mlp_name)}")

        torch.save(self.model_query.state_dict(), query_name)
        torch.save(self.model_key.state_dict(), key_name)
        torch.save(self.model_mlp.state_dict(), mlp_name)
