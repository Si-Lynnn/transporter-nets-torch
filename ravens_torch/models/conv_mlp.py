# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Conv MLP module."""

from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

from ravens_torch.models.gt_state import MlpModel
from ravens_torch.models.resnet import ResNet43_8s
from ravens_torch.utils import to_device
import torchvision.models as models
from einops import rearrange



def init_normal_weights_bias(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight)
        torch.nn.init.normal_(m.bias)


def DenseBlock(in_channels, out_channels, activation=None):
    if activation is not None:
        fc = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            activation(),
        )
    else:
        fc = nn.Linear(in_channels, out_channels)

    fc.apply(init_normal_weights_bias)

    return fc


class SpatialSoftArgmax(nn.Module):
    def __init__(self, batch_size, H=149, W=69, C=64):  # pylint: disable=invalid-name
        """Parameter-less, extract coordinates for each channel.

        Args:
        x: shape: (batch_size, C, ~H, ~W)
        batch_size: int
        H: size related to original image H size
        W: size related to original image W size
        C: channels

        Returns:
        shape: (batch_size, C, 2)
        """
        super(SpatialSoftArgmax, self).__init__()

        self.layers = nn.Sequential(
            Rearrange('b c h w -> (b c) (h w)'),
            nn.Softmax(dim=1),
            Rearrange('(b c) (h w) -> b c h w 1', b=batch_size, c=C, h=H, w=W),
        )

        posx, posy = torch.meshgrid(
            torch.linspace(-1., 1., steps=H),
            torch.linspace(-1., 1., steps=W))
        image_coords = torch.stack((posx, posy), dim=2)  # (H, W, 2)

        self.image_coords = image_coords.unsqueeze(0)
        self.image_coords = self.image_coords.to(device='cuda:0')
        # print(self.image_coords.shape)
        # Convert image coords to shape [1, H, W, 2]
        # self.image_coords = Rearrange('h w 2 -> 1 h w 2')(image_coords)

    def forward(self, x):
        # Apply softmax and convert to shape [B, C, H, W, 1]
        softmax = self.layers(x)

        # Multiply (with broadcasting) and reduce over image dimensions to get the
        # result of shape [B, C, 2].
        # CHECK THE RESULT OF THE FOLLOWING LINE
        spatial_soft_argmax = torch.mean(
            softmax * self.image_coords, dim=[2, 3])

        return spatial_soft_argmax


class ConvMLP(nn.Module):
    """Conv MLP module."""

    def __init__(self, d_action, use_mdn, pretrained=False, verbose=False):
        super(ConvMLP, self).__init__()

        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = True

        self.d_action = d_action

        self.mlp = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, d_action)
            # DenseBlock(1000, 128, activation=nn.ReLU),
            # DenseBlock(128, 32, activation=nn.ReLU),
            # DenseBlock(32, d_action, activation=None),
        )
        # note: no dropout on top of conv
        # the conv layers seem to help regularize the mlp layers

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def forward(self, x):
        """FPROP through module.

        Args:
          x: shape: (batch_size, H, W, C)

        Returns:
          shape: (batch_size, self.d_action)
        """
        x = x.unsqueeze(0)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.resnet(x)
        x = self.mlp(x)
        return x


def DeepConvMLP(in_channels, d_action, use_mdn, verbose=False):
    """Deep conv MLP module."""
    del use_mdn

    batch_size = 4

    channel_depth_dim = 16
    resnet43_8s_model = ResNet43_8s(
        in_channels,
        channel_depth_dim,
        cutoff_early=False,
        include_batchnorm=True)

    model = nn.Sequential(
        resnet43_8s_model,
        SpatialSoftArgmax(batch_size, 320, 160, 16),
        nn.Flatten(),
        DenseBlock(32, 128, activation=nn.ReLU),
        DenseBlock(128, 128, activation=nn.ReLU),
        DenseBlock(128, d_action, activation=None),
    )

    _ = to_device([model], "DeepConvMLP", verbose=verbose)

    return model


def main(verbose=False):
    conv_mlp = ConvMLP(d_action=3, use_mdn=None, verbose=verbose)
    device = to_device([conv_mlp], "ConvMLP", verbose=verbose)

    img = torch.randn(7, 320, 160, 3).to(device)
    out = conv_mlp(img)


if __name__ == "__main__":
    main(verbose=True)
