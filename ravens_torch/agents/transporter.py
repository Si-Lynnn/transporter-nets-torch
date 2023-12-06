# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transporter Agent."""

import os

import numpy as np
from ravens_torch.models.attention_convmlp import AttentionConvMLP
from ravens_torch.models.transport import Transport
from ravens_torch.models.img2real import Img2Real
from ravens_torch.tasks import cameras
from ravens_torch.utils import utils


### attention loss on image coord u, v
### transporter loss on image coord u, v
### extra mlp to convert placing u, v to real world coord x, y

class TransporterAgent:
    """Agent that uses Transporter Networks."""

    def __init__(self, name, task, root_dir, n_rotations=36, verbose=False):
        self.name = name
        self.task = task
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = n_rotations
        self.pix_size = 0.003125
        self.in_shape = (240, 130, 3)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.models_dir = os.path.join(root_dir, 'checkpoints', self.name)
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        # input: image, output: pick loc in image coord u, v
        self.attention = AttentionConvMLP(
            preprocess=utils.preprocess_convmlp,
            verbose=False)

        # input: image, output: place loc in image coord u, v
        self.transport = Transport(
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess_convmlp,
            verbose=False)

        # input: image coord u,v, output: real world pose x, y for placing
        self.img2real = Img2Real(verbose=False)

    def get_sample(self, dataset):
        """Get a dataset sample.

        Args:
          dataset: a ravens_torch.Dataset (train or validation)
          augment: if True, perform data augmentation.

        Returns:
          tuple of data for training:
            (input_image, p0, p0_theta, p1, p1_theta)
          tuple additionally includes (z, roll, pitch) if self.six_dof
          if self.use_goal_image, then the goal image is stacked with the
          current image in `input_image`. If splitting up current and goal
          images is desired, it should be done outside this method.
        """

        (obs, act, _, _), _ = dataset.sample()
        img = obs['color'][0]

        # Get training labels from data sample.
        p0_xyz, p0_xyzw = act['pose0']
        p1_xyz, p1_xyzw = act['pose1']
        p0 = p0_xyz[:2]
        p1 = p1_xyz[:2]

        ## todo: add pick & place loc in image coord during annotation
        uv0 = act['uv0']
        uv1 = act['uv1']

        return img, p0, p1, uv0, uv1

    def train(self, dataset, writer=None):
        """Train on a dataset sample for 1 iteration.

        Args:
          dataset: a ravens_torch.Dataset.
          writer: a TensorboardX SummaryWriter.
        """
        self.attention.train_mode()
        self.transport.train_mode()
        self.img2real.train_mode()

        img, p0, p1, uv0, uv1 = self.get_sample(dataset)

        # Get training losses.
        step = self.total_steps + 1
        loss0 = self.attention.train(img, uv0)
        loss1 = self.transport.train(img, uv0, uv1)
        loss2 = self.img2real.train(uv1, p1)

        writer.add_scalars([
            ('train_loss/attention', loss0, step),
            ('train_loss/transport', loss1, step),
            ('train_loss/img2real', loss2, step),
        ])

        print(
            f'Train Iter: {step} \t Attention Loss: {loss0:.4f} \t Transport Loss: {loss1:.4f} \t Img2Real Loss: {loss2:.4f}')
        self.total_steps = step

    def validate(self, dataset, writer=None):  # pylint: disable=unused-argument
        """Test on a validation dataset for 10 iterations."""

        n_iter = 10
        loss0, loss1, loss2 = 0, 0, 0
        for _ in range(n_iter):
            img, p0, p1, uv0, uv1 = self.get_sample(dataset)

            # Get validation losses. Do not backpropagate.
            loss0 += self.attention.test(img, uv0)
            loss1 += self.transport.test(img, uv0, uv1)
            loss2 += self.img2real.test(uv1, p1)
        loss0 /= n_iter
        loss1 /= n_iter
        loss2 /= n_iter

        writer.add_scalars([
            ('test_loss/attention', loss0, self.total_steps),
            ('test_loss/transport', loss1, self.total_steps),
            ('test_loss/img2real', loss2, self.total_steps),
        ])

        print(
            f'Validation: \t Attention Loss: {loss0:.4f} \t Transport Loss: {loss1:.4f} \t Img2Real Loss: {loss2:.4f}')

    def act(self, img):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        self.attention.eval_mode()
        self.transport.eval_mode()
        self.img2real.eval_mode()

        # Attention model forward pass.
        pick_uv = self.attention.forward(img)
        # Transport model forward pass.
        ## TODO: here we can replace the pick_uv to a opencv detected pick uv
        place_uv = self.transport.forward(img, pick_uv)
        # Img2Real model forward pass.
        pick_xy = self.img2real.forward(pick_uv)
        place_xy = self.img2real.forward(place_uv)

        return pick_uv, place_uv, pick_xy, place_xy

    def get_checkpoint_names(self, n_iter):
        attention_fname = 'attention-ckpt-%d.pth' % n_iter
        transport_fname = 'transport-ckpt-%d.pth' % n_iter
        img2real_fname = 'img2real-ckpt-%d.pth' % n_iter

        attention_fname = os.path.join(self.models_dir, attention_fname)
        transport_fname = os.path.join(self.models_dir, transport_fname)
        img2real_fname = os.path.join(self.models_dir, img2real_fname)

        return attention_fname, transport_fname, img2real_fname

    def load(self, n_iter, verbose=False):
        """Load pre-trained models."""
        attention_fname, transport_fname, img2real_fname = self.get_checkpoint_names(n_iter)

        self.attention.load(attention_fname, verbose)
        self.transport.load(transport_fname, verbose)
        self.img2real.load(img2real_fname, verbose)
        self.total_steps = n_iter

    def save(self, verbose=False):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        attention_fname, transport_fname, img2real_fname = self.get_checkpoint_names(
            self.total_steps)

        self.attention.save(attention_fname, verbose)
        self.transport.save(transport_fname, verbose)
        self.img2real.save(img2real_fname, verbose)
