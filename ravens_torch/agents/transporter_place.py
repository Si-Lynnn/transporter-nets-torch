# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transporter Agent."""

import os

import numpy as np
from ravens_torch.models.attention_convmlp import AttentionConvMLP
from ravens_torch.tasks import cameras
from ravens_torch.utils import utils
import wandb

class TransporterPlaceAgent:
    """Agent that uses Transporter Networks."""

    def __init__(self, name, task, root_dir, n_rotations=36, verbose=False,use_wandb=False):
        self.name = name
        self.task = task
        self.total_steps = 0
        self.n_rotations = n_rotations
        self.in_shape = (640, 480, 3)
        self.models_dir = os.path.join(root_dir, 'checkpoints', self.name)

        self.attention_convmlp = AttentionConvMLP(
            in_shape=self.in_shape,
            preprocess=utils.preprocess_convmlp,
            verbose=False)
        # TODO: from 3D world to image coord
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="ravens",name="attention_convmlp")

    def get_sample(self, dataset, augment=False):
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

        # Data augmentation.
        if augment:
            img, _, (p0, p1), _ = utils.perturb(img, [p0, p1])

        return img, p0, p1

    def train(self, dataset, writer=None):
        """Train on a dataset sample for 1 iteration.

        Args:
          dataset: a ravens_torch.Dataset.
          writer: a TensorboardX SummaryWriter.
        """
        self.attention_convmlp.train_mode()

        img, p0, p1 = self.get_sample(dataset)

        # Get training losses.
        step = self.total_steps + 1
        loss0 = self.attention_convmlp.train(img, p1)

        writer.add_scalars([
            ('train_loss/attention_convmlp', loss0, step)
        ])
        if self.use_wandb:
            wandb.log({"train_loss/attention_convmlp": loss0})

        print(
            f'Train Iter: {step} \t AttentionConvMLP Loss: {loss0:.4f}')

        self.total_steps = step

    def validate(self, dataset, writer=None):  # pylint: disable=unused-argument
        """Test on a validation dataset for 10 iterations."""

        n_iter = 10
        loss0 = 0
        for _ in range(n_iter):
            img, p0, p1 = self.get_sample(dataset, False)

            # Get validation losses. Do not backpropagate.
            loss0 += self.attention_convmlp.test(img, p1)
        loss0 /= n_iter

        writer.add_scalars([
            ('test_loss/attention_convmlp', loss0, self.total_steps)
        ])
        if self.use_wandb:
            wandb.log({"test_loss/attention_convmlp": loss0})

        print(f'Validation: \t AttentionConvMLP Loss: {loss0:.4f}')

    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        self.attention_convmlp.eval_mode()

        # Attention model forward pass.
        img = obs['color'][0]
        pick_conf = self.attention_convmlp.forward(img)
        argmax = np.argmax(pick_conf.to('cpu').detach().numpy())
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p1_pix = argmax[:2]

        return {
            'pose0': (p1_pix, np.array([0,0,0,1])),
            'pose1': (p1_pix, np.array([0,0,0,1]))
        }

    def get_checkpoint_names(self, n_iter):
        attention_fname = 'attention-ckpt-%d.pth' % n_iter

        attention_fname = os.path.join(self.models_dir, attention_fname)

        return attention_fname

    def load(self, n_iter, verbose=False):
        """Load pre-trained models."""
        attention_fname = self.get_checkpoint_names(n_iter)
        self.attention_convmlp.load(attention_fname, verbose)
        self.total_steps = n_iter

    def save(self, verbose=False):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        attention_fname = self.get_checkpoint_names(
            self.total_steps)

        self.attention_convmlp.save(attention_fname, verbose)

    def place_loc(self, img):
        """Pick location from attention model."""
        self.attention_convmlp.eval_mode()

        # Attention model forward pass.
        place_conf = self.attention_convmlp.test_single_img(img)

        return place_conf
