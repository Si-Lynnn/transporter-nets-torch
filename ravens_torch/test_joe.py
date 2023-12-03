# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens
"""Ravens main training script."""

import os
import numpy as np
import pickle
from absl import app, flags

from ravens_torch import agents, tasks
from ravens_torch.environments.environment import Environment
from ravens_torch.constants import EXPERIMENTS_DIR, ENV_ASSETS_DIR, VIDEOS_DIR
from ravens_torch.utils.initializers import set_seed
from ravens_torch.utils.text import bold
from ravens_torch.utils.video_recorder import VideoRecorder
from ravens_torch.dataset import load_data


flags.DEFINE_string('root_dir', EXPERIMENTS_DIR, help='Location of test data')
flags.DEFINE_string('data_dir', EXPERIMENTS_DIR, '')
flags.DEFINE_string('assets_root', ENV_ASSETS_DIR,
                    help='Location of assets directory to build the environment')
flags.DEFINE_bool('disp', True, help='Display OpenGL window')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'block-insertion', help='Task to complete')
flags.DEFINE_string('agent', 'transporter',
                    help='Agent to perform Pick-and-Place')
flags.DEFINE_inravens_torch/test_joe.pyteger('n_demos', 100,
                     help='Number of training demos')
flags.DEFINE_integer('n_steps', 40000,
                     help='Number of training steps performed')
flags.DEFINE_integer('n_runs', 1, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('gpu_limit', None, '')
flags.DEFINE_boolean('verbose', True,
                     help='Print more information while running this script')
flags.DEFINE_boolean('record_mp4', False,
                     help='Record mp4 videos of the tasks being completed')
FLAGS = flags.FLAGS


def main(unused_argv):
    # Initialize environment and task.
    env = Environment(
        FLAGS.assets_root,
        disp=FLAGS.disp,
        shared_memory=FLAGS.shared_memory,
        hz=240)
    task = tasks.names[FLAGS.task]()
    task._set_mode('test')
    print(bold("=" * 20 + "\n" + f"TASK: {FLAGS.task}" + "\n" + "=" * 20))

    # Load test dataset.
    ds = load_data(FLAGS, only_test=True)
    agent = agents.names[FLAGS.agent](name, FLAGS.task, FLAGS.root_dir)
    # test commit
    # # Run testing for each training run.
    # for train_run in range(FLAGS.n_runs):
    #     name = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-{train_run}'

    #     # Initialize agent.
    #     set_seed(train_run)
        

    #     # # Run testing every interval.
    #     # for train_step in range(0, FLAGS.n_steps + 1, FLAGS.interval):

    #     # Load trained agent.
    #     agent.load(FLAGS.n_steps, FLAGS.verbose)



def get_video_dir(FLAGS, train_run):
    task_video_dir = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-{train_run}'
    return os.path.join(VIDEOS_DIR, task_video_dir)


if __name__ == '__main__':
    app.run(main)
