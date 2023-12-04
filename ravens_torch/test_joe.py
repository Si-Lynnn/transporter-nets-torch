# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens
"""Ravens main training script."""

import os
import numpy as np
import pickle as pkl
from absl import app, flags

from ravens_torch import agents, tasks
from ravens_torch.environments.environment import Environment
from ravens_torch.constants import EXPERIMENTS_DIR, ENV_ASSETS_DIR, VIDEOS_DIR
from ravens_torch.utils.initializers import set_seed
from ravens_torch.utils.text import bold
# from ravens_torch.utils.video_recorder import VideoRecorder
from ravens_torch.dataset import load_data
from ravens_torch import agents
from ravens_torch.constants import EXPERIMENTS_DIR
from ravens_torch.dataset import load_data
from ravens_torch.utils import SummaryWriter
from ravens_torch.utils.initializers import get_log_dir, set_seed
from transform import *
import k4a 
from frankapy import FrankaArm
from autolab_core import RigidTransform
from transform import *
import matplotlib.pyplot as plt

flags.DEFINE_string('root_dir', EXPERIMENTS_DIR, help='Location of test data')
flags.DEFINE_string('data_dir', EXPERIMENTS_DIR, '')
flags.DEFINE_string('assets_root', ENV_ASSETS_DIR,
                    help='Location of assets directory to build the environment')
flags.DEFINE_bool('disp', True, help='Display OpenGL window')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'block-insertion', help='Task to complete')
flags.DEFINE_string('agent', 'transporter_convmlp',
                    help='Agent to perform Pick-and-Place')
flags.DEFINE_integer('n_demos', 150,
                     help='Number of training demos')
flags.DEFINE_integer('n_steps', 1000,
                     help='Number of training steps performed')
flags.DEFINE_integer('n_runs', 1, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('gpu_limit', None, '')
flags.DEFINE_boolean('verbose', True,
                     help='Print more information while running this script')
flags.DEFINE_boolean('record_mp4', False,
                     help='Record mp4 videos of the tasks being completed')
FLAGS = flags.FLAGS


class robot_arm:
    def __init__(self, translate_height=0.1, pick_height=0.01, xrange=[0.025, 0.35], yrange=[0, 0.6]): 
        self.translate_height = translate_height
        self.pick_height = pick_height
        self.xrange = xrange
        self.yrange = yrange

        self.init_world_goal = np.array([(xrange[1]/2 - xrange[0]/2), yrange[1]/2-yrange[0]/2, translate_height])
        self.init_pick_rob_frame = world2rob(self.init_world_goal)[:3]
        self.current_world = self.init_world_goal

        self.init_robot()
        self.init_camera()

    def init_robot(self):
        print("Initializing robot...")
        self.fa = FrankaArm()
        self.fa.reset_pose()
        self.fa.reset_joints()
        self.fa.open_gripper()
    
    def init_camera(self):
        self.kinect = k4a.Device.open()
        device_config = k4a.DEVICE_CONFIG_BGRA32_1080P_NFOV_UNBINNED_FPS15
        self.kinect.start_cameras(device_config)
        # Get calibration
        calibration = self.kinect.get_calibration(
            depth_mode=device_config.depth_mode, 
            color_resolution=device_config.color_resolution)
        # Create transformation
        self.transformation = k4a.Transformation(calibration)
    
    def stop_camera(self):
        self.kinect.stop_cameras()

    def reset_arm(self):
        self.fa.reset_pose()
        self.fa.reset_joints()
        self.fa.open_gripper()

    def grasp(self):
        self.fa.close_gripper()
    
    def ungrasp(self):
        self.fa.open_gripper()
    
    def get_image(self):
        capture = self.kinect.get_capture(-1)
        color_img = capture.color.data
        color_img = cv2.resize(color_img, (640, 480))
        depth_img = capture.transformed_depth.data
        depth_img = cv2.resize(depth_img, (640, 480))
        return color_img, depth_img

    def translate(self, loc):
        T_ee_world = self.fa.get_pose()
        T_ee_goal = copy.deepcopy(T_ee_world)

        # move up
        current_pose_world = self.current_world
        desired_pose_world = copy.deepcopy(current_pose_world)
        desired_pose_world[-1] = self.translate_height
        desired_pose_rob = world2rob(desired_pose_world)
        T_ee_goal.translation = desired_pose_robot
        self.fa.goto_pose(T_ee_goal, use_impedance=False)

        # translate to goal at translate height
        T_ee_goal.translation = translate_goal
        self.fa.goto_pose(T_ee_goal, use_impedance=False)

    def goto_above_pick(self, loc):
        rob_translate_goal = world2rob(np.append(loc, self.translate_height))
        self.translate(rob_translate_goal)
        self.current_world = rob2world(rob_translate_goal)
        self.open_gripper()

    def pick(self, loc):
        rob_pick_goal = world2rob(np.append(loc, self.pick_height))
        self.translate(rob_pick_goal)
        self.current_world = rob2world(rob_pick_goal)
        self.grasp()
    
    def place(self, loc):
        rob_place_goal = world2rob(np.append(loc, self.translate_height))
        self.translate(rob_place_goal)
        self.current_world = rob2world(rob_place_goal)
        self.ungrasp()

    def stop(self):
        self.reset_arm()
        self.stop_camera()
        

def main(unused_argv):
    path = "/home/student/team-joe/ai4m_project-main/data/"
    file_name = "data_10steps_test_blue.pkl"

    print("env")
    #env = Environment(
    #    FLAGS.assets_root,
    #    disp=FLAGS.disp,
    #    shared_memory=FLAGS.shared_memory,
    #    hz=240)
    print("tasls")
    task = tasks.names[FLAGS.task]()
    task._set_mode('test')
    print(bold("=" * 20 + "\n" + f"TASK: {FLAGS.task}" + "\n" + "=" * 20))
    # Load agent
    name = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-0'
    agent = agents.names[FLAGS.agent](name, FLAGS.task, FLAGS.root_dir)
    agent.load(FLAGS.n_steps, FLAGS.verbose)

    # Load data
    with open(f'{path}action/{file_name}', 'rb') as f:
        actions = pkl.load(f)
    with open(f'{path}color/{file_name}', 'rb') as f:
        color = pkl.load(f)
    with open(f'{path}depth/{file_name}', 'rb') as f:
        depth = pkl.load(f)
    
    print("Starting images")

    for img in color:
        print(img.shape)
        # img = img[:, :, :-1]
        img = img[:,:,:3] # BGR -> RGB is done in preprocessing
        plt.imshow(img)
        loc = agent.pick_loc(img)
        loc = loc.to('cpu').detach().numpy()[0]
        loc = np.append(loc,np.array([0.03]))
        
        # print(loc[0])
        # plt.imshow(img)
        # plt.scatter(loc[0], loc[1], c='r', s=100)

        # plt.plot(loc, '*')
        print("robot coords inference:")
        print(loc)
        plt.show()


    '''
    rob = robot_arm()
    color_img, depth_img = rob.get_image()
    pick_loc = pick_loc(color_img)
    rob.goto_pick_loc(pick_loc)
    rob.pick(pick_loc)
    place_loc = pick_loc + np.array([0.05, 0.05])
    rob.place(place_loc)
    rob.stop() 
    '''
    

if __name__ == '__main__':
    app.run(main)
