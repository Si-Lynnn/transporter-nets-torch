import cv2
import os
import rospy
import numpy as np
import k4a
import argparse
from frankapy import FrankaArm
import time
import pickle as pkl
from autolab_core import RigidTransform
from transform import *
import copy
import matplotlib.pyplot as plt
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

flags.DEFINE_string('root_dir', EXPERIMENTS_DIR, help='Location of test data')
flags.DEFINE_string('data_dir', EXPERIMENTS_DIR, '')
flags.DEFINE_string('assets_root', ENV_ASSETS_DIR,
                    help='Location of assets directory to build the environment')
flags.DEFINE_bool('disp', True, help='Display OpenGL window')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'block-insertion', help='Task to complete')
flags.DEFINE_string('pick_agent', 'transporter_convmlp',
                    help='Agent to perform Pick-and-Place')
flags.DEFINE_string('place_agent', 'transporter_place',
                    help='Agent to perform Pick-and-Place')
flags.DEFINE_integer('n_demos', 200,
                     help='Number of training demos')
flags.DEFINE_integer('n_steps', 10000,
                     help='Number of training steps performed')
flags.DEFINE_integer('n_runs', 1, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('gpu_limit', None, '')
flags.DEFINE_boolean('verbose', True,
                     help='Print more information while running this script')
flags.DEFINE_boolean('record_mp4', False,
                     help='Record mp4 videos of the tasks being completed')
FLAGS = flags.FLAGS

class data_collector:
    def __init__(self, FLAGS, translate_height=0.12,pick_height=0.02,xrange=0.35,yrange=0.6,calib_offset = [0.02,0.005]):
        self.calib_offset = calib_offset
        # self.init_world_goal = np.array([0.05,0.05,pick_height]) 
        # previous used for start point for data_50steps.pkl and data_50steps_red.pkl
        self.pick_height=pick_height
        self.init_world_goal = np.array([0.09,0.066,pick_height])
        # used as start points for data_50steps_2_blue.pkl and data_50steps_2_red.pkl
        # also used translate height of 0.06
        #self.init_world_goal = np.array([0.05,0.05,pick_height])
        self.init_camera()
        self.save_path = "/home/student/team-joe/transporter-nets-torch/results/"
        
        
        
        self.translate_height=translate_height
        self.xrange = xrange
        self.yrange = yrange
        self.current_world = self.init_world_goal
        self.init_rotation = None
        self.init_robot()
        self.init_pick_rob_frame, self.init_rotation_goal = self.get_pick_pose()
        self.init_model(FLAGS)

    def init_model(self, FLAGS):
        
        task = tasks.names[FLAGS.task]()
        task._set_mode('test')
        print(bold("=" * 20 + "\n" + f"TASK: {FLAGS.task}" + "\n" + "=" * 20))
        # Load agent
        name = f'{FLAGS.task}-{FLAGS.pick_agent}-{FLAGS.n_demos}-0'
        self.pick_agent = agents.names[FLAGS.pick_agent](name, FLAGS.task, FLAGS.root_dir)
        self.pick_agent.load(FLAGS.n_steps, FLAGS.verbose)

        name = f'{FLAGS.task}-{FLAGS.place_agent}-{FLAGS.n_demos}-0'
        self.place_agent = agents.names[FLAGS.place_agent](name, FLAGS.task, FLAGS.root_dir)
        self.place_agent.load(FLAGS.n_steps, FLAGS.verbose)
            
    def init_robot(self):
        print('Starting robot')
        self.fa = FrankaArm()
        self.fa.reset_pose()
        self.fa.reset_joints()
        self.fa.open_gripper()

        pose = self.fa.get_pose()
        self.init_rotation = pose.rotation
    
    def z_rotation(self,angle):
        # angle in radians
        R = np.array([[np.cos(angle),-np.sin(angle),0],
                      [np.sin(angle),np.cos(angle),0],
                      [0,0,1]])
        return R
    
    def init_camera(self):
        # Start Cameras
        self.kinect = k4a.Device.open()
        device_config = k4a.DEVICE_CONFIG_BGRA32_1080P_NFOV_UNBINNED_FPS15
        self.kinect.start_cameras(device_config)
        # Get Calibration
        calibration = self.kinect.get_calibration(
            depth_mode=device_config.depth_mode,
            color_resolution=device_config.color_resolution)
        # Create Transformation
        self.transformation = k4a.Transformation(calibration)
        self.calib = k4a.ECalibrationType.DEPTH

    def stop_camera(self):
        self.kinect.stop_cameras()

    def reset_arm(self):
        # self.fa.reset_pose()
        self.fa.reset_joints()

    def grasp(self):
        self.fa.close_gripper()

    def ungrasp(self):
        self.fa.open_gripper()
    
    def move_arm(self, dx=0.1, dy=0.1, dz=0.1):
        ### dx, dy, dz are in meter
        delta_pose = RigidTransform(rotation=np.eye(3), translation=np.array([dx, dy, dz]),
                                    from_frame="world", to_frame="world")
        self.fa.goto_pose_delta(delta_pose)
    
    def get_action(self):
        #pose = self.fa.get_pose() # bart:False this is not a reliable way to get the current pose
        pose = world2rob(self.current_world)[:3]
        return pose

    def get_image(self):
        capture = self.kinect.get_capture(-1)
        color_img = capture.color.data
        # print(color_img.shape)
        # # color_img = cv2.resize(color_img, dsize=(640, 480))
        # print(color_img.shape)

        depth_img = capture.depth.data
        color_img = cv2.resize(color_img, dsize=(320, 180))
        depth_img = cv2.resize(depth_img, dsize=(320, 288))

        # depth_img = cv2.resize(depth_img, dsize=(640, 480))
        # depth_cap = self.transformation.depth_image_to_point_cloud(capture.depth, self.calib)
        # height, width, channels = depth_cap.data.shape
        # depth_img = depth_cap.data.reshape(height*width, channels)
        return color_img, depth_img

    def pick(self,rob_frame_goal,rob_frame_rotation,pose_goal=None):
        self.fa.open_gripper()
        T_ee_world = self.fa.get_pose()
        T_ee_goal = T_ee_world
        T_ee_goal.translation = rob_frame_goal
        T_ee_goal.rotation = rob_frame_rotation
        self.fa.goto_pose(T_ee_goal,use_impedance=False)
        T_ee_goal.translation[2] = self.pick_height
        self.fa.goto_pose(T_ee_goal,use_impedance=False)
        self.fa.close_gripper()
        self.current_world = rob2world(rob_frame_goal)

    def translate(self,translate_goal,goal_rotation=None):
        # time.sleep(1)
        T_ee_world = self.fa.get_pose()
        T_ee_goal = copy.deepcopy(T_ee_world)
        
        # move up 
        current_pose_world = self.current_world
        print("current pose world: ",current_pose_world)
        desired_pose_world = copy.deepcopy(current_pose_world)
        desired_pose_world[-1] = self.translate_height
        desired_pose_robot = world2rob(desired_pose_world)
        T_ee_goal.translation = desired_pose_robot
        if goal_rotation is not None:
            T_ee_goal.rotation = goal_rotation
        self.fa.goto_pose(T_ee_goal,use_impedance=False)
        
        # translate to goal at translate height
        T_ee_goal.translation = translate_goal
        self.fa.goto_pose(T_ee_goal,use_impedance=False)
    
    def pick_and_place(self):
        color_img, depth_img = self.get_image()

        # get ground truth to compare
        pick_gt, pick_rotation = self.get_pick_pose()

        # get predicted pick loc from model 

        pick_loc, place_loc, place_gt, place_gt_img = self.get_pred_pick_pose()

        print("GT pick loc: ", pick_gt)
        print("Predicted loc: ", pick_loc)
        print("GT place loc: ", place_gt)
        print("Predicted place loc: ", place_loc)

        # # execution
        self.pick(rob_frame_goal=pick_loc,rob_frame_rotation = pick_rotation)

        # p = input("Pause")
        self.translate(translate_goal=place_loc,goal_rotation=pick_rotation)
        
        self.ungrasp()
        self.reset_arm()

        color_img, depth_img = self.get_image()
        bag_color = np.array([80, 35, 20])
        # grab image 
        color_img = color_img[:,:,:3]
        color_img = np.ascontiguousarray(color_img)

        # Preprocessing
        edge = int((320-240)/2)
        bottom = int(50)
        desired_size = (240, 130)
        
        color_img = color_img[:-bottom,edge*2:,:]
        color_img = cv2.resize(color_img, desired_size)
        color_img_copy = np.copy(color_img)


        # Annotate virtual box
        angle = 0.0
        square_size = 17.5
        center = place_gt_img
        square_box = np.array([[square_size, square_size], [square_size, -square_size], [-square_size, -square_size], [-square_size, square_size]])
        box = center + np.dot(square_box, np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]))
        box = box.astype(np.int32)
        # mask = self.detect_bag(color_img, bag_color)
        # cv2.drawContours(color_img, [box], -1, (0, 255, 0), -1)
        # color_img = np.where(mask[:,:,None], color_img_copy, color_img)
        color_img_copy = np.copy(color_img)

        debug = True
        if debug:
            # draw box centroid with cv2 circle
            cv2.imshow("Img", color_img)
            cv2.waitKey(0)
            cv2.imwrite(self.save_path + "place_img.png", color_img)        


    def pix2rob(self,pix):
        # pix: (x,y) in pixel
        # return: (x,y) in robot frame
        pix_h = np.array([pix[1],pix[0],1.0]).T
        H = np.array([[ 0.00655029,  0.00069354,  0.01121504],
            [-0.0001672,   0.00687641, -1.35739522],
            [-0.00020222,  0.0014612,   1.        ]])
        rob_h = H @ pix_h
        rob = rob_h[:2]/rob_h[2]
        return rob

    def get_pred_pick_pose(self):
        bag_color = np.array([80, 35, 20])
        # grab image 
        color_img, depth_img = self.get_image()
        # print(color_img.shape)
        # color_img[:,:,[0,2]] = color_img[:,:,[2,0]]
        color_img = color_img[:,:,:3]
        color_img = np.ascontiguousarray(color_img)

        # Preprocessing
        edge = int((320-240)/2)
        bottom = int(50)
        desired_size = (240, 130)
        
        color_img = color_img[:-bottom,edge*2:,:]
        color_img = cv2.resize(color_img, desired_size)
        color_img_copy = np.copy(color_img)


        # Annotate virtual box
        # x = np.random.randint(low = 60, high = 180)
        # y = np.random.randint(low = 30, high = 100)

        box = self.detect_rectangle(color_img, bag_color, 15)
        box_centroid = np.mean(box, axis=0)

        found_point = False
        goal_dist = 60
        while not found_point:
            x_goal = np.random.randint(low=60, high=180) # limited to prevent self cpllision
            y_goal = np.random.randint(low=30, high=100)

            dist = np.linalg.norm(np.array([x_goal,y_goal])-box_centroid)
            if dist > goal_dist:
                found_point = True

        center = np.array([x_goal, y_goal])
        angle = 0.0
        square_size = 17.5
        square_box = np.array([[square_size, square_size], [square_size, -square_size], [-square_size, -square_size], [-square_size, square_size]])
        box = center + np.dot(square_box, np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]))
        box = box.astype(np.int32)
        #mask = self.detect_bag(color_img, bag_color)
        #cv2.drawContours(color_img, [box], -1, (0, 255, 0), -1)
        #color_img = np.where(mask[:,:,None], color_img_copy, color_img)
        color_img_copy = np.copy(color_img)

        debug = True
        if debug:
            # draw box centroid with cv2 circle
            cv2.imshow("Img", color_img)
            cv2.waitKey(0)
            cv2.imwrite(self.save_path + "pick_img.png", color_img)

        pick_loc = self.pick_agent.pick_loc(color_img)
        pick_loc = pick_loc.to('cpu').detach().numpy()[0]
        # pick_loc = [0, 0] 

        pick = np.append(pick_loc, self.translate_height)

        place_loc = self.place_agent.pick_loc(color_img_copy)
        place_loc = place_loc.to('cpu').detach().numpy()[0]
        

        place = np.append(place_loc, self.translate_height)

        
        # return predicted pick loc, place loc, gt place loc
        gt_place = self.pix2rob(np.array([center[0] + edge * 2, center[1]])) + self.calib_offset
        return pick, place, gt_place, center
    
    def get_pick_pose(self):
        # grab image 
        color_img, depth_img = self.get_image()
        # print(color_img.shape)
        color_img[:,:,[0,2]] = color_img[:,:,[2,0]]
        color_img = color_img[:,:,:3]
        color_img = np.ascontiguousarray(color_img)

        bag_color = np.array([14, 28, 68])
        thresh_rect = 15
        thresh_bag = 20

        box = self.detect_rectangle(color_img, bag_color, thresh_rect)
        box_centroid = np.mean(box, axis=0)
        pick = self.pix2rob(box_centroid)
        pick += self.calib_offset
        
        
        # compute angle in radians from box
        angle = np.arctan2(box[0,1] - box[1,1], box[0,0] - box[1,0])
        #find the angle of the box closest to zero 
        if angle > 0:
            tmp_angle = copy.deepcopy(angle)
            while tmp_angle > 0:
                tmp_angle = angle - np.pi/2
                if tmp_angle < 0:
                    break
                else:
                    angle = tmp_angle

        else:
            tmp_angle = copy.deepcopy(angle)
            while tmp_angle < 0:
                tmp_angle = angle + np.pi/2
                if tmp_angle > 0:
                    break
                else:
                    angle = tmp_angle

        if abs(np.radians(45) - abs(angle)) < np.radians(10):
            angle = 0
            pick += self.calib_offset

        pick = np.append(pick, self.translate_height)

        angle = 0
        goal_rotation = self.init_rotation @ self.z_rotation(angle)
        
        debug = False
        if debug:
            cv2.drawContours(color_img, [box], -1, (0, 255, 0), 2)
            # draw box centroid with cv2 circle
            cv2.circle(color_img, (int(box_centroid[0]), int(box_centroid[1])), 5, (255, 0, 0), -1)    
            plt.imshow(color_img)
            plt.show()
        
        return pick, goal_rotation

    def detect_rectangle(self,color, bag_color, thresh=15):
        """ Detect rectangle using minArea rectangle"""
        # mask with (70, 30, 10) +- 10
        lower_bound = bag_color - thresh
        upper_bound = bag_color + thresh
        mask = cv2.inRange(color, lower_bound, upper_bound)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key = cv2.contourArea)
        min_rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(min_rect)
        box = np.int0(box)
        center = np.mean(box, axis=0)
        angle = np.radians(min_rect[2])
        square_size = 17.5
        square_box = np.array([[square_size, square_size], [square_size, -square_size], [-square_size, -square_size], [-square_size, square_size]])
        box = center + np.dot(square_box, np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]))
        box = box.astype(np.int32)
        return box

    def detect_bag(self,color, bag_color, thresh=20):
        """ Detect rectangle using minArea rectangle"""
        # mask with (70, 30, 10) +- 10
        lower_bound = bag_color - thresh
        upper_bound = bag_color + thresh
        mask = cv2.inRange(color, lower_bound, upper_bound)
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # largest_contour = max(contours, key = cv2.contourArea)
        return mask


def main(unused_argv):
    collector = data_collector(FLAGS)
    collector.pick_and_place()

if __name__ == "__main__":
    app.run(main)
    