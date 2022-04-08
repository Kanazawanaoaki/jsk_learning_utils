#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, sys
import rosbag
import matplotlib.pyplot as plt
import math
import csv
import pickle
import numpy as np
from dataclasses import dataclass
from sensor_msgs.msg import JointState, CompressedImage, Image
from typing import List, Union

from cv_bridge import CvBridge, CvBridgeError
import cv2
import PIL.Image

import config_reader


@dataclass
class AngleVector:
    data: np.ndarray

    @classmethod
    def from_ros_msg(cls, msg: JointState, joint_names: List[str]) -> 'AngleVector':
        joint_rads = []
        for j_name in joint_names:
            idx = msg.name.index(j_name)
            joint_rads.append(msg.position[idx])
        joint_rads = cls.clamp_rad_list(joint_rads)
        joint_angles = np.array([math.degrees(rad) for rad in joint_rads])
        return cls(joint_angles)

    @staticmethod
    def clamp_rad_list(rad_list) -> List[float]:
        min_val = -1 * math.pi
        max_val = math.pi
        rad_list = map(lambda x: min_val if x < min_val else x, rad_list)
        rad_list = map(lambda x: max_val if x > max_val else x, rad_list)
        return list(rad_list)

    def numpy(self) -> np.ndarray:
        return self.data

@dataclass
class RGBImage:
    data: PIL.Image.Image

    @classmethod
    def from_ros_msg(cls, msg: Union[CompressedImage, Image], image_config: config_reader.ImageConfig) -> 'RGBImage':
        bridge = CvBridge()
        cv2_img = bridge.compressed_imgmsg_to_cv2(msg)
        cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        cv2_img_rgb_cropped = cv2_img_rgb[image_config.x_min:image_config.x_max, image_config.y_min:image_config.y_max, :]
        pil_img = PIL.Image.fromarray(cv2_img_rgb_cropped)
        return cls(pil_img)

    def pil_image(self) -> PIL.Image.Image:
        return self.data

class RosbagReader(object):
    def __init__(self, bag_dir, data_dir, config):
        self.bag_dir = bag_dir
        self.data_dir = data_dir
        self.hz = config.rosbag_convert_hz
        self.image_config = config.image_config
        self.config = config
        print("hz : {}, bag_dir : {}, data_dir : {}".format(self.hz, self.bag_dir, self.data_dir))
        # self.joint_names = ["r_upper_arm_roll_joint","r_shoulder_pan_joint","r_shoulder_lift_joint","r_forearm_roll_joint", "r_elbow_flex_joint","r_wrist_flex_joint","r_wrist_roll_joint"] # joint_states order
        # self.joint_names = ["r_shoulder_pan_joint","r_shoulder_lift_joint","r_upper_arm_roll_joint","r_elbow_flex_joint","r_forearm_roll_joint","r_wrist_flex_joint","r_wrist_roll_joint"] # eus rarm angle-vector order
        self.joint_names = config.control_joint_names

    def check_and_make_dir(self,dir_path):
        if False == os.path.exists(dir_path):
            os.makedirs(dir_path)
            print("make dir in path: {}".format(dir_path))

    def load_rosbag(self):
        self.check_and_make_dir(self.data_dir)
        bag_files = glob.glob(os.path.dirname(os.path.abspath(__file__)) + '/' + self.bag_dir +'*.bag')
        print(bag_files)

        angle_vector_list: List[AngleVector] = []
        rgb_image_list: List[RGBImage] = []
        for file_name in bag_files:
            bag = rosbag.Bag(file_name)

            bag_save_dir = self.data_dir+str(file_name[file_name.rfind('/')+1:])+"/"
            self.check_and_make_dir(bag_save_dir)
            bag_rgb_image_list = []
            bag_angle_vector_list = []

            # 最初の画像トピックの時間から1/hz秒後にその直前のデータを保存する．
            preb_time = None
            preb_img_msg = None
            next_save_time = None
            time_span = 1.0 / self.hz
            for topic, msg, t in bag.read_messages():
                if topic == self.config.image_topic:
                    t = msg.header.stamp
                    time = t.secs + 1e-9 * t.nsecs
                    if preb_time:
                        if time > next_save_time:
                            # print("save img topic time :{}, target time :{}".format(preb_time,next_save_time))
                            next_save_time += time_span

                            # 画像の保存 listに追加と画像でも保存．
                            pil_img = RGBImage.from_ros_msg(msg, self.image_config)
                            rgb_image_list.append(pil_img)
                            bag_rgb_image_list.append(pil_img)
                            img_file_name = bag_save_dir + str(preb_time) + ".png"
                            pil_img.pil_image().save(img_file_name)
                            # plt.imshow(pil_img)
                            # plt.draw() # グラフの描画
                            # plt.pause(0.01)

                            # jointの情報を保存
                            angles = AngleVector.from_ros_msg(preb_joints_msg, self.joint_names)
                            angle_vector_list.append(angles)
                            bag_angle_vector_list.append(angles)
                            # print(type(preb_joints_msg.position[0]))
                            # print(preb_joints_msg.position[0])
                    else:
                        next_save_time = time + time_span
                    preb_time = time
                    preb_img_msg = msg
                if topic == self.config.joint_states_topic:
                    preb_joints_msg = msg
            print("joint topics : {}, image topics : {}".format(len(bag_angle_vector_list), len(bag_rgb_image_list)))
            file_name = bag_save_dir + "joints.csv"
            with open(file_name, 'w') as f:
                writer =csv.writer(f)
                for angle_vector in bag_angle_vector_list:
                    writer.writerow(angle_vector.numpy().tolist())
            print("joint saved in {}".format(file_name))
            dump_file = bag_save_dir + "images.txt"
            f = open(dump_file,'wb')
            bag_rgb_image_pil_list = [i.pil_image() for i in bag_rgb_image_list]
            pickle.dump(bag_rgb_image_pil_list, f)
            f.close
            print("image also saved in {}".format(dump_file))

        # data saves
        file_name = self.data_dir + "joints.csv"
        with open(file_name, 'w') as f:
            writer =csv.writer(f)
            for angle_vector in angle_vector_list:
                writer.writerow(angle_vector.numpy().tolist())
        print("joint saved in {}".format(file_name))
        dump_file = self.data_dir + "images.txt"
        f = open(dump_file,'wb')
        rgb_image_pil_list = [i.pil_image() for i in rgb_image_list]
        pickle.dump(rgb_image_pil_list, f)
        f.close
        print("image also saved in {}".format(dump_file))

if __name__ == '__main__':
    config_file = sys.argv[sys.argv.index("-c") + 1] if "-c" in sys.argv else "../configs/rarm_pr2.yaml"
    # hz = int(sys.argv[sys.argv.index("-hz") + 1]) if "-hz" in sys.argv else 5
    bag_dir = sys.argv[sys.argv.index("-b") + 1] if "-b" in sys.argv else '../bags/'
    if bag_dir[-1:] != '/':
        bag_dir += '/'
    data_dir = sys.argv[sys.argv.index("-d") + 1] if "-d" in sys.argv else 'data/from_rosbag/'
    if data_dir[-1:] != '/':
        data_dir += '/'

    now_config = config_reader.construct_config(config_file)
    print("config!  hz:{}, image_x_min:{}, image_resolution:{}".format(now_config.rosbag_convert_hz, now_config.image_config.x_min, now_config.image_config.resolution))
    reader=RosbagReader(bag_dir,data_dir, now_config)
    reader.load_rosbag()
