#!/usr/bin/env python3
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
import rospkg

from cv_bridge import CvBridge, CvBridgeError
import cv2
import PIL.Image

from jsk_learning_utils.config import Config 
from jsk_learning_utils.config import construct_config
from jsk_learning_utils.project_data import get_project_dir
from jsk_learning_utils.project_data import get_rosbag_dir
from jsk_learning_utils.project_data import get_dataset_dir
from jsk_learning_utils.project_data import get_kanazawa_specific_rosbag_dir
from jsk_learning_utils.project_data import get_kanazawa_specific_rosbag_dir


@dataclass
class AngleVector:
    data: np.ndarray

    @classmethod
    def from_ros_msg(cls, msg: JointState, config: Config) -> 'AngleVector':
        joint_angles = []
        for j_dict in config.control_joints:
            idx = msg.name.index(j_dict["name"])
            if j_dict["type"] == "revolute":
                joint_angles.append(msg.position[idx])
            elif j_dict["type"] == "prismatic":
                joint_angles.append(msg.position[idx])
            elif j_dict["type"] == "continuous":
                if j_dict["clamp"]:
                    joint_angles.append(cls.clamp_rad(msg.position[idx]))
                else:
                    joint_angles.append(msg.position[idx])
            else:
                assert False, 'joint type {} is not expected'.format(j_dict["type"])
        np_joint_angles = np.array(joint_angles)
        return cls(np_joint_angles)

    @staticmethod
    def clamp_rad(rad) -> float:
        min_val = -1 * math.pi
        max_val = math.pi
        rad = min_val if rad < min_val else rad
        rad = max_val if rad > max_val else rad
        return rad

    def get_numpy(self) -> np.ndarray:
        return self.data

@dataclass
class RGBImage:
    data: np.ndarray

    @classmethod
    def from_ros_msg(cls, msg: Union[CompressedImage, Image], config: Config) -> 'RGBImage':
        bridge = CvBridge()
        cv2_img = bridge.compressed_imgmsg_to_cv2(msg)
        cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        cv2_img_rgb_cropped = cv2_img_rgb[config.image_config.x_min:config.image_config.x_max, config.image_config.y_min:config.image_config.y_max, :]
        return cls(cv2_img_rgb_cropped)

    def get_numpy(self) -> np.ndarray:
        return self.data

class RosbagReader(object):
    def __init__(self, bag_dir, data_dir, config, project_name):
        self.project_name = project_name
        self.bag_dir = bag_dir
        self.data_dir = data_dir
        self.hz = config.rosbag_convert_hz
        self.config = config
        print("hz : {}, bag_dir : {}, data_dir : {}".format(self.hz, self.bag_dir, self.data_dir))

    def load_rosbag(self):
        angle_vector_list: List[AngleVector] = []
        rgb_image_list: List[RGBImage] = []

        for file_name in os.listdir(self.bag_dir):
            _, ext =  os.path.splitext(file_name)
            if ext != ".bag":
                continue

            bag = rosbag.Bag(os.path.join(self.bag_dir, file_name))
            bag_save_dir = get_kanazawa_specific_rosbag_dir(self.project_name, file_name)

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
                            rgb_image = RGBImage.from_ros_msg(msg, self.config)
                            rgb_image_list.append(rgb_image)
                            bag_rgb_image_list.append(rgb_image)
                            img_file_name = os.path.join(bag_save_dir, str(preb_time) + ".png")
                            PIL.Image.fromarray(rgb_image.get_numpy()).save(img_file_name)

                            # jointの情報を保存
                            angles = AngleVector.from_ros_msg(preb_joints_msg, self.config)
                            angle_vector_list.append(angles)
                            bag_angle_vector_list.append(angles)
                    else:
                        next_save_time = time + time_span
                    preb_time = time
                    preb_img_msg = msg
                if topic == self.config.joint_states_topic:
                    preb_joints_msg = msg
            print("joint topics : {}, image topics : {}".format(len(bag_angle_vector_list), len(bag_rgb_image_list)))
            file_name = os.path.join(bag_save_dir, "joints.csv")
            with open(file_name, 'w') as f:
                writer =csv.writer(f)
                for angle_vector in bag_angle_vector_list:
                    writer.writerow(angle_vector.get_numpy().tolist())
            print("joint saved in {}".format(file_name))
            dump_file = os.path.join(bag_save_dir, "images.txt")
            f = open(dump_file,'wb')
            bag_rgb_image_pil_list = [PIL.Image.fromarray(i.get_numpy()) for i in bag_rgb_image_list]
            pickle.dump(bag_rgb_image_pil_list, f)
            f.close
            print("image also saved in {}".format(dump_file))

        # data saves
        file_name = os.path.join(self.data_dir, "joints.csv")
        with open(file_name, 'w') as f:
            writer =csv.writer(f)
            for angle_vector in angle_vector_list:
                writer.writerow(angle_vector.get_numpy().tolist())
        print("joint saved in {}".format(file_name))
        rgb_image_pil_list = [PIL.Image.fromarray(i.get_numpy()) for i in rgb_image_list]

        dump_file = os.path.join(self.data_dir, "images.txt")
        with open(dump_file, 'wb') as f:
            pickle.dump(rgb_image_pil_list, f)
        print("image also saved in {}".format(dump_file))

if __name__ == '__main__':
    project_name = 'sample_rcup_pick'
    pkg_path = rospkg.RosPack().get_path('jsk_learning_utils')
    config_file = os.path.join(pkg_path, 'configs', 'rarm_pr2.yaml')
    bag_dir = get_rosbag_dir(project_name)
    data_dir = get_dataset_dir(project_name)

    now_config = construct_config(config_file)
    print("config!  hz:{}, image_x_min:{}, image_resolution:{}".format(now_config.rosbag_convert_hz, now_config.image_config.x_min, now_config.image_config.resolution))
    reader=RosbagReader(bag_dir, data_dir, now_config, project_name)
    reader.load_rosbag()
