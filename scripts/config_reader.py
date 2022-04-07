#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys
import yaml

class ImageConfig(object):
    def __init__(self, img_dict):
        self.x_min = img_dict['x_min']
        self.x_max = img_dict['x_max']
        self.y_min = img_dict['y_min']
        self.y_max = img_dict['y_max']
        self.resolution = img_dict['resolution']

class Config(object):
    def __init__(self,
            rosbag_convert_hz,
            control_joint_names,
            init_joint_names,
            init_joint_angles,
            image_config,
            ):
        self.rosbag_convert_hz = rosbag_convert_hz
        self.control_joint_names = control_joint_names
        self.init_joint_names = init_joint_names
        self.init_joint_angles = init_joint_angles
        self.image_config = ImageConfig(image_config)

def construct_config(config_file):
    with open(config_file) as f:
        dic = yaml.safe_load(f)
    config = Config(
            rosbag_convert_hz = dic['rosbag_convert_hz'],
            control_joint_names = dic['control_joint_names'],
            init_joint_names = dic['init_joint_names'],
            init_joint_angles = dic['init_joint_angles'],
            image_config = dic['image_config']
            )
    return config

if __name__ == '__main__':
    config_file = sys.argv[sys.argv.index("-c") + 1] if "-c" in sys.argv else "../configs/rarm_pr2.yaml"
    now_config = construct_config(config_file)
