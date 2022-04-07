#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import matplotlib.pyplot as plt
import math
import numpy as np

from PIL import Image
import csv
import pickle

from network import DCAE, LSTM

import sys
import torch
import torchvision

import rospy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64MultiArray
import sensor_msgs.msg
import cv2

import config_reader

class DCAECompOne(object):
  def __init__(self, model_dir, z_dim):
    self.model_dir = model_dir
    self.z_dim = z_dim
    print("model_dir : {}, z_dim: {}".format(self.model_dir, self.z_dim))
    # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.device = torch.device('cpu')
    print("device : {}".format(self.device))
    self.net = DCAE(channel=3, height=224, width=224, z_dim=z_dim).to(self.device)
    print(self.net)
    self.load_model()
    self.net.eval()

  def load_model(self):
    model_path = self.model_dir + "model.pt"
    self.net.load_state_dict(torch.load(model_path))
    print("load model in {}".format(model_path))

  def comp_one(self, data):
    img = Image.fromarray(data)
    img = img.resize((224,224))
    x = torchvision.transforms.functional.to_tensor(img)
    x = x.to(self.device).unsqueeze(0) # [784] -> [1, 784]
    y, z = self.net.forward(x)
    return z.cpu().detach().numpy()

  def dec_one(self, z_input):
    z_input = torch.from_numpy(z_input).to(self.device)
    x = self.net._decoder(z_input)
    return x.cpu().detach().numpy()

class LSTMPred(object):
  def __init__(self, model_dir, hidden_size, z_dim):
    self.model_dir = model_dir
    self.z_dim = z_dim
    self.data_dims = self.z_dim + 7
    print("model_dir : {}, data_dim : {}".format(self.model_dir, self.data_dims))
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device : {}".format(self.device))
    self.net = LSTM(input_size=self.data_dims, output_size=self.data_dims, hidden_size=hidden_size, batch_first=True).to(self.device)
    print(self.net)
    self.load_model()
    self.net.eval()

  def load_model(self):
    model_path = self.model_dir + "model.pt"
    self.net.load_state_dict(torch.load(model_path))
    print("load model in {}".format(model_path))

  def pred_one(self, data):
    self.net.eval()
    x = torch.tensor(data)
    x = x.to(self.device).unsqueeze(0) # [784] -> [1, 784]
    output, state = self.net(x)
    return output.cpu().detach().numpy()

class PR2Imitation(object):
  def __init__(self, comp_model, pred_model, z_dim, hidden_size, data_length, config):
    self.compresser = DCAECompOne(comp_model, z_dim)
    self.preder = LSTMPred(pred_model, hidden_size, z_dim)
    self.data_length = data_length
    self.hz = config.rosbag_convert_hz
    self.image_config = config.image_config

    self.img = None
    self.img_sub = rospy.Subscriber("/kinect_head/rgb/image_rect_color/compressed", sensor_msgs.msg.CompressedImage, self.img_cb, queue_size=1)
    self.av = None
    self.state_sub = rospy.Subscriber("/joint_states", sensor_msgs.msg.JointState, self.state_cb, queue_size=1)
    self.bridge = CvBridge()
    # self.joint_names = ["r_shoulder_pan_joint","r_shoulder_lift_joint","r_upper_arm_roll_joint","r_elbow_flex_joint","r_forearm_roll_joint","r_wrist_flex_joint","r_wrist_roll_joint"] # eus rarm angle-vector order
    self.joint_names = config.control_joint_names

    self.pred_img_pub = rospy.Publisher("/imitation/pred_img", sensor_msgs.msg.Image, queue_size=1)
    self.reconstruction_img_pub = rospy.Publisher("/imitation/reconstruction_img", sensor_msgs.msg.Image, queue_size=1)
    self.cmd_pub = rospy.Publisher("/imitation/command", Float64MultiArray, queue_size=1)
    print("finish PR2Imitation initialization")

  def img_cb(self, msg):
    self.img = self.bridge.compressed_imgmsg_to_cv2(msg)

  def state_cb(self, msg):
    self.av = self.get_angle_from_joint_states(msg)

  def clamp_rad_list(self,rad_list):
    min_val = -1 * math.pi
    max_val = math.pi
    rad_list = map(lambda x: min_val if x < min_val else x, rad_list)
    rad_list = map(lambda x: max_val if x > max_val else x, rad_list)
    return rad_list

  def angle_vector_rad2deg(self,angle_vector):
    return [math.degrees(rad) for rad in angle_vector]

  def get_angle_from_joint_states(self, joint_states_msg):
    # とりあえず右腕のangle-vectorを取り出す
    joint_rads = []
    for j_name in self.joint_names:
      idx = joint_states_msg.name.index(j_name)
      joint_rads.append(joint_states_msg.position[idx])
    joint_rads = self.clamp_rad_list(joint_rads)
    joint_angles = self.angle_vector_rad2deg(joint_rads)
    return joint_angles

  def execute(self):
    input_data = []
    while True:
      if type(self.img) == type(None) or self.av == None:
        continue
      ## TODO resize and cv2 color problem
      img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
      # crop img tmp !!!!!
      img_rgb = img_rgb[self.image_config.x_min:self.image_config.x_max, self.image_config.y_min:self.image_config.y_max, :]
      # plt.imshow(img_rgb)
      # plt.draw() # グラフの描画
      # plt.pause(0.01)
      comp_img = self.compresser.comp_one(img_rgb)

      # publish reconstruction image for debug
      # reconstruction_img = self.compresser.dec_one(comp_img)[0].transpose(1,2,0) * 255
      reconstruction_img = self.compresser.dec_one(comp_img)[0].transpose(1,2,0)
      reconstruction_img_cv2 = cv2.cvtColor((reconstruction_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
      reconstruction_img_msg = self.bridge.cv2_to_imgmsg(reconstruction_img_cv2, encoding='bgr8')
      reconstruction_img_msg.header.stamp = rospy.Time.now()
      self.reconstruction_img_pub.publish(reconstruction_img_msg)

      input_data.append(comp_img[0].tolist() + self.av)
      # pred image and angle-vector
      if self.data_length > 0:
        # data len limit
        if len(input_data) <= self.data_length:
          output = self.preder.pred_one(input_data)
        else:
          output = self.preder.pred_one(input_data[-1*self.data_length:])
      else:
        # without data len limit
        output = self.preder.pred_one(input_data)

      print("output command : {}".format(output[0][-1][-7:]))
      cmd_lists = output[0][-1][-7:].tolist()
      img_lists = output[0][-1][:-7]

      # publish command and image
      cmd_msg = Float64MultiArray()
      cmd_msg.data = cmd_lists
      self.cmd_pub.publish(cmd_msg)
      pred_img = self.compresser.dec_one(np.expand_dims(img_lists, axis = 0))[0].transpose(1,2,0)
      print(pred_img)
      print(pred_img.shape)
      print(pred_img.dtype)
      print(type(pred_img))
      if view_flag:
        plt.imshow(pred_img)
        plt.draw() # グラフの描画
        plt.pause(0.01)
      pred_img_cv2 = cv2.cvtColor((pred_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
      pred_img_msg = self.bridge.cv2_to_imgmsg(pred_img_cv2, encoding='bgr8')
      pred_img_msg.header.stamp = rospy.Time.now()
      self.pred_img_pub.publish(pred_img_msg)


if __name__ == '__main__':
  config_file = sys.argv[sys.argv.index("-c") + 1] if "-c" in sys.argv else "../configs/rarm_pr2.yaml"
  z_dim = int(sys.argv[sys.argv.index("-z") + 1]) if "-z" in sys.argv else 50
  data_length = int(sys.argv[sys.argv.index("-l") + 1]) if "-l" in sys.argv else 30
  hidden_size = int(sys.argv[sys.argv.index("-h") + 1]) if "-h" in sys.argv else 50
  view_flag = True if "-v" in sys.argv else False
  print("data_length : {}, hidden_size : {}".format(data_length,hidden_size))
  pred_model = sys.argv[sys.argv.index("-p") + 1] if "-p" in sys.argv else "../models/rosbag_LSTM_dataset_from_series/"
  if pred_model[-1:] != '/':
    pred_model += '/'
  comp_model = sys.argv[sys.argv.index("-m") + 1] if "-m" in sys.argv else "../models/rosbag_DCAE/"
  if comp_model[-1:] != '/':
    comp_model += '/'

  # load config
  now_config = config_reader.construct_config(config_file)

  # ROSの設定
  rospy.init_node('pr2_imitation_play', anonymous=True)
  imitation = PR2Imitation(comp_model, pred_model, z_dim, hidden_size, data_length, now_config)

  # 模倣を開始
  imitation.execute()
