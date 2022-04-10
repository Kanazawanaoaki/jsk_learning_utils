#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import math
import numpy as np

from PIL import Image
import csv
import pickle

from network import DCAE, LSTM

import sys
import torch
import torchvision

from jsk_learning_utils.project_data import get_dataset_dir
from jsk_learning_utils.project_data import get_project_dir


class DCAECompOne(object):
  def __init__(self, model_dir, z_dim):
    self.model_dir = model_dir
    self.z_dim = z_dim
    print("model_dir : {}, z_dim: {}".format(self.model_dir, self.z_dim))
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device : {}".format(self.device))
    self.net = DCAE(channel=3, height=224, width=224, z_dim=z_dim).to(self.device)
    print(self.net)
    self.load_model()
    self.net.eval()

  def load_model(self):
    model_path = os.path.join(self.model_dir, "model.pt")
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
    model_path = os.path.join(self.model_dir, "model.pt")
    self.net.load_state_dict(torch.load(model_path))
    print("load model in {}".format(model_path))

  def pred_one(self, data):
    self.net.eval()
    x = torch.tensor(data)
    x = x.to(self.device).unsqueeze(0) # [784] -> [1, 784]
    output, state = self.net(x)
    return output.cpu().detach().numpy()


def save_video(frames, path):
  clip = mpy.ImageSequenceClip(frames, fps=30)
  clip.write_gif(path, fps=30)

def check_and_make_dir(data_path):
  if False == os.path.exists(data_path):
    os.makedirs(data_path)
    print("make dir in path: {}".format(data_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', action='store_true', help='show debug view')
    parser.add_argument('-project', type=str, help='project name')
    parser.add_argument('-z', type=int, default=50, help='latent dim')
    parser.add_argument('-l', type=int, default=50, help='data length')
    parser.add_argument('-hidden', type=int, default=50, help='hidden size')

    args = parser.parse_args()
    z_dim = args.z
    project_name = args.project
    data_length = args.l
    hidden_size = args.hidden
    view_flag = args.view

    print("data_length : {}, hidden_size : {}".format(data_length,hidden_size))
    lstm_model_dir = os.path.join(get_project_dir(project_name), 'lstm_h{}'.format(hidden_size))
    dcae_model_dir = os.path.join(get_project_dir(project_name), 'dcae_z{}'.format(z_dim))

    base_dir = get_dataset_dir(project_name)
    for test_data_dir_tmp in os.listdir(base_dir):
        test_data_dir = os.path.join(base_dir, test_data_dir_tmp)
        if not os.path.isdir(test_data_dir):
            continue
        movie_file_prefix, _ = os.path.splitext(test_data_dir_tmp)
        compresser = DCAECompOne(dcae_model_dir, z_dim)
        preder = LSTMPred(lstm_model_dir, hidden_size, z_dim)
        movie_dir = os.path.join(get_project_dir(project_name), 'movies')
        check_and_make_dir(movie_dir)
        movie_file = os.path.join(movie_dir, movie_file_prefix + ".gif")
        reconstruction_movie_file = os.path.join(movie_dir, movie_file_prefix + "_reconstruction.gif")
      
        # 画像と関節角度を読み込む 最初の物を初期とする
        np_file = os.path.join(test_data_dir, "comp_image.txt")
        np_image = np.loadtxt(np_file)
        joints_file = os.path.join(test_data_dir, "joints.csv")
        joints_lists = []
        with open(joints_file) as f:
          reader = csv.reader(f)
          for row in reader:
            row = [float (v) for v in row]
            joints_lists.append(row)
        # print(type(img_lists))
        # rgbImg = compresser.dec_one(np.expand_dims(img_lists, axis = 0))[0].transpose(1,2,0)
        # img = Image.fromarray(img_lists)
        # img = frames[0].resize((224,224))
        # plt.imshow(img); plt.show()
        input_data = []
        frames = []
        reconstruction_frames = []
        for i in range(len(np_image)):
          img_lists = np_image[i]
          cmd_lists = joints_lists[i]
          
          # print(type(img_lists),type(cmd_lists))
          # print(img_lists)
          input_data.append(img_lists.tolist() + cmd_lists)
          # print(type(input_data))
      
          if data_length > 0:
            # data len limit
            if len(input_data) <= data_length:
              output = preder.pred_one(input_data)
            else:
              output = preder.pred_one(input_data[-1*data_length:])
          else:
            # without data len limit
            output = preder.pred_one(input_data)
          # print(output[0].shape)
      
          # print("input data : {}".format(input_data[-1][-7:]))
          print("output data : {}".format(output[0][-1][-7:]))
          cmd_lists = output[0][-1][-7:].tolist()
          img_lists = output[0][-1][:-7]
          # print("output img data : {}".format(output[0][-1][:-7]))
          # print("original img data : {}".format(np_image[i]))
      
          rgbImg = compresser.dec_one(np.expand_dims(img_lists, axis = 0))[0].transpose(1,2,0)
          # rgbImg *= 255
          frames.append(rgbImg*255)
          if view_flag:
            plt.imshow(rgbImg)
            plt.draw() # グラフの描画
            plt.pause(0.01)
      
          reconstructionRgbImg = compresser.dec_one(np.expand_dims(np_image.astype('float32')[i], axis = 0))[0].transpose(1,2,0)
          reconstruction_frames.append(reconstructionRgbImg*255)
      
        save_video(frames, movie_file)
        save_video(reconstruction_frames, reconstruction_movie_file)
