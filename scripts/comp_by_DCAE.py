#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from network import DCAE

from PIL import Image
import glob
import csv
import pickle

class DCAECompresser(object):
    def __init__(self, model_dir, data_dir, z_dim):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.z_dim = z_dim
        print("model_dir : {}, z_dim: {}".format(self.model_dir, self.z_dim))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("device : {}".format(self.device))
        self.net = DCAE(channel=3, height=224, width=224, z_dim=z_dim).to(self.device)
        print(self.net)

        print("load data from {}".format(self.data_dir))
        
        dump_file = self.data_dir + "images.txt"
        f = open(dump_file,'rb')
        frames = pickle.load(f)
        f.close
        dataset = []
        for img in frames:
            img = img.resize((224,224))
            img_tensor = torchvision.transforms.functional.to_tensor(img)
            dataset.append(img_tensor)
        print(len(dataset))
        
        # dataset = []
        # for img_path in glob.glob(os.path.join(self.data_dir, '*png')):
        #     img = Image.open(img_path)
        #     img = img.resize((224,224))
        #     img_tensor = torchvision.transforms.functional.to_tensor(img)
        #     dataset.append(img_tensor)
        # print(len(dataset))

        self.dataset = dataset
        self.load_model()

    def load_model(self):
        model_path = self.model_dir + "model.pt"
        self.net.load_state_dict(torch.load(model_path))
        print("load model in {}".format(model_path))

    def compress(self):
        self.compress_data(self.data_dir,self.dataset)
        files = os.listdir(self.data_dir)
        files_dir = [f for f in files if os.path.isdir(os.path.join(self.data_dir, f))]
        print(files_dir)    # ['dir1', 'dir2']
        for dir_name in files_dir:
            now_dir = self.data_dir + dir_name + "/"
            dump_file = now_dir + "images.txt"
            f = open(dump_file,'rb')
            frames = pickle.load(f)
            f.close
            dataset = []
            for img in frames:
                img = img.resize((224,224))
                img_tensor = torchvision.transforms.functional.to_tensor(img)
                dataset.append(img_tensor)
            print(len(dataset))
            self.compress_data(now_dir,dataset)
            
    def compress_data(self,save_dir,dataset):
        self.net.eval()
        # z_list = []
        z_numpy = np.empty((0,self.z_dim))
        for i in range(len(dataset)):
            x = dataset[i]
            x = x.to(self.device).unsqueeze(0) # [784] -> [1, 784]
            y, z = self.net.forward(x)
            # z_list.append(z.cpu().detach().numpy())
            z_numpy = np.append(z_numpy,z.cpu().detach().numpy(),axis=0)
        # file_name = self.data_dir + "comp_image.csv"
        # with open(file_name, 'w') as f:
        #     writer =csv.writer(f)
        #     for z in z_list:
        #         print(type(z))
        #         print(z)
        #         writer.writerow(z)
        # print("compressed image saved in {}".format(file_name))
        np_file = save_dir + "comp_image.txt"
        np.savetxt(np_file, z_numpy)
        print("compressed image saved in {}".format(np_file))
        
if __name__ == '__main__':
    z_dim = int(sys.argv[sys.argv.index("-z") + 1]) if "-z" in sys.argv else 50
    data_dir = sys.argv[sys.argv.index("-d") + 1] if "-d" in sys.argv else "data/from_rosbag/"
    if data_dir[-1:] != '/':
        data_dir += '/'
    model_dir = sys.argv[sys.argv.index("-m") + 1] if "-m" in sys.argv else "../models/rosbag_DCAE/"
    if model_dir[-1:] != '/':
        model_dir += '/'
    compresser = DCAECompresser(model_dir, data_dir, z_dim)
    compresser.compress()
    
