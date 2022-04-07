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
import pickle

class DCAETester(object):
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

    def view_result(self):
        # original image plot
        fig1 = plt.figure(figsize=(10, 10))
        fig1.suptitle('Original Image', fontsize=16)        
        for i in range(20):
            x = self.dataset[i]
            im = x.view(-1, 224, 224).permute(1, 2, 0).squeeze().numpy()
            ax = fig1.add_subplot(10, 10, i+1, xticks=[], yticks=[])
            ax.imshow(im, 'gray')
        fig1.tight_layout(rect=[0,0,1,0.96])
        
        # reconstruction image plot
        fig2 = plt.figure(figsize=(10, 10))
        fig2.suptitle('Reconstruction Image', fontsize=16)
        self.net.eval()
        for i in range(20):
            x = self.dataset[i]
            x = x.to(self.device).unsqueeze(0) # [784] -> [1, 784]
            y, z = self.net.forward(x)
            im = y.view(-1, 224, 224).permute(1, 2, 0).cpu().squeeze().detach().numpy()
            ax = fig2.add_subplot(10, 10, i+1, xticks=[], yticks=[])
            ax.imshow(im, 'gray')
        fig2.tight_layout(rect=[0,0,1,0.96])
        log1_path = self.model_dir + "image_original.png"
        fig1.savefig(log1_path)
        print("original image saved in {}".format(log1_path))
        log2_path = self.model_dir + "image_reconstruction.png"
        fig2.savefig(log2_path)
        print("reconstruction image saved in {}".format(log2_path))
        plt.show()
        plt.clf()
        
if __name__ == '__main__':
    z_dim = int(sys.argv[sys.argv.index("-z") + 1]) if "-z" in sys.argv else 50
    data_dir = sys.argv[sys.argv.index("-d") + 1] if "-d" in sys.argv else "data/from_rosbag/"
    if data_dir[-1:] != '/':
        data_dir += '/'
    model_dir = sys.argv[sys.argv.index("-m") + 1] if "-m" in sys.argv else "../models/rosbag_DCAE/"
    if model_dir[-1:] != '/':
        model_dir += '/'
    tester = DCAETester(model_dir, data_dir, z_dim)
    tester.view_result()
