#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

from network import LSTM

from sklearn.model_selection import train_test_split
import csv

class LSTMrosbag(object):
    def __init__(self, epoch, model_dir, data_dir, hidden_size, z_dim):
        self.epoch = epoch
        self.hidden_size = hidden_size
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.z_dim = z_dim
        self.data_dims = self.z_dim + 7
        print("epoch : {} , hidden_size : {}, model_dir : {}, data_dir : {}".format(self.epoch,self.hidden_size,self.model_dir,self.data_dir))
        self.check_and_make_dir()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        print("device : {}".format(self.device))
        self.net = LSTM(input_size=self.data_dims, output_size=self.data_dims, hidden_size=self.hidden_size, batch_first=True).to(self.device)
        print(self.net)

        self.data_preparation()
        
    def data_preparation(self):

        files = os.listdir(self.data_dir)
        files_dir = [f for f in files if os.path.isdir(os.path.join(self.data_dir, f))]
        print(files_dir)    # ['dir1', 'dir2']
        np_image_list = []
        joints_lists_list = []
        max_length = 0
        for dir_name in files_dir:
            now_dir = self.data_dir + dir_name + "/"
            np_file = now_dir + "comp_image.txt"
            np_image = np.loadtxt(np_file)
            np_image_list.append(np_image)
            joints_file = now_dir + "joints.csv"
            joints_lists = []
            with open(joints_file) as f:
                reader = csv.reader(f)
                for row in reader:
                    row = [float (v) for v in row]
                    joints_lists.append(row)
            # print(len(image_lists),len(joints_lists))
            joints_lists_list.append(joints_lists)
            if len(np_image)>max_length:
                max_length=len(np_image)
            print("image data num : {}, joints data num : {}".format(len(np_image),len(joints_lists)))
            
        # make dataset
        self.data_length = max_length
        dataset_x = []
        dataset_t = []
        for np_img, joint_list in zip(np_image_list,joints_lists_list):
            # concate image and joints
            np_joints = np.array(joint_list)
            np_data = np.concatenate([np_img,np_joints],1)
            np_data_cov = np.cov(np_data,rowvar=0)
            np_data_noise = np.random.multivariate_normal(np.zeros(self.data_dims),np_data_cov,1)*0.1
            # print(np_data_noise.shape)
            list_data = np_data.tolist()
            # add to dataset           
            # no noise
            dataset_x.append(torch.tensor(list_data[:-1]))
            dataset_t.append(torch.tensor(list_data[1:]))
            # x noise small
            dataset_x.append(torch.tensor([(np.array(list_data[i]) +  np.random.multivariate_normal(np.zeros(self.data_dims),np_data_cov,1)[0]*0.1).tolist() for i in range(len(list_data)-1)]))
            dataset_t.append(torch.tensor([list_data[i+1] for i in range(len(list_data)-1)]))
            # x noise small
            dataset_x.append(torch.tensor([(np.array(list_data[i]) +  np.random.multivariate_normal(np.zeros(self.data_dims),np_data_cov,1)[0]*0.2).tolist() for i in range(len(list_data)-1)]))
            dataset_t.append(torch.tensor([list_data[i+1] for i in range(len(list_data)-1)]))

        train_x, test_x, train_t, test_t = train_test_split(dataset_x, dataset_t, test_size=0.1)
        self.train_x = torch.nn.utils.rnn.pad_sequence(train_x,batch_first=True)
        self.train_t = torch.nn.utils.rnn.pad_sequence(train_t,batch_first=True)
        self.test_x = torch.nn.utils.rnn.pad_sequence(test_x,batch_first=True)
        self.test_t = torch.nn.utils.rnn.pad_sequence(test_t,batch_first=True)
        print(self.train_x.size())
        print(self.train_x.size()[1])
        print(self.test_x.size())
        print(self.test_x.size()[1])
        print("train data num : {}, test data num : {} ".format(len(self.train_x),len(self.test_x)))
        
    def check_and_make_dir(self):
        model_dir = self.model_dir
        if False == os.path.exists(model_dir):
            os.makedirs(model_dir)
            print("make dir in path: {}".format(model_dir))
        
    def save_model(self):
        model_path = self.model_dir + "model.pt"
        torch.save(self.net.to('cpu').state_dict(), model_path)
        print("model saved in {}".format(model_path))
        
    def train(self):
        batch_size = 10
        optimizer = torch.optim.Adam(self.net.parameters(),lr=0.01)
        criterion = torch.nn.MSELoss()
        
        # train and val
        t_accuracies = []
        v_accuracies = []
        t_losses = []
        for e in range(self.epoch):
            # train
            running_loss = 0.0
            training_accuracy = 0.0
            self.net.train()
            for i in range(1):
            # for i in range(len(self.train_x)):
                optimizer.zero_grad()
                
                data, label = self.train_x, self.train_t
                data = data.to(self.device)

                # print(data.shape,label.shape)
                output, state = self.net(data)
                # print(output.shape)
                output = output.cpu()

                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)

            # val
            self.net.eval()
            test_accuracy = 0.0
            # for i in range(int(len(self.test_x))):
            for i in range(1):
                data, label = self.test_x , self.test_t
                data = data.to(self.device)

                output,state = self.net(data)
                output = output.cpu()

                test_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)

            training_accuracy /= (self.train_t.size()[0] * self.train_t.size()[1] * self.data_dims)
            test_accuracy /= (self.test_t.size()[0] * self.test_t.size()[1] * self.data_dims)
            t_accuracies.append(training_accuracy)
            v_accuracies.append(test_accuracy)
            t_losses.append(running_loss)
            if e%100 == 99:
                print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
                    e + 1, running_loss, training_accuracy, test_accuracy))

        fig = plt.figure()
        plt.plot(t_accuracies, label="train")
        plt.plot(v_accuracies, label="valid")
        log_path = self.model_dir + "log.png"
        fig.legend()
        fig.savefig(log_path)
        print("log image saved in {}".format(log_path))
        loss_fig = plt.figure()
        plt.yscale("log")
        plt.plot(t_losses, label="train")
        loss_log_path = self.model_dir + "loss_log.png"
        loss_fig.legend()
        loss_fig.savefig(loss_log_path)
        print("loss log image saved in {}".format(loss_log_path))
        plt.show()
        plt.clf()

if __name__ == '__main__':
    epoch = int(sys.argv[sys.argv.index("-e") + 1]) if "-e" in sys.argv else 5000
    z_dim = int(sys.argv[sys.argv.index("-z") + 1]) if "-z" in sys.argv else 50
    hidden_size = int(sys.argv[sys.argv.index("-h") + 1]) if "-h" in sys.argv else 50
    data_dir = sys.argv[sys.argv.index("-d") + 1] if "-d" in sys.argv else "data/from_rosbag/"
    if data_dir[-1:] != '/':
        data_dir += '/'
    model_dir = sys.argv[sys.argv.index("-m") + 1] if "-m" in sys.argv else "../models/rosbag_LSTM_series_batch/"
    if model_dir[-1:] != '/':
        model_dir += '/'
    trainer = LSTMrosbag(epoch, model_dir, data_dir, hidden_size, z_dim)
    trainer.train()
    trainer.save_model()
