#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys
import pickle
import cv2
from moviepy.editor import ImageSequenceClip
from mohou.types import RGBImage, DepthImage, AngleVector
from mohou.types import ElementSequence, EpisodeData, MultiEpisodeChunk
import csv
import numpy as np


data_dir = sys.argv[sys.argv.index("-d") + 1] if "-d" in sys.argv else 'data/rcup_20220218_pick/'
if data_dir[-1:] != '/':
    data_dir += '/'
dump_name = sys.argv[sys.argv.index("-n") + 1] if "-n" in sys.argv else 'rcup_20220218_pick'
if dump_name[-1:] != '/':
    dump_name += '/'

files = os.listdir(data_dir)
files_dir = [f for f in files if os.path.isdir(os.path.join(data_dir, f))]
print(files_dir)    # ['dir1', 'dir2']
np_image_list = []
joints_lists_list = []
max_length = 0

cv2_list_list = []
av_list_list = []
for dir_name in files_dir:
    now_dir = os.path.join(data_dir ,dir_name)
    print(now_dir)
    img_files = []
    for file_name in os.listdir(now_dir):
        _, ext = os.path.splitext(file_name)
        if (ext != ".png"):
            continue
        img_files.append(file_name)

    print(sorted(img_files))
    cv2_img_list = []
    for img_file_name in sorted(img_files):
        cv2_img = cv2.imread(os.path.join(now_dir,img_file_name))
        cv2_img_list.append(cv2_img)

    # filename =  os.path.join(now_dir,"debug.gif")
    # clip = ImageSequenceClip(cv2_img_list,fps=50)
    # clip.write_gif(filename, fps=50)

    cv2_list_list.append(cv2_img_list)

    joints_file = os.path.join(now_dir, "joints.csv")
    av_list = []
    with open(joints_file) as f:
        reader = csv.reader(f)
        for row in reader:
            row = [float (v) for v in row]
            av_np = np.array(row)
            av_list.append(av_np)

    av_list_list.append(av_list)

print(len(cv2_list_list), len(av_list_list))

## mohou
episodedata_list = []
for img_list, angles_list in zip(cv2_list_list, av_list_list):
    rgb_seq = ElementSequence()
    av_seq = ElementSequence()
    for img, angles in zip(img_list, av_list):
        rgb = RGBImage(img)
        rgb.resize((224,224))
        av = AngleVector(angles)

        rgb_seq.append(rgb)
        av_seq.append(av)

    episodedata_list.append(EpisodeData((rgb_seq, av_seq)))

chunk = MultiEpisodeChunk(episodedata_list)
chunk.dump(dump_name)  # dumps to ~/.mohou/(dump_name)/MultiEpisodeChunk.pkl
