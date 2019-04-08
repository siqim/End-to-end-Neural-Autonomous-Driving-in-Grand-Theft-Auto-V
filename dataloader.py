# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:26:25 2019

@author: msq96
"""


import os
import time
import gzip
import pickle

import torch
import torchvision


class GTAV(torch.utils.data.Dataset):

    def __init__(self, data_dir='./data/', datatype='train', batch_size=64, bin_fname='y_bin_info.pickle'):

        data_fp = data_dir + datatype + '/'
        y_bin_fname = data_dir + bin_fname

        with open(y_bin_fname, 'rb') as fin:
            y_bin = pickle.load(fin)

        self.data_file = [data_fp + each for each in os.listdir(data_fp)]
        self.y_bin = y_bin

        # [max_throttle, no_brake, go_straight, zero_speed]
        self.init_y = {}
        for key in y_bin.keys():
            if key == 'throttle':
                self.init_y[key] = torch.LongTensor([[len(y_bin['throttle']) - 1]]*batch_size)
            elif key == 'brake':
                self.init_y[key] = torch.LongTensor([[0]]*batch_size)
            elif key == 'steering':
                self.init_y[key] = torch.LongTensor([[len(y_bin['steering'])//2]]*batch_size)
            elif key == 'speed':
                self.init_y[key] = torch.LongTensor([[0]]*batch_size)

        self.y_keys_info = {k:len(v) for k, v in self.y_bin.items()}


    def __getitem__(self, index):

        with gzip.open(self.data_file[index], 'rb') as f:
            x_raw_y_dict = pickle.load(f)

        y = {}
        for k, v in x_raw_y_dict.items():
            if k == 'frame':
                x = v
            else:
                # Performance bottleneck
                y[k] = torch.zeros_like(v, dtype=torch.int64)
                for i in range(v.shape[0]):
                    for idx, each_bin in enumerate(self.y_bin[k], 0):
                        if each_bin['min'] <= v[i] <= each_bin['max']:
                            y[k][i] = idx
                            break
                # Performance bottleneck

        return x, y

    def __len__(self):
        return len(self.data_file)
