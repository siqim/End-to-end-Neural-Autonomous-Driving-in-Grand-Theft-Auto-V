# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:26:25 2019

@author: msq96
"""

import os
import gzip
import pickle
import numpy as np

import torch
import torchvision

class GTAV(torch.utils.data.Dataset):

    def __init__(self, data_dir='./data/', datatype='train', bin_fname='y_bin.pickle'):

        data_fp = data_dir + datatype + '/'
        y_bin_fname = data_dir + bin_fname

        with open(y_bin_fname, 'rb') as fin:
            y_bin = pickle.load(fin)
            for k, v in y_bin.items():
                y_bin[k] = [{'min': np.min(each_bin), 'mean': np.mean(each_bin), 'max': np.max(each_bin)} for each_bin in v]
                temp = []
                for each in y_bin[k]:
                    if each not in temp:
                        temp.append(each)
                y_bin[k] = temp

        self.data_file = [data_fp + each for each in os.listdir(data_fp)]
        self.y_bin = y_bin

    def __getitem__(self, index):

        with gzip.open(self.data_file[0], 'rb') as f:
            x_raw_y_dict = pickle.load(f)

        y = {}
        for k, v in x_raw_y_dict.items():
            if k == 'frame':
                x = v
            else:

                y[k] = torch.zeros_like(v, dtype=torch.int64)
                for i in range(v.shape[0]):
                    for idx, each_bin in enumerate(self.y_bin[k], 0):
                        if each_bin['min'] <= v[i] <= each_bin['max']:
                            y[k][i] = idx
                            break
        return x, y

    def __len__(self):
        return len(self.data_file)
