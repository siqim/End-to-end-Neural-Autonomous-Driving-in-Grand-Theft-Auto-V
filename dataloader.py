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


def collate_fn(batch):
    """
        Params:
            batch: [(tensor, dict)]

        Returns:
            x: shape = batch_size  x num_channel x height x width
            y: dict{action: tensor}
    """

    raw_x, raw_y = zip(*batch)

    x = torch.stack(raw_x)
    y= {action: torch.LongTensor([each_y[action] for each_y in raw_y]) for action in raw_y[0].keys()}
    return x, y


class GTAV(torch.utils.data.Dataset):

    def __init__(self, data_dir='./data/', datatype='train', bin_fname='y_bin_info.pickle'):

        data_fp = data_dir + datatype + '/'
        y_bin_fname = data_dir + bin_fname

        with open(y_bin_fname, 'rb') as fin:
            y_bin = pickle.load(fin)

        self.data_file = [data_fp + each for each in os.listdir(data_fp)]
        self.y_bin = y_bin

        self.y_keys_info = {k:len(v) for k, v in self.y_bin.items() if k != 'brake'}

        self.weight_info = {k: [1/each['num_samples'] for each in v]  for k, v in self.y_bin.items() if k!='brake'}

    def __getitem__(self, index):

        with gzip.open(self.data_file[index], 'rb') as f:
            x_raw_y_dict = pickle.load(f)

        y = {}
        for k, v in x_raw_y_dict.items():
            if k == 'frame':
                x = v
            elif k != 'brake':
                # Performance no good
                for idx, each_bin in enumerate(self.y_bin[k], 0):
                    if each_bin['min'] <= v.item() <= each_bin['max']:
                        y[k] = torch.LongTensor([idx])
                        break
                # Performance no good

        return x, y

    def __len__(self):
        return len(self.data_file)
