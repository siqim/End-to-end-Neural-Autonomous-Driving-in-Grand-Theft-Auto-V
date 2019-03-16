# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:26:25 2019

@author: msq96
"""

import os
import pickle

import torch
import torchvision

class GTAV(torch.utils.data.Dataset):

    def __init__(self, file_dir, transform=None):

        self.data_file = [file_dir+each for each in os.listdir(file_dir)]
        self.transform = transform

    def __getitem__(self, index):

        with open(self.data_file[index], 'rb') as f:
            screen, controller_data = pickle.load(f)

        if self.transform:
            screen = self._transform(screen)

        return screen, torch.FloatTensor(controller_data)

    def __len__(self):
        return len(self.data_file)

    def _transform(self, screen):
        return torch.stack([self.transform(each) for each in screen])
