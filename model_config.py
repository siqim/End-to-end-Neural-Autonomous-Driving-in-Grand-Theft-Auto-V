# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:52:45 2019

@author: msq96
"""



import torch
from utils import try_mkdir
from dataloader import GTAV, collate_fn


class Config(object):

#    manual_change = True
    manual_change = False
    lr = 4e-4
    wd = 4e-4

    DEBUG = False
    MODEL_NAME = 'xception'
    ENCODER_NAME = 'xception'
    EPOCH = 10
    NUM_WORKERS = 0

    batch_size = 16

    decoder_dim = 512
    attention_dim = 512

    clip_value = 5.0
    dropout_prob = 0.5
    init_lr = 4e-3
    init_wd = 1e-3
    patience = 1

    save_freq_per_epoch = 2

    data_dir = './data/'
    model_dir = './models/%s/' % MODEL_NAME
    params_dir = model_dir + 'params/'
    logs_dir = model_dir + 'logs/'

    def __init__(self):

        try_mkdir(self.model_dir)
        try_mkdir(self.params_dir)
        try_mkdir(self.logs_dir)

        self._init_data_loaders()

    def _init_data_loaders(self):

        self.train_set = GTAV(data_dir=self.data_dir, datatype='train')
        self.val_set = GTAV(data_dir=self.data_dir, datatype='val')
        self.y_keys_info = self.train_set.y_keys_info
        self.weight_info = self.train_set.weight_info

        self.trainloader = torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.batch_size, collate_fn=collate_fn,
                                                       shuffle=True, drop_last=False, num_workers=self.NUM_WORKERS)

        self.valloader = torch.utils.data.DataLoader(dataset=self.val_set, batch_size=self.batch_size, collate_fn=collate_fn,
                                                     shuffle=False, drop_last=False, num_workers=self.NUM_WORKERS)

        self.check_point = len(self.trainloader) // self.save_freq_per_epoch

