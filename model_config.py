# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:52:45 2019

@author: msq96
"""



import torch
from utils import try_mkdir
from dataloader import GTAV


class Config(object):

    MODEL_NAME = 'se_resnext50_32x4d-LSTM-Apr-6'
    ENCODER_NAME = 'se_resnext50_32x4d'
    EPOCH = 10
    NUM_WORKERS = 0
    DEBUG = False

    num_layers = 1
    decoder_batch_size = 1
    seq_len = 16

    decoder_dim = 512
    attention_dim = 512

    clip_value = 5.0
    dropout_prob = 0.5
    lr = 4e-4
    wd = 1e-4
    patience = 2

    check_point = 300

    data_dir = './data/'
    model_dir = './models/%s/' % MODEL_NAME
    params_dir = model_dir + 'params/'
    logs_dir = model_dir + 'logs/'

    def __init__(self):

        try_mkdir(self.model_dir)
        try_mkdir(self.params_dir)
        try_mkdir(self.logs_dir)

        self._init_data_loaders()

        if self.DEBUG:
            self.check_point = 1

    def _init_data_loaders(self):

        train_set = GTAV(data_dir=self.data_dir, datatype='train', bin_fname='y_bin.pickle')
        val_set = GTAV(data_dir=self.data_dir, datatype='val', bin_fname='y_bin.pickle')
        self.init_y = torch.FloatTensor([train_set.init_y]*self.decoder_batch_size)
        self.y_keys_info = train_set.y_keys_info

        self.trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.decoder_batch_size,
                                                       shuffle=True, drop_last=True, num_workers=self.NUM_WORKERS)

        self.valloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=self.decoder_batch_size,
                                                     shuffle=True, drop_last=True, num_workers=self.NUM_WORKERS)
