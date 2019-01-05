# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:06:14 2019

@author: msq96
"""


import os
import torch
from preprocess import build_data
from utils import try_mkdir


class Config():

    MODEL_NAME = 'Xception-LSTM-Jan-4'
    FREEZE_ENCODER = 1
    ENCODER_NAME = 'xception'
    EPOCH = 10
    NUM_WORKERS = 4
    DEBUG = True

    decoder_batch_size = 8
    seq_len = 32

    decoder_dim = 512
    attention_dim = 512

    num_actions = 3
    num_layers = 2

    clip_value = 5.0
    dropout_prob = 0.5
    lr = 4e-4
    wd = 1e-4
    patience = 2

    check_point = 1000

    model_dir = '../models/%s/' % MODEL_NAME
    params_dir = model_dir + 'params/'
    logs_dir = model_dir + 'logs/'
    data_dir = '../data/processed_data/seq_len_%d/' % seq_len
    raw_data_dir = '../data/raw_data/'
    train_ratio = 0.8


    train_fp = data_dir + 'train/'
    val_fp = data_dir + 'val/'

    init_action = torch.FloatTensor([[0.5, 0.0, 0.0]]*decoder_batch_size)

    def __init__(self):

        try_mkdir(self.model_dir)
        try_mkdir(self.params_dir)
        try_mkdir(self.logs_dir)

        if self.DEBUG:
            self.check_point = 1

        if not os.listdir(self.train_fp):
            print('Building training data...')
            build_data(self.seq_len, self.raw_data_dir, self.train_ratio, self.data_dir)