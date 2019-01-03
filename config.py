# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:06:14 2019

@author: msq96
"""


import torch


class Config():

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

    train_fp = ''
    val_fp = ''
    log_dir = ''
    model_dir = ''
    check_point = 1000

    init_action = torch.FloatTensor([[0.5, 0.0, 0.0]]*decoder_batch_size)

    if DEBUG:
        check_point = 1
