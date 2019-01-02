# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 17:21:43 2019

@author: msq96
"""

import torch
from models import Encoder

def encoder_test(seq_len=4, decoder_batch_size=2, model_name='xception'):
    encoder = Encoder(seq_len=seq_len, decoder_batch_size=decoder_batch_size, model_name=model_name)
    encoder.cuda()
    encoder.eval()

    with torch.no_grad():
        images = []
        for i in range(decoder_batch_size):
            images.append(torch.rand((seq_len, 3, 299, 299)))
        images = torch.stack(images).cuda()
        features = encoder.forward(images)

        split_features = []
        for i in range(decoder_batch_size):
            split_features.append(encoder._forward_old(images[i]))
        split_features = torch.stack(split_features)

    assert(torch.all(split_features == features) == 1)
    print('encoder test passed!')

encoder_test()