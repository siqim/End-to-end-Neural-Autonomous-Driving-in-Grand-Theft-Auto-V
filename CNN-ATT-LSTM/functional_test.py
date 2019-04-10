# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 17:21:43 2019

@author: msq96
"""


import time
import numpy as np

import torch
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils as pretrained_utils

from models import Encoder


print(pretrainedmodels.model_names)
test_models = ['pnasnet5large', 'nasnetalarge', 'senet154', 'polynet', 'inceptionv4', 'xception', 'resnet152']
attr = {model_name:{} for model_name in test_models}


for model_name in test_models:
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.cuda()
    model.eval()
    with torch.no_grad():
        load_img = pretrained_utils.LoadImage()
        tf_img = pretrained_utils.TransformImage(model)
        path_img = '../test/2.png'
        input_img = load_img(path_img)
        input_tensor = tf_img(input_img)
        input_tensor = input_tensor.unsqueeze(0).cuda()
        time_used_per_model = []
        for i in range(100):
            s = time.time()
            output_features  = model.features(input_tensor)
            e = time.time()
            time_used_per_model.append(e-s)
            print(e-s, model_name)
        attr[model_name]['time_used'] = time_used_per_model
        attr[model_name]['size'] = input_tensor.size()


for key, values in attr.items():
    time_used = values['time_used']
    size = values['size']
    print('%.4f'%np.mean(time_used), size, key)

# On GTX 1060
# 0.0208 torch.Size([1, 3, 299, 299]) xception
# 0.0605 torch.Size([1, 3, 224, 224]) resnet152
# 0.0621 torch.Size([1, 3, 299, 299]) inceptionv4
# 0.1192 torch.Size([1, 3, 331, 331]) pnasnet5large
# 0.1534 torch.Size([1, 3, 331, 331]) nasnetalarge
# 0.1729 torch.Size([1, 3, 224, 224]) senet154
# 0.2439 torch.Size([1, 3, 331, 331]) polynet

# Xception won my trust!


class RNN_speed_test(nn.Module):

    def __init__(self,  input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.lstmcell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, inputs, flag, times=1000):
        if flag == 'lstm':
            time_used = []
            for i in range(times):
                s = time.time()
                outputs, states = self.lstm(inputs)
                e = time.time()
                print(e-s)
                time_used.append(e-s)
            return [flag, '%.4f'%np.mean(time_used)]

        elif flag == 'lstmcell':
            time_used = []
            for i in range(times):
                s = time.time()
                for j in range(inputs.size(1)):
                    if j == 0:
                        state = self.lstmcell(inputs[:, j, :])
                    else:
                        state = self.lstmcell(inputs[:, j, :], state)
                e = time.time()
                print(e-s)
                time_used.append(e-s)
            return [flag, '%.4f'%np.mean(time_used)]

batch_size, seq_len, input_size, hidden_size = 1, 1, 2048*3, 2048*3

model = RNN_speed_test(input_size, hidden_size)
model.cuda()
inputs = torch.rand([batch_size, seq_len, input_size]).cuda()

lstm = model.forward(inputs, 'lstm')
lstmcell = model.forward(inputs, 'lstmcell')
print(lstm)
print(lstmcell)

# When seqence length >1, lstm is faster; lstmcell is faster otherwise.


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
