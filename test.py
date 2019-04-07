# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:40:33 2019

@author: msq96
"""


import time
import numpy as np

import torch
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils as pretrained_utils

import cv2
import gzip
import pickle
from models import Encoder
from game_play.deepgtav.messages import frame2numpy



print(pretrainedmodels.model_names)
test_models = ['pnasnet5large', 'nasnetalarge', 'senet154', 'polynet', 'inceptionv4', 'xception', 'resnet152']
dataname = './game_play/dataset.pz'
dataset = gzip.open(dataname)


batch_images = []
for i in range(30):
    data_dct = pickle.load(dataset)
    frame = data_dct['frame']
    image = frame2numpy(frame, (350,205+20))[20:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, [2,0,1])
    batch_images.append(torch.FloatTensor(image/255))

batch_images = torch.stack(batch_images)


model = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=500, pretrained=None)
model.cuda()

model.train()


input_tensor = batch_images.cuda()
s = time.time()
output_features  = model.features(input_tensor)
e = time.time()
print(e-s)

