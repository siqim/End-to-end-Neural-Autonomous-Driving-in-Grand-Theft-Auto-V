# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:12:02 2018

@author: msq96
"""


import time
import numpy as np
import torch
import pretrainedmodels
import pretrainedmodels.utils as utils


print(pretrainedmodels.model_names)
test_models = ['pnasnet5large', 'nasnetalarge', 'senet154', 'polynet', 'inceptionv4', 'xception', 'resnet152']
attr = {model_name:{} for model_name in test_models}


for model_name in test_models:
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.cuda()
    model.eval()
    with torch.no_grad():
        load_img = utils.LoadImage()
        tf_img = utils.TransformImage(model)
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
