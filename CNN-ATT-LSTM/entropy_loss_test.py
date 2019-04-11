# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:51:48 2019

@author: msq96
"""

import torch

loss = torch.nn.CrossEntropyLoss()

num_correct = 10
y_pred = torch.zeros((16,20))

for i in range(16):
    for j in range(16):
        if i == j and i<=num_correct:
            y_pred[i][i] = 1
        else:
            y_pred[i][j] = 0.7

y = torch.arange(0,16)

print(loss(y_pred, y))
print((y_pred.max(1)[1] == y).sum())