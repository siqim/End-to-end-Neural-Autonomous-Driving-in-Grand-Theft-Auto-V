# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:44:41 2019

@author: msq96
"""


import os
import numpy as np

import torch
from torch.nn.utils.clip_grad import clip_grad_value_


def try_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        return True
    else:
        return False

def save_model_optimizer(encoder, decoder, optimizer, scheduler, epoch, global_batch_counter, global_timer, config):
   state = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),

            'epoch': epoch,
            'global_batch_counter': global_batch_counter,
            'global_timer': global_timer
            }

   torch.save(state, open(config.params_dir + config.MODEL_NAME + '_' + str(global_batch_counter) + '.tar', 'wb'))

def load_lastest_states(params_dir, params_list):
    lastest_states_idx = np.argmax([int(each_params.split('_')[1][:-4]) for each_params in params_list])
    lastest_states_path = params_dir + params_list[lastest_states_idx]
    lastest_states = torch.load(open(lastest_states_path, 'rb'))
    return lastest_states

def load_model_optimizer(encoder, decoder, optimizer, scheduler, config):
    params_list = os.listdir(config.params_dir)
    if params_list:
        print('Loading lastest checkpoint...')
        states = load_lastest_states(config.params_dir, params_list)

        encoder.load_state_dict(states['encoder'])
        decoder.load_state_dict(states['decoder'])
        optimizer.load_state_dict(states['optimizer'])
        scheduler.load_state_dict(states['scheduler'])

        current_epoch = states['epoch'] + 1
        global_batch_counter = states['global_batch_counter']
        global_timer = states['global_timer']

        return encoder, decoder, optimizer, scheduler, current_epoch, global_batch_counter, global_timer

    else:
        return encoder, decoder, optimizer, scheduler, 1, 0, 0

def train(train_input_images, train_actions, encoder, decoder, criterion, optimizer, model_paras, config):

    train_input_images = train_input_images.cuda()
    train_actions = train_actions.cuda()

    encoder_outputs = encoder.forward(train_input_images, config.decoder_batch_size, config.seq_len)
    y = decoder.forward(encoder_outputs, train_actions, config.decoder_batch_size, config.seq_len)

    train_loss = criterion(y, train_actions)

    optimizer.zero_grad()
    train_loss.backward()

    clip_grad_value_(model_paras, config.clip_value)
    optimizer.step()

    return y.data.cpu(), train_loss.item()

def validate(val_input_images, val_actions, encoder, decoder, criterion, config):

    val_input_images = val_input_images.cuda()
    val_actions = val_actions.cuda()

    encoder_outputs = encoder.forward(val_input_images, config.decoder_batch_size, config.seq_len)
    y = decoder.inference(encoder_outputs, config.init_action, config.decoder_batch_size, config.seq_len)

    val_loss = criterion(y, val_actions)

    return y.data.cpu(), val_loss.item()
