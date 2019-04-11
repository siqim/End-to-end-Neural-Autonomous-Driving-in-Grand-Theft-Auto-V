# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:44:41 2019

@author: msq96
"""


import os
import time
import numpy as np


import torch
from torch.nn.utils.clip_grad import clip_grad_value_



def try_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        return True
    else:
        return False

def save_model_optimizer(encoder, decoder, optimizer, scheduler, epoch, global_batch_counter,
                         global_batch_counter_val, global_timer, config):
   state = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),

            'epoch': epoch,
            'global_batch_counter': global_batch_counter,
            'global_batch_counter_val': global_batch_counter_val,
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
        global_batch_counter_val = states['global_batch_counter_val']
        global_timer = states['global_timer']

        return encoder, decoder, optimizer, scheduler, current_epoch, global_batch_counter, global_batch_counter_val, global_timer

    else:
        return encoder, decoder, optimizer, scheduler, 1, 0, 0, 0

def train(train_input_images, train_actions, encoder, decoder, criterion, optimizer, model_paras, config, sampling_prob):

    train_input_images = train_input_images.cuda()
    train_actions = {action: values.cuda() for action, values in train_actions.items()}

    encoder.zero_grad()
    decoder.zero_grad()
    encoder_outputs = encoder.forward(train_input_images)
    y = decoder.forward(encoder_outputs)

    losses = []
    losses_log = {}
    for action in config.y_keys_info.keys():
        loss = criterion[action](y[action], train_actions[action])
        losses.append(loss)
        losses_log[action] = loss.item()


    total_loss = sum(losses)
    total_loss.backward()

    clip_grad_value_(model_paras, config.clip_value)
    optimizer.step()

    return total_loss.item(), losses_log

def validate(val_input_images, val_actions, encoder, decoder, criterion, config, num_correct):

    val_input_images = val_input_images.cuda()
    val_actions = {action: values.cuda() for action, values in val_actions.items()}

    encoder_outputs = encoder.forward(val_input_images)
    y = decoder.forward(encoder_outputs)

    losses = []
    losses_log = {}
    for action in config.y_keys_info.keys():
        loss = criterion[action](y[action], val_actions[action])
        losses.append(loss)
        losses_log[action] = loss.item()

    total_loss = sum(losses)

    for action in config.y_keys_info.keys():
        _, y_pred = y[action].max(dim=1)

        num_correct[action] += (y_pred == val_actions[action]).sum().item()

    return total_loss.item(), losses_log, num_correct

def get_models():

    from models import Encoder, Decoder
    from model_config import Config

    print('Initializing configuration...')
    config = Config()

    print('Initializing models...')
    encoder = Encoder(encoder_name=config.ENCODER_NAME, show_feature_dims=True)
    decoder = Decoder(encoder_dim=encoder.encoder_dim, decoder_dim=config.decoder_dim, attention_dim=config.attention_dim,
                      num_loc=encoder.num_loc, y_keys_info=config.y_keys_info, dropout_prob=config.dropout_prob)
    encoder.cuda()
    decoder.cuda()

    params_list = os.listdir(config.params_dir)
    states = load_lastest_states(config.params_dir, params_list)

    encoder.load_state_dict(states['encoder'])
    decoder.load_state_dict(states['decoder'])

    print('Loading Epoch', states['epoch'])

    return encoder, decoder
