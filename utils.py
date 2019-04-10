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

def train(train_input_images, train_actions, encoder, decoder, criterion, optimizer, model_paras, config, sampling_prob):

    times = int(train_input_images.shape[1] / config.seq_len)

    train_loss = 0
    for i in range(times):
        train_input_image_seq = train_input_images[:, i*config.seq_len:(i+1)*config.seq_len].cuda()
        train_action_seq = {action: torch.cat((config.init_y[action], values[:, i*config.seq_len:(i+1)*config.seq_len]), dim=1).cuda()\
                            for action, values in train_actions.items()}

        encoder.zero_grad()
        decoder.zero_grad()
        encoder_outputs = encoder.forward(train_input_image_seq, config.decoder_batch_size, config.seq_len)
        y = decoder.forward(encoder_outputs, train_action_seq, config.decoder_batch_size, config.seq_len, sampling_prob)

        losses = []
        for action in config.y_keys_info.keys():
            losses.append(criterion[action](y[action], train_action_seq[action][:, 1:]))
        total_loss = sum(losses)
        total_loss.backward()

        train_loss += total_loss.item()

        clip_grad_value_(model_paras, config.clip_value)
        optimizer.step()

        accuracy = {}
        for action in config.y_keys_info.keys():
            _, y_pred = y[action].max(dim=1)
            accuracy[action] = (y_pred == train_action_seq[action][:, 1:]).sum().item() / (config.decoder_batch_size*config.seq_len)
        print(accuracy)

    return train_loss/times

def validate(val_input_images, val_actions, encoder, decoder, criterion, config):
    encoder.eval()
    decoder.eval()

    times = int(val_input_images.shape[1] / config.seq_len)

    val_loss = 0

    for i in range(times):
#    with torch.no_grad():
        val_input_image_seq = val_input_images[:, i*config.seq_len:(i+1)*config.seq_len].cuda()
        val_action_seq = {action: torch.cat((config.init_y[action], values[:, i*config.seq_len:(i+1)*config.seq_len]), dim=1).cuda()\
                            for action, values in val_actions.items()}

        encoder_outputs = encoder.forward(val_input_image_seq, config.decoder_batch_size, config.seq_len)
        y = decoder.validate(encoder_outputs, config.init_y, config.decoder_batch_size, config.seq_len)

        losses = []
        for action in config.y_keys_info.keys():
            losses.append(criterion[action](y[action], val_action_seq[action][:, 1:]))
        total_loss = sum(losses)

        val_loss += total_loss.item()

#        accuracy = {}
#        y_pred = {}
#        for action in config.y_keys_info.keys():
#            _, y_pred[action] = y[action].max(dim=1)
#            accuracy[action] = (y_pred[action] == val_action_seq[action][:, 1:]).sum().item() / (config.decoder_batch_size*config.seq_len)
#        print(accuracy)

    return val_loss/times

def get_models():

    from models import Encoder, Decoder
    from model_config import Config

    print('Initializing configuration...')
    config = Config()

    print('Initializing models...')
    encoder = Encoder(encoder_name=config.ENCODER_NAME, show_feature_dims=True)
    decoder = Decoder(encoder_dim=encoder.encoder_dim, decoder_dim=config.decoder_dim, attention_dim=config.attention_dim,
                      action_dim=config.action_dim, num_loc=encoder.num_loc, y_keys_info=config.y_keys_info, num_layers=config.num_layers,
                      dropout_prob=config.dropout_prob)
    encoder.cuda()
    decoder.cuda()

    params_list = os.listdir(config.params_dir)
    states = load_lastest_states(config.params_dir, params_list)

    encoder.load_state_dict(states['encoder'])
    decoder.load_state_dict(states['decoder'])

    return encoder, decoder, config.init_y
