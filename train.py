# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:06:17 2019

@author: msq96
"""


import time

import torch
from torch.nn.utils.clip_grad import clip_grad_value_
from tensorboardX import SummaryWriter

from config import Config
from dataloader import GTAV
from utils import save_model_optimizer, load_model_optimizer
from models import Encoder, Decoder


def train(train_input_images, train_actions, encoder, decoder, criterion, optimizer, model_paras, args):

    train_input_images = train_input_images.cuda()
    train_actions = train_actions.cuda()

    encoder_outputs = encoder.forward(train_input_images, args.decoder_batch_size, args.seq_len)
    y = decoder.forward(encoder_outputs, train_actions, args.decoder_batch_size, args.seq_len)

    train_loss = criterion(y, train_actions)

    optimizer.zero_grad()
    train_loss.backward()

    clip_grad_value_(model_paras, args.clip_value)
    optimizer.step()

    return train_loss.item()

def validate(val_input_images, val_actions, encoder, decoder, criterion, args):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        encoder_outputs = encoder.forward(val_input_images, args.decoder_batch_size, args.seq_len)
        y = decoder.inference(encoder_outputs, args.init_action, args.decoder_batch_size, args.seq_len)

        val_loss = criterion(y, val_actions)

        return val_loss.item()


args = Config()

# TODO
print('initializing datasets...')
trainset = GTAV(args.train_fp)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.decoder_batch_size,
                                          shuffle=True, num_workers=args.NUM_WORKERS, drop_last=True)
valset = GTAV(args.val_fp)
valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=args.decoder_batch_size,
                                        shuffle=True, num_workers=args.NUM_WORKERS, drop_last=True)

print('initializing models...')
encoder = Encoder(model_name='xception', freeze=True, show_feature_dims=True)
decoder = Decoder(encoder_dim=encoder.encoder_dim, decoder_dim=args.decoder_dim, attention_dim=args.attention_dim,
                  num_loc=encoder.num_loc, num_actions=args.num_actions, num_layers=args.num_layers, dropout_prob=args.dropout_prob)
encoder.cuda()
decoder.cuda()

print('Initializing optimizer...')
model_paras = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(model_paras, lr=args.lr, weight_decay=args.wd)

# TODO
load_model_optimizer()

criterion = torch.nn.MSELoss()
writer = SummaryWriter(args.log_dir)
global_batch_counter = 0
train_loss_cp = 0.0
val_loss_cp = 0.0
print('Start training!')
for epoch in range(args.EPOCH):
    start_epoch_time = time.time()
    encoder.train()
    decoder.train()

    for train_batch_idx, (train_input_images, train_actions) in enumerate(trainloader, 1):

        train_loss_batch = train(train_input_images, train_actions, encoder, decoder, criterion, optimizer, model_paras, args)

        writer.add_scalar('batch/train_loss_batch', train_loss_batch, global_batch_counter)
        train_loss_cp += train_loss_batch
        global_batch_counter += 1

        if global_batch_counter % args.check_point == 0:
            print('Saving models...')
            # TODO
            save_model_optimizer()

            print('Start validating...')
            for val_batch_idx, (val_input_images, val_actions) in enumerate(valloader, 1):

                val_loss_batch = validate(val_input_images, val_actions, encoder, decoder, criterion, args)

                writer.add_scalar('batch/val_loss_batch', val_loss_batch, global_batch_counter)
                val_loss_cp += val_loss_batch

                if args.DEBUG:
                    break

            print('[%d] Epoch reach check point. [%.3f] training loss; [%.3f] validation loss; [%.2f] hours used.'
                  %(epoch, train_loss_cp/train_batch_idx, val_loss_cp/val_batch_idx, (time.time()-start_epoch_time)/3600))

        if args.DEBUG:
            break

    print('[%d] Epoch finished. [%.2f] hours used.' % (epoch, (time.time()-start_epoch_time)/3600))

    if args.DEBUG:
        break

writer.close()
