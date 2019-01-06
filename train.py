# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:06:17 2019

@author: msq96
"""


import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from model_config import Config
from utils import save_model_optimizer, load_model_optimizer, train, validate
from models import Encoder, Decoder


config = Config()

print('Initializing models...')
encoder = Encoder(encoder_name=config.ENCODER_NAME, freeze=config.FREEZE_ENCODER, show_feature_dims=True)
decoder = Decoder(encoder_dim=encoder.encoder_dim, decoder_dim=config.decoder_dim, attention_dim=config.attention_dim,
                  num_loc=encoder.num_loc, num_actions=config.num_actions, num_layers=config.num_layers, dropout_prob=config.dropout_prob)
encoder.cuda()
decoder.cuda()


print('Initializing datasets...')
config.init_data_loaders(input_size=encoder.input_size, mean=encoder.mean, std=encoder.std)
trainloader = config.trainloader
valloader = config.valloader


print('Initializing optimizer...')
model_paras = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(model_paras, lr=config.lr, weight_decay=config.wd)


print('Initializing scheduler...')
scheduler = ReduceLROnPlateau(optimizer, patience=config.patience, verbose=True)


print('Loading parameters...')
encoder, decoder, optimizer, scheduler, current_epoch, global_batch_counter, global_timer = load_model_optimizer(encoder, decoder, optimizer, scheduler, config)


criterion = torch.nn.MSELoss()
writer = SummaryWriter(config.logs_dir)
for epoch in range(current_epoch, config.EPOCH):
    print('[%d] epoch starts training...'%epoch)
    start_epoch_time = time.time()


    train_loss_cp = 0.0
    for train_batch_idx, (train_input_images, train_actions) in enumerate(trainloader, 1):

        train_loss_batch = train(train_input_images, train_actions, encoder, decoder, criterion, optimizer, model_paras, config)

        writer.add_scalar('batch/train_loss_batch', train_loss_batch, global_batch_counter)
        train_loss_cp += train_loss_batch
        global_batch_counter += 1

        if global_batch_counter % config.check_point == 0:

            print('Start validating...')
            encoder.eval()
            decoder.eval()

            val_loss_cp = 0.0
            with torch.no_grad():
                for val_batch_idx, (val_input_images, val_actions) in enumerate(valloader, 1):

                    val_loss_batch = validate(val_input_images, val_actions, encoder, decoder, criterion, config)

                    writer.add_scalar('batch/val_loss_batch', val_loss_batch, global_batch_counter)
                    val_loss_cp += val_loss_batch

                    if config.DEBUG and val_batch_idx == 2:
                        break

            print('[%d] Epoch reaches check point. [%.3f] training loss; [%.3f] validation loss; [%.2f] hours used.'
                  %(epoch, train_loss_cp/train_batch_idx, val_loss_cp/val_batch_idx, (time.time()-start_epoch_time)/3600))

            print('Saving models...')
            save_model_optimizer(encoder, decoder, optimizer, scheduler, epoch, global_batch_counter, global_timer, config)

            encoder.train()
            decoder.train()

        if config.DEBUG and train_batch_idx == 2:
            break

    time_used_epoch = (time.time() - start_epoch_time) / 3600
    global_timer += time_used_epoch

    print('[%d] Epoch finished; [%.2f] hours used for this epoch; [%.2f] hours used in total; [%d] batches have been trained.'
          % (epoch, time_used_epoch, global_timer, global_batch_counter))

    scheduler.step(val_loss_cp)

    print('Saving models...')
    save_model_optimizer(encoder, decoder, optimizer, scheduler, epoch, global_batch_counter, global_timer, config)

    writer.add_scalar('epoch/freeze_encoder', config.FREEZE_ENCODER, epoch)
    for idx, param_group in enumerate(optimizer.param_groups, 1):
        writer.add_scalar('epoch/lr_%d'%idx, param_group['lr'], epoch)

    if config.DEBUG:
        break

writer.close()
