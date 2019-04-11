# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:06:17 2019

@author: msq96
"""


import time
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from model_config import Config
from utils import save_model_optimizer, load_model_optimizer, train, validate
from models import Encoder, Decoder

print('Initializing configuration...')
config = Config()

print('Initializing models...')
encoder = Encoder(encoder_name=config.ENCODER_NAME, show_feature_dims=True)
decoder = Decoder(encoder_dim=encoder.encoder_dim, decoder_dim=config.decoder_dim, attention_dim=config.attention_dim,
                  num_loc=encoder.num_loc, y_keys_info=config.y_keys_info, dropout_prob=config.dropout_prob)
encoder.cuda()
decoder.cuda()


print('Initializing datasets...')
trainloader = config.trainloader
valloader = config.valloader

print('Initializing optimizer...')
model_paras = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(model_paras, lr=config.init_lr, weight_decay=config.init_wd)


print('Initializing scheduler...')
scheduler = ReduceLROnPlateau(optimizer, patience=config.patience, verbose=True)


print('Loading parameters...')
encoder, decoder, optimizer, scheduler, current_epoch, global_batch_counter, global_batch_counter_val, global_timer = load_model_optimizer(encoder, decoder, optimizer, scheduler, config)

if config.manual_change:
    print('Changing learning rate and weight decay manually!')
    for param_group in optimizer.param_groups:
        param_group['lr'] = config.lr
        param_group['weight_decay'] = config.wd

criterion = {}
for action in config.y_keys_info:
    criterion[action] = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(config.weight_info[action]).cuda())

encoder.train()
decoder.train()

writer = SummaryWriter(config.logs_dir)
for epoch in range(current_epoch, config.EPOCH):
    print('[%d] epoch starts training...'%epoch)
    start_epoch_time = time.time()


    train_loss_cp = 0.0
    for train_batch_idx, (train_input_images, train_actions) in enumerate(tqdm(trainloader), 1):

        train_loss_batch, train_losses_batch_log = train(train_input_images, train_actions, encoder, decoder,
                                                 criterion, optimizer, model_paras, config, sampling_prob=0.5)

        writer.add_scalar('batch_train/train_loss_batch', train_loss_batch, global_batch_counter)
        [writer.add_scalar('batch_train/train_loss_batch_%s'%action, loss, global_batch_counter)\
         for action, loss in train_losses_batch_log.items()]

        train_loss_cp += train_loss_batch
        global_batch_counter += 1

        if global_batch_counter % config.check_point == 0 or config.DEBUG:

            print('Start validating...')
            num_correct = {action: 0 for action in config.y_keys_info.keys()}
            encoder.eval()
            decoder.eval()

            val_loss_cp = 0.0
            with torch.no_grad():
                for val_batch_idx, (val_input_images, val_actions) in enumerate(valloader, 1):

                    val_loss_batch, val_losses_batch_log, num_correct = validate(val_input_images, val_actions, encoder,
                                                                                 decoder, criterion, config, num_correct)

                    writer.add_scalar('batch_val/val_loss_batch', val_loss_batch, global_batch_counter_val)
                    [writer.add_scalar('batch_val/val_loss_batch_%s'%action, loss, global_batch_counter_val)\
                     for action, loss in val_losses_batch_log.items()]

                    val_loss_cp += val_loss_batch
                    global_batch_counter_val += 1

                    if config.DEBUG and val_batch_idx == 2:
                        break

            [writer.add_scalar('accuracy/val_accuracy_batch_%s'%action, num/len(config.val_set), global_batch_counter_val)\
             for action, num in num_correct.items()]

            print('[%d] Epoch reaches check point. [%.3f] training loss; [%.3f] validation loss; [%.2f] hours used.'
                  %(epoch, train_loss_cp/train_batch_idx, val_loss_cp/val_batch_idx, (time.time()-start_epoch_time)/3600))

            print('Saving models...')
            save_model_optimizer(encoder, decoder, optimizer, scheduler, epoch, global_batch_counter,
                                 global_batch_counter_val, global_timer, config)
            print('Saved!')

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
    save_model_optimizer(encoder, decoder, optimizer, scheduler, epoch, global_batch_counter,
                         global_batch_counter_val, global_timer, config)
    print('Saved!')

    for idx, param_group in enumerate(optimizer.param_groups, 1):
        writer.add_scalar('epoch/lr_%d'%idx, param_group['lr'], epoch)

    if config.DEBUG:
        break

writer.close()
