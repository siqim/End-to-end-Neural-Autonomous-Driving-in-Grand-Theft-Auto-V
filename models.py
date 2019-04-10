# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 13:31:11 2018

@author: msq96
"""


import gzip
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pretrainedmodels


class Encoder(nn.Module):

    def __init__(self, encoder_name, show_feature_dims=True, data_dir='./data/'):
        super().__init__()

        self.model = pretrainedmodels.__dict__[encoder_name](num_classes=1000, pretrained=None)

        if show_feature_dims:
            with gzip.open(data_dir+'train/data_0.gz', 'rb') as f:
                frame = pickle.load(f)['frame']
                input_shape = [1, 1] + list(frame.shape[1:])
            fake_image = torch.rand(input_shape)
            with torch.no_grad():
                x = self.forward(fake_image, 1, 1)

            self.num_loc = x.size(2)
            self.encoder_dim = x.size(3)
        else:
            self.num_loc = None
            self.encoder_dim = None

    def forward(self, x):
        """
        Encoder forward.

        Params:
            x: Input images with size: batch_size x num_channels x height x width
                                      (batch_size x 3 x 205 x 350 if use xception)

        Returns:
            encoder_outputs: Output features with size: batch_size x num_loc x encoder_dim
                                                       (batch_size x (7x11) x 2048 if use Xception)
        """

        # batch_size x encoder_dim x feature_width x feature_height
        x = self.model.features(x)
        # batch_size x num_loc x encoder_dim
        encoder_outputs = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
        return encoder_outputs


class Attention(nn.Module):

    def __init__(self, num_loc, encoder_dim, decoder_dim, attention_dim, dropout_prob):
        super().__init__()

        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.full_attention = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(num_loc),
                nn.Dropout(p=dropout_prob),

                nn.Linear(attention_dim, attention_dim),
                nn.ReLU(),
                nn.BatchNorm1d(num_loc),
                nn.Dropout(p=dropout_prob),

                nn.Linear(attention_dim, 1)
                )

        self.softmax = nn.Softmax(dim=1)

    def generate_encoder_atts(self, encoder_outputs):
        """
        Generate attentions of encoder for a batch of images.
        Params:
            encoder_outputs: Output features from encoder with size: batch_size x num_loc x encoder_dim
        Returns:
            encoder_atts: attention of encoder with size: batch_size x num_loc x attention_dim
        """
        encoder_atts = self.encoder_attention(encoder_outputs)  # batch_size x num_loc x attention_dim
        return encoder_atts


    def forward(self, encoder_output, encoder_att):
        """
        Attention forward.
        Params:
            encoder_output: Output of encoder for a batch image at same time step with size:
                            batch_size x num_loc x encoder_dim
            encoder_att: Attention of encoder for a batch image at same time step with size:
                         batch_size x num_loc x attention_dim
        Returns:
            alpha: Weight for each location of encoder_output with size: batch_size x num_loc x 1
            attention_output: attention weighted encoding with size: batch_size x encoder_dim
        """

        full_att = self.full_attention(encoder_att)  # batch_size x num_loc x 1

        alpha = self.softmax(full_att)  # batch_size x num_loc x 1
        attention_output = (alpha * encoder_output).sum(dim=1)  # batch_size x encoder_dim
        return alpha, attention_output


class Decoder(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim,
                 num_loc, y_keys_info, num_layers, dropout_prob=0.5):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_loc = num_loc
        self.y_keys_info = y_keys_info
        self.dropout_prob = dropout_prob

        self.att_model = Attention(num_loc, encoder_dim, decoder_dim, attention_dim, dropout_prob)

        self.fc_output = {}
        for action, num_class in y_keys_info.items():
            self.fc_output[action] = nn.Sequential(
                                            nn.Linear(encoder_dim, decoder_dim),
                                            nn.ReLU(),
                                            nn.Dropout(p=dropout_prob),

                                            nn.Linear(decoder_dim, num_class)
                                            ).cuda()
            self.add_module(name='fc_output_%s'%action, module=self.fc_output[action])

    def forward(self, encoder_outputs):

        encoder_atts = self.att_model.generate_encoder_atts(encoder_outputs) # decoder_batch_size x num_loc x attention_dim
        alpha, attention_output = self.att_model.forward(encoder_outputs, encoder_atts) # batch_size x encoder_dim

        y = {}
        for action, num_class in self.y_keys_info.items():
            # batch_size x num_class
            logits = self.fc_output[action](attention_output)
            y[action] = logits

        return y




if __name__ == '__main__':

    import time
    import numpy as np
    from dataloader import GTAV

    seq_len, decoder_batch_size, num_layers = 8, 2, 1
    decoder_dim, attention_dim, action_dim = 512, 512, 50
    lr = 4e-4

    train_set = GTAV(data_dir='./data/', datatype='train', batch_size=decoder_batch_size)
    trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=decoder_batch_size,
                                              shuffle=True, drop_last=True, num_workers=0)

    init_y = train_set.init_y
    y_keys_info = train_set.y_keys_info


    encoder = Encoder(encoder_name='xception', show_feature_dims=True)
    num_loc, encoder_dim = encoder.num_loc, encoder.encoder_dim
    encoder.cuda()
    encoder.train()

    decoder = Decoder(encoder_dim=encoder_dim, decoder_dim=decoder_dim, attention_dim=attention_dim, action_dim=action_dim,
                      num_loc=num_loc, y_keys_info=y_keys_info, num_layers=num_layers)
    decoder.cuda()
    decoder.train()


    time_used = []
    trainloss = 0
    criterion = nn.CrossEntropyLoss()
    model_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(model_params, lr=lr)
    for i in range(100):
        s = time.time()

        input_images, actions = next(iter(trainloader))
        input_images = input_images[:decoder_batch_size, :seq_len].cuda()
        actions = {action: torch.cat((init_y[action], values[:decoder_batch_size, :seq_len]), dim=1).cuda() for action, values in actions.items()}

        encoder.zero_grad()
        decoder.zero_grad()
        encoder_outputs = encoder.forward(input_images, decoder_batch_size, seq_len)
        y = decoder.forward(encoder_outputs, actions, decoder_batch_size, seq_len)

        losses = []
        for action in y_keys_info.keys():
            losses.append(criterion(y[action], actions[action][:decoder_batch_size, 1:]))
        total_loss = sum(losses)
        total_loss.backward()
        print(total_loss.item())

        accuracy = {}
        for action in y_keys_info.keys():
            _, y_pred = y[action].max(dim=1)
            accuracy[action] = (y_pred == actions[action][:, 1:]).sum().item() / (decoder_batch_size*seq_len)
        print(accuracy)

        optimizer.step()

        e = time.time()
#        print(e-s)
        time_used.append(e-s)
    print('----------------')
    print(np.mean(time_used[1:]))



    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        time_used = []
        input_action = {k: v[0,:].cuda() for k,v in init_y.items()}
        decoder_state = decoder._init_state(1)
        input_images, actions = next(iter(trainloader))

        input_images = input_images[0, :].cuda()
        actions = {action: values[0, :].cuda() for action, values in actions.items()}

    with torch.no_grad():
        actions_pred = {action: [] for action in y_keys_info.keys()}
        for i in range(input_images.size(0)):
            s = time.time()

            encoder_output = encoder.inference(input_images[[i]])
            input_action, decoder_state = decoder.inference(encoder_output, input_action, decoder_state)
            for action in y_keys_info.keys():
                actions_pred[action].append(input_action[action])

            e = time.time()
#            print(e-s)
            time_used.append(e-s)

        print('----------------')
        print(np.mean(time_used[1:]))
