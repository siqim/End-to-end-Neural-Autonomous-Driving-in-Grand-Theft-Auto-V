# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 13:31:11 2018

@author: msq96
"""

import torch
import torch.optim as optim
import torch.nn as nn
import pretrainedmodels


class Encoder(nn.Module):

    def __init__(self, seq_len, decoder_batch_size, model_name='xception', show_feature_dims=False):
        super().__init__()
        self.seq_len = seq_len
        self.decoder_batch_size = decoder_batch_size

        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

        self.mean = self.model.mean
        self.std = self.model.std
        self.input_size = self.model.input_size

        if show_feature_dims:
            fake_image = torch.rand(self.input_size).unsqueeze(0)
            x = self.model.features(fake_image)
            x = x.view(x.size(0), -1, x.size(1))

            self.num_loc = x.size(1)
            self.encoder_dim = x.size(2)
        else:
            self.num_loc = None
            self.encoder_dim = None

    def forward(self, x):
        """
        Encoder forward.

        Params:
            x: Input images with size: decoder_batch_size x seq_len x num_channels x width x height
                                      (decoder_batch_size x seq_len x 3 x 299 x 299 if use xception)

        Returns:
            encoder_outputs: Output features with size: decoder_batch_size x seq_len x num_loc x encoder_dim
                                                       (decoder_batch_size x seq_len x (10x10) x 2048 if use xception)
        """

        # (decoder_batch_size x seq_len) x encoder_dim x feature_width x feature_height
        x = self.model.features(x.view(-1, x.size(2), x.size(3), x.size(4)))
        encoder_outputs = x.view(self.decoder_batch_size, self.seq_len, -1, x.size(1))
        return encoder_outputs

    def _forward_old(self, x):
        """
        Encoder forward. For test use.

        Params:
            x: Input images with size: seq_len x 3 x 299 x 299
        Returns:
            x: Output features with size: seq_len x num_loc x encoder_dim
                                         (seq_len x (10x10) x 2048 if use xception)
        """
        x = self.model.features(x)
        x = x.view(x.size(0), -1, x.size(1))
        return x



class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim, dropout_prob):
        super().__init__()

        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)
        self.full_attention = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(attention_dim),
                nn.Dropout(p=dropout_prob),

                nn.Linear(attention_dim, attention_dim),
                nn.ReLU(),
                nn.BatchNorm1d(attention_dim),
                nn.Dropout(p=dropout_prob),

                nn.Linear(attention_dim, 1)
                )

        self.softmax = nn.Softmax(dim=1)

    def generate_encoder_atts(self, encoder_outputs):
        """
        Generate attentions of encoder for a batch of images.

        Params:
            encoder_outputs: Output features from encoder with size: decoder_batch_size x seq_len x num_loc x encoder_dim

        Returns:
            encoder_atts: attention of encoder with size: encoder_batch_size x num_loc x attention_dim
        """
        encoder_atts = self.encoder_attention(encoder_outputs)  # encoder_batch_size x num_loc x attention_dim
        return encoder_atts


    def forward(self, encoder_output, encoder_att, decoder_state):
        """
        Attention forward.

        Params:
            encoder_output: Output of encoder for one image with size: num_loc x encoder_dim
            encoder_att: Attention of encoder for one image with size: num_loc x attention_dim
            decoder_state: Hidden state of decoder with size: decoder_batch_size x decoder_dim
                                                             (decoder_batch_size = 1 for this implementation)

        Returns:
            alpha: Weight for each location of encoder_output with size: 1 x num_loc
            attention_output: attention weighted encoding with size: 1 x encoder_dim
        """

        decoder_att = self.decoder_attention(decoder_state)  # decoder_batch_size x attention_dim
        full_att = self.full_attention(encoder_att + decoder_att).t()  # 1 x num_loc

        alpha = self.softmax(full_att)  # 1 x num_loc
        attention_output = alpha @ encoder_output  # 1 x encoder_dim
        return alpha, attention_output


class Decoder(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim, num_actions, dropout_prob=0.5, num_layers=1, decoder_batch_size=1):
        super().__init__()

        self.att_model = Attention(encoder_dim, decoder_dim, attention_dim, dropout_prob)

        self.decoder_dim = decoder_dim
        self.decoder_batch_size = decoder_batch_size
        self.num_actions = num_actions
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
                input_size = encoder_dim + num_actions,
                hidden_size = decoder_dim,
                num_layers = num_layers,
                batch_first = True
                )

        self.fc_output = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),

                nn.Linear(decoder_dim, num_actions)
                )

        self.sigmoid = nn.Sigmoid()

    def _init_state(self):
        h_0 = torch.zeros([self.num_layers, self.decoder_batch_size, self.decoder_dim]).cuda()
        c_0 = torch.zeros([self.num_layers, self.decoder_batch_size, self.decoder_dim]).cuda()

        return (h_0, c_0)

    def forward(self, encoder_outputs, actions):
        """
        Decoder forward.

        Params:
            encoder_outputs: Output features from encoder with size: encoder_batch_size x num_loc x encoder_dim
            actions: True actions at each time step with size: encoder_batch_size x num_actions

        Returns:
            y: Predicted actions from decoder with size: encoder_batch_size x num_actions
        """

        encoder_atts = self.att_model.generate_encoder_atts(encoder_outputs)  # encoder_batch_size x num_loc x attention_dim
        encoder_batch_size = encoder_atts.size(0)
        y = torch.zeros((encoder_batch_size, self.num_actions)).cuda()

        decoder_state = self._init_state()

        for i in range(encoder_batch_size):
            h = decoder_state[0]  # num_layers x decoder_batch_size x decoder_dim

            # (1 x num_loc, 1 x encoder_dim)
            alpha, attention_output = self.att_model.forward(encoder_outputs[i], encoder_atts[i], h[0])

            decoder_input = torch.cat((attention_output, actions[[i]]), dim=1).unsqueeze(0)  # 1 x 1 x (encoder_dim + num_actions)

            # (1 x 1 x decoder_dim, 1 x 1 x decoder_dim), (num_layers x decoder_batch_size x decoder_dim)*2
            output, decoder_state = self.lstm(decoder_input, decoder_state)

            y[i] = self.sigmoid(self.fc_output(output[0][0])).squeeze(0)  # num_actions

        return y


if __name__ == '__main__':

    import time
    import numpy as np

    encoder_batch_size, num_actions = 8, 3
    decoder_dim, attention_dim = 1024, 1024

    encoder = Encoder(model_name='xception', show_feature_dims=False)
    encoder.cuda()
    encoder.train()

    mean, std, input_size = encoder.mean, encoder.std, encoder.input_size
    num_loc, encoder_dim = encoder.num_loc, encoder.encoder_dim

    input_images = torch.rand((encoder_batch_size, input_size[0], input_size[1], input_size[2])).cuda()
    actions = torch.rand((encoder_batch_size, num_actions)).cuda()

    decoder = Decoder(encoder_dim=2048, decoder_dim=decoder_dim, attention_dim=attention_dim, num_actions=num_actions, flag='lstm', num_layers=1)
    decoder.cuda()
    decoder.train()

    time_used = []
    trainloss = 0
    criterion = nn.MSELoss()
    model_paras = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(model_paras, lr=4e-4)
    for i in range(100):
        s = time.time()

        encoder.zero_grad()
        decoder.zero_grad()
        encoder_outputs = encoder.forward(input_images)
        y = decoder.forward(encoder_outputs, actions)

        loss = criterion(y, actions)
        loss.backward()
        optimizer.step()

        e = time.time()
        print(e-s)
        time_used.append(e-s)

    print(np.mean(time_used))
