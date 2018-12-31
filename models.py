# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 13:31:11 2018

@author: msq96
"""


import torch
import torch.nn as nn
import pretrainedmodels


class Encoder(nn.Module):

    def __init__(self, model_name='xception'):
        super().__init__()
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    def forward(self, x):
        """
        Encoder forward.

        Params:
            x: Input images with size: encoder_batch_size x 3 x 299 x 299

        Returns:
            x: Output features with size: encoder_batch_size x num_loc x encoder_dim
                                         (encoder_batch_size x (10x10) x 2048 if use xception)
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

    def forward(self, encoder_outputs, decoder_state):
        """
        Attention forward.

        Params:
            encoder_outputs: Output features from encoder with size: encoder_batch_size x num_loc x encoder_dim
            decoder_state: Hidden state of decoder with size: decoder_batch_size x decoder_dim
                                                             (decoder_batch_size = 1 for this implementation)

        Returns:
            alpha: Weight for each location of encoder_output with size: encoder_batch_size x num_loc
            attention_outputs: attention weighted encoding with size: encoder_batch_size x encoder_dim
        """

        encoder_att = self.encoder_attention(encoder_outputs)  # encoder_batch_size x num_loc x attention_dim
        decoder_att = self.decoder_attention(decoder_state).unsqueeze(1)  # encoder_batch_size x 1 x attention_dim
        full_att = self.full_attention(encoder_att + decoder_att).squeeze(2)  # encoder_batch_size x num_loc

        alpha = self.softmax(full_att)  # encoder_batch_size x num_loc
        attention_outputs = (encoder_outputs * alpha.unsqueeze(2)).sum(1)  # encoder_batch_size x encoder_dim
        return alpha, attention_outputs


class Decoder(nn.Module):

    def __init__(self, encoder_dim, num_actions, hidden_size, num_layers, dropout_prob):
        super().__init__()

        self.lstm = nn.LSTM(
                input_size = encoder_dim + num_actions,
                hidden_size = hidden_size,
                num_layers = num_layers,
                batch_first = True
                )

        self.fc_output = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(p=dropout_prob),

                nn.Linear(hidden_size, num_actions)
                )


    def _init_state(self):
        h_0 = None
        c_0 = None
        y_0 = None

        return (h_0, c_0), y_0

    def forward(self, attention_outputs, actions):
        """
        Decoder forward.

        Params:
            attention_outputs: Output of attention model with size: encoder_batch_size x encoder_dim
            actions: True actions at each time step with size: encoder_batch_size x num_actions

        Returns:
            y: Predicted actions from decoder with size: encoder_batch_size x num_actions
        """

        # 1 x encoder_batch_size x (encoder_dim + num_actions)
        inputs = torch.cat((attention_outputs.unsequeeze(0), actions.unsequeeze(0)), dim=2)
        state = self._init_state()  # num_layers x decoder_batch_size x hidden_size

        outputs, states = self.lstm(inputs, state)  # encoder_batch_size x decoder_batch_size x hidden_size
        y = self.fc_output(outputs.squeeze(1))  # encoder_batch_size x num_actions
        return y
