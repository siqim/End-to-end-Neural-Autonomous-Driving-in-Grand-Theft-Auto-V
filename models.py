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

    def __init__(self, encoder_name='xception', freeze=1, show_feature_dims=False):
        super().__init__()

        if freeze:
            self.model = pretrainedmodels.__dict__[encoder_name](num_classes=1000, pretrained='imagenet')
            for param in self.model.parameters():
                param.requires_grad_(requires_grad=False)
        else:
            self.model = pretrainedmodels.__dict__[encoder_name](num_classes=1000, pretrained=None)

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

    def forward(self, x, decoder_batch_size, seq_len):
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
        encoder_outputs = x.view(decoder_batch_size, seq_len, -1, x.size(1))
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

    def __init__(self, num_loc, encoder_dim, decoder_dim, attention_dim, dropout_prob):
        super().__init__()

        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)
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
            encoder_outputs: Output features from encoder with size: decoder_batch_size x seq_len x num_loc x encoder_dim

        Returns:
            encoder_atts: attention of encoder with size: decoder_batch_size x seq_len x num_loc x attention_dim
        """
        encoder_atts = self.encoder_attention(encoder_outputs)  # decoder_batch_size x seq_len x num_loc x attention_dim
        return encoder_atts


    def forward(self, encoder_output, encoder_att, decoder_state):
        """
        Attention forward.

        Params:
            encoder_output: Output of encoder for a batch image at same time step with size:
                            decoder_batch_size x num_loc x encoder_dim
            encoder_att: Attention of encoder for a batch image at same time step with size:
                            decoder_batch_size x num_loc x attention_dim
            decoder_state: Hidden state of decoder with size: num_layers x decoder_batch_size x decoder_dim

        Returns:
            alpha: Weight for each location of encoder_output with size: decoder_batch_size x num_loc x 1
            attention_output: attention weighted encoding with size: decoder_batch_size x encoder_dim
        """

        decoder_att = self.decoder_attention(decoder_state).unsqueeze(2)  # num_layers x decoder_batch_size x 1 x attention_dim
        full_att = self.full_attention(encoder_att + decoder_att.sum(dim=0))  # decoder_batch_size x num_loc x 1

        alpha = self.softmax(full_att)  # decoder_batch_size x num_loc x 1
        attention_output = (alpha * encoder_output).sum(dim=1)  # decoder_batch_size x encoder_dim
        return alpha, attention_output


class Decoder(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim,
                 num_loc, num_actions, num_layers, dropout_prob=0.5):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.num_loc = num_loc
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.att_model = Attention(num_loc, encoder_dim, decoder_dim, attention_dim, dropout_prob)

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

    def _init_state(self, decoder_batch_size):
        h_0 = torch.zeros([self.num_layers, decoder_batch_size, self.decoder_dim]).cuda()
        c_0 = torch.zeros([self.num_layers, decoder_batch_size, self.decoder_dim]).cuda()

        return (h_0, c_0)

    def forward(self, encoder_outputs, actions, decoder_batch_size, seq_len):
        """
        Decoder forward.

        Params:
            encoder_outputs: Output features from encoder with size: decoder_batch_size x seq_len x num_loc x encoder_dim
            actions: True actions at each time step with size: decoder_batch_size x seq_len x num_actions
                     Note that actions[:, 0, :] should be the init_action to start decoder forward process

        Returns:
            y: Predicted actions from decoder with size: decoder_batch_size x seq_len x num_actions
        """

        encoder_atts = self.att_model.generate_encoder_atts(encoder_outputs)  # decoder_batch_size x seq_len x num_loc x attention_dim
        y = torch.zeros((decoder_batch_size, seq_len, self.num_actions)).cuda()

        decoder_state = self._init_state(decoder_batch_size)

        for i in range(seq_len):

            # (decoder_batch_size x num_loc x 1, decoder_batch_size x encoder_dim)
            alpha, attention_output = self.att_model.forward(encoder_outputs[:, i], encoder_atts[:, i], decoder_state[0])

            # decoder_batch_size x 1 x (encoder_dim + num_actions)
            # 1 means sequence length for decoder; since we have a for loop, seq_len here = 1
            decoder_input = torch.cat((attention_output, actions[:, i]), dim=1).unsqueeze(1)

            # (decoder_batch_size x 1 x decoder_dim, (num_layers x decoder_batch_size x decoder_dim)*2)
            # 1 means sequence length for decoder; since we have a for loop, seq_len here = 1
            output, decoder_state = self.lstm(decoder_input, decoder_state)

            y[:, i] = self.sigmoid(self.fc_output(output.squeeze(1)))  # decoder_batch_size x num_actions

        return y

    def inference(self, encoder_outputs, init_action, decoder_batch_size, seq_len):
        """
        Decoder inference.

        Params:
            encoder_outputs: Output features from encoder with size: decoder_batch_size x seq_len x num_loc x encoder_dim
            init_action: y_0, to start decoder forward process with size: decoder_batch_size x num_actions

        Returns:
            y: Predicted actions from decoder with size: decoder_batch_size x seq_len x num_actions
        """

        encoder_atts = self.att_model.generate_encoder_atts(encoder_outputs)  # decoder_batch_size x seq_len x num_loc x attention_dim
        y = torch.zeros((decoder_batch_size, seq_len+1, self.num_actions)).cuda()  # +1 to make the for loop easier to implement
        y[:, 0] = init_action

        decoder_state = self._init_state(decoder_batch_size)

        for i in range(seq_len):

            # (decoder_batch_size x num_loc x 1, decoder_batch_size x encoder_dim)
            alpha, attention_output = self.att_model.forward(encoder_outputs[:, i], encoder_atts[:, i], decoder_state[0])

            # decoder_batch_size x 1 x (encoder_dim + num_actions)
            # 1 means sequence length for decoder; since we have a for loop, seq_len here = 1
            decoder_input = torch.cat((attention_output, y[:, i]), dim=1).unsqueeze(1)

            # (decoder_batch_size x 1 x decoder_dim, (num_layers x decoder_batch_size x decoder_dim)*2)
            # 1 means sequence length for decoder; since we have a for loop, seq_len here = 1
            output, decoder_state = self.lstm(decoder_input, decoder_state)

            y[:, i+1] = self.sigmoid(self.fc_output(output.squeeze(1)))  # decoder_batch_size x num_actions

        return y[:, 1:]


if __name__ == '__main__':

    import time
    import numpy as np

    seq_len, decoder_batch_size, num_actions, num_layers = 45, 2, 3, 2
    decoder_dim, attention_dim = 512, 512
    lr = 4e-4

    encoder = Encoder(encoder_name='xception', freeze=1, show_feature_dims=True)
    encoder.cuda()
    encoder.train()

    mean, std, input_size = encoder.mean, encoder.std, encoder.input_size
    num_loc, encoder_dim = encoder.num_loc, encoder.encoder_dim

    input_images = torch.rand((decoder_batch_size, seq_len, input_size[0], input_size[1], input_size[2])).cuda()
    actions = torch.rand((decoder_batch_size, seq_len, num_actions)).cuda()

    decoder = Decoder(encoder_dim=encoder_dim, decoder_dim=decoder_dim, attention_dim=attention_dim,
                      num_loc=num_loc, num_actions=num_actions, num_layers=num_layers)
    decoder.cuda()
    decoder.train()

    time_used = []
    trainloss = 0
    criterion = nn.MSELoss()
    model_paras = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(model_paras, lr=lr)
    for i in range(10):
        s = time.time()

        encoder.zero_grad()
        decoder.zero_grad()
        encoder_outputs = encoder.forward(input_images, decoder_batch_size, seq_len)
        y = decoder.forward(encoder_outputs, actions, decoder_batch_size, seq_len)
        loss = criterion(y, actions)
        loss.backward()

        optimizer.step()

        e = time.time()
        print(e-s)
        time_used.append(e-s)
    print('----------------')
    print(np.mean(time_used[1:]))


#    with torch.no_grad():
#        encoder.eval()
#        decoder.eval()
#        time_used = []
#        init_action = torch.FloatTensor([[0.5, 0.0, 0.0]]*decoder_batch_size)
#
#        for i in range(10):
#            s = time.time()
#
#            encoder_outputs = encoder.forward(input_images, decoder_batch_size, seq_len)
#            y = decoder.inference(encoder_outputs, init_action, decoder_batch_size, seq_len)
#
#            e = time.time()
#            print(e-s)
#            time_used.append(e-s)
#
#        print('----------------')
#        print(np.mean(time_used[1:]))
#
#
#    with torch.no_grad():
#        encoder.eval()
#        decoder.eval()
#        decoder_batch_size, seq_len = 1, 1
#        time_used = []
#        input_image = torch.rand((1, 1, input_size[0], input_size[1], input_size[2])).cuda()
#        init_action = torch.FloatTensor([[0.5, 0.0, 0.0]]*decoder_batch_size)
#
#        for i in range(100):
#            s = time.time()
#
#            encoder_outputs = encoder.forward(input_image, decoder_batch_size, seq_len)
#            y = decoder.inference(encoder_outputs, init_action, decoder_batch_size, seq_len)
#
#            e = time.time()
#            print(e-s)
#            time_used.append(e-s)
#
#        print('----------------')
#        print(np.mean(time_used[1:]))
