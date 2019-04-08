# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 13:31:11 2018

@author: msq96
"""


import gzip
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import pretrainedmodels


# TODO: Support changing seq_len
# TODO: clean docs and check flow again
class Encoder(nn.Module):

    def __init__(self, encoder_name, show_feature_dims=False, data_dir='./data/'):
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

    def forward(self, x, decoder_batch_size, seq_len):
        """
        Encoder forward.

        Params:
            x: Input images with size: decoder_batch_size x seq_len x num_channels x height x width
                                      (decoder_batch_size x seq_len x 3 x 205 x 350 if use xception)

        Returns:
            encoder_outputs: Output features with size: decoder_batch_size x seq_len x num_loc x encoder_dim
                                                       (decoder_batch_size x seq_len x (7x11) x 2048 if use xception)
        """

        # (decoder_batch_size x seq_len) x encoder_dim x feature_width x feature_height
        x = self.model.features(x.view(-1, x.size(2), x.size(3), x.size(4)))
        encoder_outputs = x.view(decoder_batch_size, seq_len, x.size(1), -1).transpose(2, 3)
        return encoder_outputs


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

    def __init__(self, encoder_dim, decoder_dim, attention_dim, action_dim,
                 num_loc, y_keys_info, num_layers, dropout_prob=0.5):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.num_loc = num_loc
        self.y_keys_info = y_keys_info
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.att_model = Attention(num_loc, encoder_dim, decoder_dim, attention_dim, dropout_prob)

        self.emb_layer = {}
        for action, num_class in y_keys_info.items():
            self.emb_layer[action] = nn.Embedding(num_class, action_dim)
            self.add_module(name='emb_layer_%s'%action, module=self.emb_layer[action])

        self.lstm = nn.LSTM(
                input_size = encoder_dim + len(y_keys_info)*action_dim,
                hidden_size = decoder_dim,
                num_layers = num_layers,
                batch_first = True
                )

        self.fc_output = {}
        for action, num_class in y_keys_info.items():
            self.fc_output[action] = nn.Sequential(
                                            nn.Linear(decoder_dim, decoder_dim),
                                            nn.ReLU(),
                                            nn.Dropout(p=dropout_prob),

                                            nn.Linear(decoder_dim, num_class)
                                            )
            self.add_module(name='fc_output_%s'%action, module=self.fc_output[action])

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

        # decoder_batch_size x (seq_len + 1) x (action_dim * num_actions)
        actions_embs = torch.stack([self.emb_layer[action](actions[action]) for action in self.y_keys_info.keys()], dim=3)
        actions_embs = actions_embs.view(decoder_batch_size, seq_len+1, -1)

        y = {}
        for action, num_class in self.y_keys_info.items():
            y[action] = torch.zeros((decoder_batch_size, num_class, seq_len)).cuda()

        decoder_state = self._init_state(decoder_batch_size)

        # do not do seq_len + 1 since we do not need last y
        for i in range(seq_len):

            # (decoder_batch_size x num_loc x 1, decoder_batch_size x encoder_dim)
            alpha, attention_output = self.att_model.forward(encoder_outputs[:, i], encoder_atts[:, i], decoder_state[0])

            # decoder_batch_size x 1 x (encoder_dim + num_actions)
            # 1 means sequence length for decoder; since we have a for loop, seq_len here = 1
            decoder_input = torch.cat((attention_output, actions_embs[:, i]), dim=1).unsqueeze(1)

            # (decoder_batch_size x 1 x decoder_dim, (num_layers x decoder_batch_size x decoder_dim)*2)
            # 1 means sequence length for decoder; since we have a for loop, seq_len here = 1
            output, decoder_state = self.lstm(decoder_input, decoder_state)

            for action in self.y_keys_info.keys():
                y[action][:, :, i] = self.fc_output[action](output.squeeze(1))

        return y

#    def inference(self, encoder_outputs, init_action, decoder_batch_size, seq_len):
#        """
#        Decoder inference.
#
#        Params:
#            encoder_outputs: Output features from encoder with size: decoder_batch_size x seq_len x num_loc x encoder_dim
#            init_action: y_0, to start decoder forward process with size: decoder_batch_size x num_actions
#
#        Returns:
#            y: Predicted actions from decoder with size: decoder_batch_size x seq_len x num_actions
#        """
#
#        encoder_atts = self.att_model.generate_encoder_atts(encoder_outputs)  # decoder_batch_size x seq_len x num_loc x attention_dim
#        y = torch.zeros((decoder_batch_size, seq_len+1, self.num_actions)).cuda()  # +1 to make the for loop easier to implement
#        y[:, 0] = init_action
#
#        decoder_state = self._init_state(decoder_batch_size)
#
#        for i in range(seq_len):
#
#            # (decoder_batch_size x num_loc x 1, decoder_batch_size x encoder_dim)
#            alpha, attention_output = self.att_model.forward(encoder_outputs[:, i], encoder_atts[:, i], decoder_state[0])
#
#            # decoder_batch_size x 1 x (encoder_dim + num_actions)
#            # 1 means sequence length for decoder; since we have a for loop, seq_len here = 1
#            decoder_input = torch.cat((attention_output, y[:, i]), dim=1).unsqueeze(1)
#
#            # (decoder_batch_size x 1 x decoder_dim, (num_layers x decoder_batch_size x decoder_dim)*2)
#            # 1 means sequence length for decoder; since we have a for loop, seq_len here = 1
#            output, decoder_state = self.lstm(decoder_input, decoder_state)
#
#            y[:, i+1] = self.sigmoid(self.fc_output(output.squeeze(1)))  # decoder_batch_size x num_actions
#
#        return y[:, 1:]


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
    for i in range(10):
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

        optimizer.step()

        e = time.time()
#        print(e-s)
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
