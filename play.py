# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 01:03:17 2019

@author: msq96
"""


from game_play.deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy
from game_play.deepgtav.client import Client

import cv2
import time
import numpy as np
from utils import get_models
from torchvision import transforms
import torch
import pickle


if __name__ == '__main__':

    host = 'localhost'
    port = 8000
    max_stop_time = 10 # in second
    max_wall_time = 10 # in hour
    frame = [350, 205+20]

    mean = [0.35739973, 0.35751262, 0.36058474]
    std = [0.19338858, 0.19159749, 0.2047393 ]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    print('Loading model...')
    encoder, decoder, init_y = get_models()
    y_bin = pickle.load(open('./data/y_bin_info.pickle', 'rb'))


    client = Client(ip=host, port=port)

    scenario = Scenario(weather='EXTRASUNNY', vehicle='voltic', time=[12, 0], drivingMode=-1,
                        location=[-2500, 3250])

    client.sendMessage(Start(scenario=scenario))

    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        input_action = {k: v[0,:].cuda() for k, v in init_y.items()}
        decoder_state = decoder._init_state(1)

        stoptime = time.time() + max_wall_time*3600
        while time.time() < stoptime:
            message = client.recvMessage()
            image = frame2numpy(message['frame'], frame)[20:]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image).unsqueeze(0).cuda()

            encoder_output = encoder.inference(image)
            input_action, decoder_state = decoder.inference(encoder_output, input_action, decoder_state)

            commands = [y_bin['throttle'][input_action['throttle'].item()]['mean'],
                        0,
                        y_bin['steering'][input_action['steering'].item()]['mean']]
            print(commands)


            client.sendMessage(Commands(commands[0], commands[1], commands[2]))

    client.sendMessage(Stop())
    client.close()
