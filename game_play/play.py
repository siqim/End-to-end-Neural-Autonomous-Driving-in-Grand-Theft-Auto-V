# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 01:03:17 2019

@author: msq96
"""


from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy
from deepgtav.client import Client

import cv2
import time
import numpy as np


class Model(object):
    def run(self, image):
        return [0.5, 0.0, 0.3] # throttle, brake, steering


if __name__ == '__main__':

    host = 'localhost'
    port = 8000
    max_stop_time = 10 # in second
    max_wall_time = 10 # in hour
    frame = [350, 205+20]

    print('Loading model...')
    model = Model()


    client = Client(ip=host, port=port)

    scenario = Scenario(weather='EXTRASUNNY', vehicle='voltic', time=[12, 0], drivingMode=-1,
                        location=[-2500, 3250])

    client.sendMessage(Start(scenario=scenario))

    stoptime = time.time() + max_wall_time*3600
    while time.time() < stoptime:
        message = client.recvMessage()
        image = frame2numpy(message['frame'], frame)[20:]

        commands = model.run(image)

        client.sendMessage(Commands(commands[0], commands[1], commands[2]))

    client.sendMessage(Stop())
    client.close()
