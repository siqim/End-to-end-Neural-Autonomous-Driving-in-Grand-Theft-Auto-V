# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 23:18:37 2019

@author: msq96
"""


import time
from deepgtav.messages import Start, Stop, Config, Dataset, Scenario
from deepgtav.client import Client
from utils import set_logger


def reset():
    dataset = Dataset(rate=rate, frame=frame, throttle=True, brake=True, steering=True, speed=True, drivingMode=True, location=True)
    scenario = Scenario(weather='EXTRASUNNY', vehicle='voltic', time=[12, 0], drivingMode=[1074528293, max_speed/1.6],
                        location=[-3000, 2500])
    client.sendMessage(Config(scenario=scenario, dataset=dataset))


if __name__ == '__main__':
    logger = set_logger()

    host = 'localhost'
    port = 8000
    dataset_path = 'dataset.pz'
    log_freq = 10 # in minute
    max_stop_time = 10 # in second
    max_wall_time = 10 # in hour
    max_speed = 120 # in km
    rate = 30 # in HZ
    frame = [350, 205+20]


    logger.info('Rate %s hz, frame %s, max_stop_time %s, max_wall_time %s, max_speed %s, dataset_path %s.'
                % (str(rate), str(frame), str(max_stop_time), str(max_wall_time), str(max_speed), str(dataset_path)))

    client = Client(ip=host, port=port, datasetPath=dataset_path, compressionLevel=0)

    dataset = Dataset(rate=rate, frame=frame, throttle=True, brake=True, steering=True, speed=True, drivingMode=True, location=True)

    scenario = Scenario(weather='EXTRASUNNY', vehicle='voltic', time=[12, 0], drivingMode=[1074528293, max_speed/1.6],
                        location=[500, 500])

    client.sendMessage(Start(scenario=scenario, dataset=dataset))

    count = 0
    old_location = [0, 0, 0]

    print('Holding until receiving message...')
    message = client.recvMessage()
    for i in range(1, 7+1)[::-1]:
        logger.info('Start recording in %d second(s)...'%i)
        time.sleep(1)

    stoptime = time.time() + max_wall_time*3600
    while time.time() < stoptime:
        message = client.recvMessage()

        if count % int(log_freq*60*rate) == 0:
            logger.info('%d frames have been saved!'%count)

        if count % int(max_stop_time*rate) == 0:
            new_location = message['location']

            if int(new_location[0]) == int(old_location[0]) and int(new_location[1]) == int(old_location[1]):
                reset()
                logger.warning('Reseting loction! Occurred in count %d'%count)

            old_location = message['location']
        count += 1


    client.sendMessage(Stop())
    client.close()