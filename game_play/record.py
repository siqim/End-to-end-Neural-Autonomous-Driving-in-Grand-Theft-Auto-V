# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 16:08:37 2018

@author: msq96
"""

# SOURCE: https://github.com/Sentdex/pygta5/blob/master/1.%20collect_data.py

import cv2
import time
import numpy as np

from xbox import XboxController
from helper_func import pause_check, grab_screen, show_grabbed_screen, init_starting_value


show_screen = False
save = True
paused = False

controller = XboxController()
root_path = '../../data/raw_data/'
starting_value = init_starting_value(root_path)
training_data = []

for i in range(5)[::-1]:
    time.sleep(1)
    print(i)
print('Start recording!')

while 1:

    if not paused:
        last_time = time.time()
        screen = grab_screen(region=(60, 25, 1280-60, 1024))
        screen = cv2.resize(screen, (331, 331), interpolation=cv2.INTER_AREA)

        controller_data = controller.read()
        training_data.append([screen, controller_data])

        if show_screen:
            show_grabbed_screen(screen)
        left_right, down, up = controller_data

        # To make frames consistent during training and inferencing
        time.sleep(0.025)

        if save:
            if len(training_data) % 100 == 0:
                print(len(training_data))

                if len(training_data) == 2000:
                    file_name = root_path + 'training_data-{}.npy'.format(starting_value)
                    np.save(file_name, training_data)
                    print('SAVED', starting_value)
                    training_data = []
                    starting_value += 1

#        print("left_right: %.2f, down: %.2f, up: %.2f" % (left_right, down, up))
#        print('loop took %.4f seconds' % (time.time()-last_time))

    if pause_check():
        if paused:
            paused = False
            print('unpaused!')
            time.sleep(1)
        else:
            print('Pausing!')
            paused = True
            time.sleep(1)
