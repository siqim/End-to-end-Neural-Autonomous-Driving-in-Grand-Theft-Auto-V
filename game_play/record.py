# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 16:08:37 2018

@author: msq96
"""

# SOURCE: # SOURCE: https://github.com/Sentdex/pygta5/blob/master/1.%20collect_data.py


import time
import cv2

from xbox import XboxController
from helper_func import pause_check, grab_screen, show_grabbed_screen


controller = XboxController()
training_data = []
show_screen = True
paused = False
while 1:

    if not paused:
        last_time = time.time()
        screen = grab_screen(region=(60, 25, 1280-60, 1024))
        screen = cv2.resize(screen, (299, 299), interpolation=cv2.INTER_AREA)

        controller_data = controller.read()
        training_data.append([screen, controller_data])

        if show_screen:
            show_grabbed_screen(screen)

        left_right, down, up = controller_data
#        time.sleep(0.03)

        print("left_right: %.2f, down: %.2f, up: %.2f" % (left_right, down, up))
        print('loop took %.4f seconds' % (time.time()-last_time))

    if pause_check():
        if paused:
            paused = False
            print('unpaused!')
            time.sleep(1)
        else:
            print('Pausing!')
            paused = True
            time.sleep(1)
