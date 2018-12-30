# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 16:08:37 2018

@author: msq96
"""

# SOURCE: https://github.com/siqim/TensorKart/blob/master/utils.py

from inputs import get_gamepad
import threading


class XboxController():
    MAX_TRIG_VAL = 255
    MIN_TRIG_VAL = 0
    MAX_JOY_VAL = 32767
    MIN_JOY_VAL = -32768

    def __init__(self):

        self.LeftJoystickX = 0.5
        self.LeftTrigger = 0
        self.RightTrigger = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller,args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def _min_max_norm(self, state, min_val, max_val):
        return (state - min_val) / (max_val - min_val)

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_X':
                    self.LeftJoystickX = self._min_max_norm(event.state, self.MIN_JOY_VAL, self.MAX_JOY_VAL)
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = self._min_max_norm(event.state, self.MIN_TRIG_VAL, self.MAX_TRIG_VAL)
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = self._min_max_norm(event.state, self.MIN_TRIG_VAL, self.MAX_TRIG_VAL)

    def read(self):
        LX = self.LeftJoystickX  # left, right
        LT = self.LeftTrigger  # brake
        RT = self.RightTrigger  # accelerator
        return [LX, LT, RT]
