# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 16:08:37 2018

@author: msq96
"""

# SOURCE: https://gist.github.com/Flandan/fdadd7046afee83822fcff003ab47087#file-vjoy-py

import ctypes
import struct
import numpy as np


CONST_DLL_VJOY = "play.dll"


class vJoy():
    MAX_VAL = 32768
    MIN_VAL = 0

    def __init__(self, reference = 1):
        self.handle = None
        self.dll = ctypes.CDLL(CONST_DLL_VJOY)
        self.reference = reference
        self.acquired = False

    def _generateJoystickPosition(self,
        wThrottle = 0, wRudder = 0, wAileron = 0,
        # left thb x        left thb y     left trigger
        wAxisX = 16384,   wAxisY = 16384,   wAxisZ = 0,
        # right thb x       right thb y        right trigger
        wAxisXRot = 16384, wAxisYRot = 16384, wAxisZRot = 0,

        wSlider = 0, wDial = 0, wWheel = 0, wAxisVX = 0, wAxisVY = 0, wAxisVZ = 0, wAxisVBRX = 0,
        wAxisVBRY = 0, wAxisVBRZ = 0, lButtons = 0, bHats = 0, bHatsEx1 = 0, bHatsEx2 = 0, bHatsEx3 = 0):
        """
        typedef struct _JOYSTICK_POSITION
        {
            BYTE    bDevice; // Index of device. 1-based
            LONG    wThrottle;
            LONG    wRudder;
            LONG    wAileron;
            LONG    wAxisX;
            LONG    wAxisY;
            LONG    wAxisZ;
            LONG    wAxisXRot;
            LONG    wAxisYRot;
            LONG    wAxisZRot;
            LONG    wSlider;
            LONG    wDial;
            LONG    wWheel;
            LONG    wAxisVX;
            LONG    wAxisVY;
            LONG    wAxisVZ;
            LONG    wAxisVBRX;
            LONG    wAxisVBRY;
            LONG    wAxisVBRZ;
            LONG    lButtons;   // 32 buttons: 0x00000001 means button1 is pressed, 0x80000000 -> button32 is pressed
            DWORD   bHats;      // Lower 4 bits: HAT switch or 16-bit of continuous HAT switch
                        DWORD   bHatsEx1;   // 16-bit of continuous HAT switch
                        DWORD   bHatsEx2;   // 16-bit of continuous HAT switch
                        DWORD   bHatsEx3;   // 16-bit of continuous HAT switch
        } JOYSTICK_POSITION, *PJOYSTICK_POSITION;
        """
        joyPosFormat = "BlllllllllllllllllllIIII"
        pos = struct.pack(joyPosFormat, self.reference, wThrottle, wRudder,
                                   wAileron, wAxisX, wAxisY, wAxisZ, wAxisXRot, wAxisYRot,
                                   wAxisZRot, wSlider, wDial, wWheel, wAxisVX, wAxisVY, wAxisVZ,
                                   wAxisVBRX, wAxisVBRY, wAxisVBRZ, lButtons, bHats, bHatsEx1, bHatsEx2, bHatsEx3)
        return pos

    def _min_max_norm(self, output):
        return [int(each_button*self.MAX_VAL) for each_button in output]

    def _update(self, joystickPosition):
        if self.dll.UpdateVJD(self.reference, joystickPosition):
            return True
        return False

    def open(self):
        if self.dll.AcquireVJD(self.reference):
            self.acquired = True
            return True
        return False

    def run(self, output):
        LX, LT, RT = vj._min_max_norm(output)
        joystickPosition = vj._generateJoystickPosition(wAxisX = LX, wAxisZ=LT, wAxisZRot=RT)
        vj._update(joystickPosition)

    def stop(self):
        vj.run([0.5, 0, 0])

    def close(self):
        if self.dll.RelinquishVJD(self.reference):
            self.acquired = False
            return True
        return False


if __name__ == '__main__':

    outputs = np.random.uniform(size=(1000, 3))
    outputs[:, 0] = 0.5
    outputs[:, 1] = 0
    outputs[:, 2] = 1

    vj = vJoy()

    vj.open()
    for output in outputs:
        vj.run(output)
    vj.stop()
    vj.close()
