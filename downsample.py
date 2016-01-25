"""
Functions for reading in object coordinates
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
from scipy.interpolate import interp1d


def donwsampleFOData(targedFrequency, logTime, time, xPos, yPos, angle):
    """ Down sample data to targedFrequency Hz """
    time_ds = np.linspace(0, logTime[-1], int(targedFrequency*round(logTime[-1])))

    numFrames_ds = len(time_ds)

    f_xPos = interp1d(time, xPos, kind='linear')
    f_yPos = interp1d(time, yPos, kind='linear')

    f_angle = interp1d(time, angle, kind='linear')

    xPos_ds = f_xPos(time_ds)
    yPos_ds = f_yPos(time_ds)

    angle_ds = f_angle(time_ds)

    return time_ds, xPos_ds, yPos_ds, angle_ds, numFrames_ds