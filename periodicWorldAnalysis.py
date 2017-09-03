"""
Methods related to analysis of fly trajectories in periodic virtual reality worlds
"""

import numpy as np


def collapseToMiniArena(xCoord, yCoord, arenaRad, objectCoords):
    """ Collapse to radial symmetric, circular 'mini-arena' while preserving the global heading """

    xCoordMA = np.copy(xCoord)
    yCoordMA = np.copy(yCoord)

    distToCone = np.zeros(len(xCoord))

    # Iterate through the objects...
    for obj in range(len(objectCoords)):
        # ...find points that are close and project them to the origin
        distToCone[:] = np.hypot(xCoord-objectCoords[obj, 0], yCoord-objectCoords[obj, 1])

        closepts = distToCone <= arenaRad
        xCoordMA[closepts] = xCoord[closepts] - objectCoords[obj, 0]
        yCoordMA[closepts] = yCoord[closepts] - objectCoords[obj, 1]

    # Remove remaining pts (= pts that are  > arenaRad away from origin)
    distToCone[:] = np.hypot(xCoordMA, yCoordMA)

    xCoordMA[distToCone > arenaRad] = np.nan
    yCoordMA[distToCone > arenaRad] = np.nan

    return xCoordMA, yCoordMA

def collapseTwoObjGrid(xFO, yFO, gridSize, gridRepeat):
    """ Collapes square grid to one square tile, preserving global heading"""
    nFrames = len(xFO)
    xPosMA = np.copy(xFO)
    yPosMA = np.copy(yFO)

    for frame in range(nFrames):

        # collapse in y
        if yFO[frame] > gridRepeat[1]*gridSize:
            yPosMA[frame] = np.NaN
        elif yFO[frame] > (gridRepeat[1]-2)*gridSize:
            yPosMA[frame] = yFO[frame] - 4*gridSize
        elif yFO[frame] > (gridRepeat[1]-4)*gridSize:
            yPosMA[frame] = yFO[frame] - 2*gridSize
        elif yFO[frame] < -gridRepeat[1]*gridSize:
            yPosMA[frame] = np.NaN
        elif yFO[frame] < -(gridRepeat[1]-2)*gridSize:
            yPosMA[frame] = yFO[frame] + 4*gridSize
        elif yFO[frame] < -(gridRepeat[1]-4)*gridSize:
            yPosMA[frame] = yFO[frame] + 2*gridSize

        # collapse in x
        if xFO[frame] > gridRepeat[0]*gridSize:
            xPosMA[frame] = np.NaN
        elif xFO[frame] > (gridRepeat[0]-2)*gridSize:
            xPosMA[frame] = xFO[frame] - 4*gridSize
        elif xFO[frame] > (gridRepeat[0]-4)*gridSize:
            xPosMA[frame] = xFO[frame] - 2*gridSize
        elif xFO[frame] < -gridRepeat[0]*gridSize:
            xPosMA[frame] = np.NaN
        elif xFO[frame] < -(gridRepeat[0]-2)*gridSize:
            xPosMA[frame] = xFO[frame] + 4*gridSize
        elif xFO[frame] < -(gridRepeat[0]-4)*gridSize:
            xPosMA[frame] = xFO[frame] + 2*gridSize
        elif xFO[frame] < 0:
            xPosMA[frame] = xFO[frame] + 2*gridSize

        if xPosMA[frame] < 0:
            xPosMA[frame] = xPosMA[frame] + 2*gridSize

    return xPosMA, yPosMA
