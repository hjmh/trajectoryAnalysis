"""
Methods related to analysis of fly trajectories in periodic virtual reality worlds
"""

import numpy as np


def collapseToMiniArena(xCoord, yCoord, arenaRad, objectCoords):
    """ Collapse to 'mini-arena' while preserving the global heading """

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
