"""
Methods related to extraction of movement parameter that can be extracted from a fly's trajectory
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
from scipy.interpolate import interp1d


def convertRawHeadingAngle(angleRaw):
    """ Convert heading angle (FlyOver) to rad """
    angle = np.zeros(len(angleRaw))
    angle[:] = np.pi/180*angleRaw

    return angle


def velocityFromTrajectory(time, angle, xPos, yPos, N, numFrames):
    """ Compute movement velocities """

    # Compute translational and rotational velocity
    vTrans = np.zeros(numFrames)
    vTrans[0:-1] = np.hypot(np.diff(xPos), np.diff(yPos)) / np.diff(time)
    vTrans[np.where(np.isnan(vTrans))[0]] = 0

    vRot = np.zeros(numFrames)
    vRot[0:-1] = np.diff(angle)
    vRot[vRot > np.pi] -= 2*np.pi
    vRot[vRot <= -np.pi] += 2*np.pi
    vRot[0:-1] = vRot[0:-1] / np.diff(time)
    vRot[np.where(np.isnan(vRot))[0]] = 0

    # Filter translational and rotational velocities
    vTransFilt = np.convolve(vTrans, np.ones((N,))/N, mode='same')
    vRotFilt = np.convolve(vRot, np.ones((N,))/N, mode='same')

    return vTrans, vRot, vTransFilt, vRotFilt


def cartesian2polar(xPosFly,yPosFly):
    raddist = np.hypot(xPosFly, yPosFly)
    theta = np.arctan2(yPosFly, xPosFly) + np.pi
    return raddist, theta


def dotproduct2d(a, b):
    # 2D dot product
    return a[0, :]*b[0, :] + a[1, :]*b[1, :]


def veclength2d(vec):
    return np.sqrt(vec[0, :]**2 + vec[1, :]**2)


def relationToObject(time, xPos, yPos, angle, objLocation):
    # Assumes only one object, thus in case of fly VR one needs to use the projected xPosMA

    # Vector to object location
    objDirection = np.vstack((objLocation[0]-xPos, objLocation[1]-yPos))

    objDistance = np.hypot(objLocation[0]-xPos, objLocation[1]-yPos)

    # Fly orientation vector
    flyDirection = np.vstack((np.cos(angle), np.sin(angle)))

    # Angle to object relative from fly's orientation
    lenFlyVec = np.hypot(flyDirection[0, :], flyDirection[1, :])
    lenObjVec = np.hypot(objDirection[0, :], objDirection[1, :])

    gamma = np.arccos(dotproduct2d(flyDirection, objDirection) / (lenFlyVec * lenObjVec))

    gammaFull = np.arctan2(flyDirection[1, :], flyDirection[0, :])-np.arctan2(objDirection[1, :], objDirection[0, :])
    gammaFull[gammaFull < 0] += 2 * np.pi
    gammaFull[gammaFull > np.pi] -= 2 * np.pi

    gammaV = np.hstack((np.diff(gamma)/np.diff(time), 0))

    return objDirection, objDistance, gammaFull, gamma, gammaV



def countVisits(dist2Obj, visitRad):
    """ Function related to landmark visit analysis, operates on trajectory in polar coordinates. """

    inside = (dist2Obj < visitRad).astype('int')
    time = np.linspace(0, 600, len(dist2Obj))

    entries = np.zeros(len(inside))
    entries[1:] = np.diff(inside) == 1

    exits = np.zeros(len(inside))
    exits[1:] = np.diff(inside) == -1

    entryTime = time[entries.astype('bool')]
    exitTime = time[exits.astype('bool')]
    if len(entryTime) != len(exitTime):
        visitT = exitTime[0:min(sum(exits), sum(entries))] - entryTime[0:min(sum(exits), sum(entries))]
    else:
        visitT = exitTime - entryTime

    return entries.astype('bool'), exits.astype('bool'), visitT, entryTime, exitTime


def countVisits_df(visitRad, flyID, trial, allFlies_df, keyind_xPos, keyind_yPos):
    """ Function related to landmark visit analysis, operates on pandas data frame 'allFlies_df'. """

    querystring = '(trial==' + str(trial) + ') & (flyID == "'+flyID+'")'
    xPosTrial = np.asarray(allFlies_df.query(querystring).iloc[:, keyind_xPos:keyind_xPos+1]).squeeze()
    yPosTrial = np.asarray(allFlies_df.query(querystring).iloc[:, keyind_yPos:keyind_yPos+1]).squeeze()
    objDistTrial, thetaTrial = cartesian2polar(xPosTrial, yPosTrial)

    entries, exits, visitT, entryTime, exitTime = countVisits(objDistTrial, visitRad)

    return entries, exits, visitT, entryTime, exitTime


def visitTime_df(visitRad, flyID, trial, allFlies_df, keyind_xPos, keyind_yPos):
    """ Function related to landmark visit analysis, operates on pandas data frame 'allFlies_df'. """

    querystring = '(trial==' + str(trial) + ') & (flyID == "'+flyID+'")'
    xPosTrial = np.asarray(allFlies_df.query(querystring).iloc[:, keyind_xPos:keyind_xPos+1]).squeeze()
    yPosTrial = np.asarray(allFlies_df.query(querystring).iloc[:, keyind_yPos:keyind_yPos+1]).squeeze()

    objDistTrial, thetaTrial = cartesian2polar(xPosTrial, yPosTrial)

    inside = (objDistTrial < visitRad).astype('int')
    time = np.linspace(0, 600, len(xPosTrial))

    entries = np.zeros(len(inside))
    entries[1:] = np.diff(inside) == 1

    exits = np.zeros(len(inside))
    exits[1:] = np.diff(inside) == -1

    entryTime = time[entries.astype('bool')]
    exitTime = time[exits.astype('bool')]

    return entries, exits, time, entryTime, exitTime
