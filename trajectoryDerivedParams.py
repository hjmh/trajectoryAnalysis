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
    # angle[np.pi/180*FOData[:,5]>np.pi] = angle[np.pi/180*FOData[:,5] > np.pi] - np.pi
    # angle[np.pi/180*FOData[:,5]<np.pi] = angle[np.pi/180*FOData[:,5] < np.pi] + np.pi

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


def dotproduct2d(a, b):
    # 2D dot product
    return a[0, :]*b[0, :] + a[1, :]*b[1, :]


def veclength2d(vec):
    return np.sqrt(vec[0, :]**2 + vec[1, :]**2)


def relationToObject(time, xPos, yPos, angle, objLocation):
    # Assumes only one object, thus in case of fly VR one needs to use the projected xPosMA

    # Vector to object location
    objDirection = np.vstack((objLocation[0]-xPos, objLocation[1]-yPos))

    objDistance = veclength2d(objDirection)

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


def computeCurvature(xPos, yPos, time, sigmaVal):
    from scipy.ndimage.filters import gaussian_filter

    nPts = len(xPos)

    # Compute first and second derivatives of x and y w.r.t. to t
    dxdt = np.zeros(nPts)
    dydt = np.zeros(nPts)

    # Smooth position and partial derivatives with gaussian kernel before taking numerical derivative
    sigma = sigmaVal
    x_filt = gaussian_filter(xPos, sigma, mode='reflect')
    y_filt = gaussian_filter(yPos, sigma, mode='reflect')

    dxdt[1:] = np.diff(x_filt)/np.diff(time)
    dydt[1:] = np.diff(y_filt)/np.diff(time)

    ddxdt = np.zeros(nPts)
    ddydt = np.zeros(nPts)

    sigma = sigmaVal
    dxdt_filt = gaussian_filter(dxdt, sigma, mode='reflect')
    dydt_filt = gaussian_filter(dydt, sigma, mode='reflect')

    ddxdt[1:] = np.diff(dxdt_filt)/np.diff(time)
    ddydt[1:] = np.diff(dydt_filt)/np.diff(time)

    # Compute and return curvature
    curvature = (dxdt*ddydt - dydt*ddxdt) / (dxdt*dxdt + dydt*dydt)**(3.0/2.0)

    return curvature
