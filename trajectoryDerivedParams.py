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


def cartesian2polar(xPosFly,yPosFly):
    raddist = np.hypot(xPosFly, yPosFly)
    theta = np.arctan2(yPosFly, xPosFly) + np.pi

    return raddist, theta


def polarCurvature(theta, objdist):

    from scipy.ndimage.filters import gaussian_filter

    # unwrap theta before taking derivatives
    # thetaU = np.copy(theta)
    # thetaU[~np.isnan(theta)] = np.unwrap(theta[~np.isnan(theta)],discont=np.pi)

    # first derivatives of the distance and angle
    d_theta = np.hstack((0, np.diff(theta)))
    d_theta[d_theta > np.pi] -= 2*np.pi
    d_theta[d_theta <= -np.pi] += 2*np.pi
    d_theta[np.where(np.isnan(d_theta))[0]] = 0

    # filter derivatives
    sigma = 2
    d_theta_filt = gaussian_filter(d_theta, sigma, mode='reflect')

    d_objdist = np.hstack((0, np.diff(objdist)))/d_theta_filt

    d_objdist_filt = gaussian_filter(d_objdist, sigma, mode='reflect')

    # second derivative
    dd_objdist = np.hstack((0, np.diff(d_objdist_filt)))/d_theta_filt

    # compute curvature
    polarCurv = (objdist**2 + 2*(d_objdist**2) - objdist*dd_objdist)/(np.sqrt(objdist**2 + d_objdist**2)**3)

    return polarCurv, d_theta, d_objdist


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


def fitGMMtoSingleTrial(perFlyvT,plotDistr):
    # Based on code from Jake VanderPlas
    from sklearn.mixture import GMM

    # Set up the dataset.
    X = perFlyvT[perFlyvT < 30.0]

    # Learn the best-fit GMM models
    # Here we'll use GMM in the standard way: the fit() method uses an Expectation-Maximization approach to find the
    # best mixture of Gaussians for the data

    # fit models with 1-3 components
    N = np.arange(1, 4)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GMM(N[i]).fit(X)

    # compute the AIC and the BIC
    AIC = [m.aic(X) for m in models]

    # Plot the results: data + best-fit mixture

    M_best = models[np.argmin(AIC)]

    x = np.linspace(-1, 30, 1000)
    logprob, responsibilities = M_best.eval(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    # find intersection points
    crosspts = np.zeros(3)
    for model in range(3):
        try:
            crossovers = np.pad(np.diff(np.array(pdf_individual[:, model] > pdf_individual[:, (model+1) % 3]).astype(int)),
                                (1, 0), 'constant', constant_values=(0,))
            crosspts[model] = np.where(crossovers != 0)[0][0]
        except IndexError:
            crosspts[model] = 0

    walkingboutTH = max(2, x[max(crosspts)])

    if(plotDistr):
        fig = plt.figure(figsize=(10, 4))
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.21, top=0.9, wspace=0.5)
        ax = fig.add_subplot(111)
        ax.hist(X, 30, normed=True, histtype='stepfilled', alpha=0.4)
        ax.plot(x, pdf, '-k', alpha=0.7)
        ax.plot(x, pdf_individual, '--r', alpha=0.5)
        ax.plot(walkingboutTH, pdf_individual[max(crosspts), model], 'bo')
        ax.plot(x[crosspts[model]], pdf_individual[crosspts[model], model], 'ro')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p(x)$')
        ax.set_xlim(-1, 30)
        ax.set_ylim(0, 0.25)
        myAxisTheme(ax)

    return walkingboutTH
