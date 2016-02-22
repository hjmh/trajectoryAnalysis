"""
Plotting methods using trajectory data in form of a pandas data frame
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import matplotlib.colors as colors


def oneDimResidency_df(radResPlt, FODataframe, movementFilter, visState, numBins, histRange):

    # normalisation factor for cirle area rings
    areaNorm = np.square(np.linspace(histRange[0], histRange[1], numBins))*np.pi
    areaNorm[1:] = areaNorm[1:]-areaNorm[:-1]

    # colormap for trials (visible object trials in colour, invisible object trials in grey shades)
    numInvTrials = sum(['invisible' in visibilityState[trial] for trial in range(len(visState))])
    numVisTrials = len(visState)-numInvTrials

    visTrialCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-2, vmax=numVisTrials), cmap='Reds')
    invTrialCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-2, vmax=numInvTrials), cmap='Greys')
    trialCMap = [visTrialCMap.to_rgba(visTrial) for visTrial in range(numVisTrials)]
    [trialCMap.append(invTrialCMap.to_rgba(invTrial)) for invTrial in range(numInvTrials)]

    legendtext = []

    for trial, cond in enumerate(visState):
        querystring = '(trialtype=="'+cond+'") &(trial=='+str(trial+1)+')&('+movementFilter+')'

        xPosMA = np.asarray(FODataframe.query(querystring).iloc[:, keyind_xPos:keyind_xPos+1]).squeeze()
        yPosMA = np.asarray(FODataframe.query(querystring).iloc[:, keyind_yPos:keyind_yPos+1]).squeeze()

        # transform trajectory to polar coordinates
        objDist, theta = cartesian2polar(xPosMA, yPosMA)

        [radresidency, edges] = np.histogram(objDist, bins=numBins, range=histRange)
        radResPlt.plot(edges[:-1]+np.diff(edges)/2.0, np.log(radresidency/areaNorm), color=trialCMap[trial])

        legendtext.append(cond + ' t' + str(trial+1))

    plt.legend(legendtext, loc='best', fontsize=12)
    radResPlt.set_xlabel('object distance [mm]', fontsize=12)
    radResPlt.set_ylabel('log(area corrected residency)', fontsize=12)
    radResFig.tight_layout()

    return radResPlt


def cartesian2polar(xPosFly, yPosFly):
    raddist = np.hypot(xPosFly, yPosFly)
    theta = np.arctan2(yPosFly, xPosFly) + np.pi

    return raddist, theta


def plotVeloHeadingDistribution_flyVR_df(mydataframe, trialtype, trial, flyIDs, keylist, vTransTH):
    """ Plot velocity and relative heading distributions (non-normalised) for a set of flies """

    numFlies = len(flyIDs)
    flyCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=numFlies), cmap='Accent')

    vRotRange = (-10, 10)

    veloDistFig = plt.figure(figsize=(15, 4))

    gs = gridspec.GridSpec(1, 4, width_ratios=(4, 4, 4, 1.25))

    vTsubplt = veloDistFig.add_subplot(gs[0])
    vTsubplt.set_xlabel('translational velocity [mm/s]')
    vTsubplt.set_ylabel('count')
    sns.despine(right=True, offset=5, trim=False)
    sns.axes_style({'axes.linewidth': 1, 'axes.edgecolor': '.8'})

    vRsubplt = veloDistFig.add_subplot(gs[1])
    vRsubplt.set_xlabel('rotational velocity [rad/s]')
    sns.despine(right=True, offset=5, trim=False)
    sns.axes_style({'axes.linewidth': 1, 'axes.edgecolor': '.8'})

    hsubplt = veloDistFig.add_subplot(gs[2])
    hsubplt.set_xlabel('heading [rad]')
    sns.despine(right=True, offset=5, trim=False)
    sns.axes_style({'axes.linewidth': 1, 'axes.edgecolor': '.8'})

    keyind_vT = keylist.index('transVelo')
    keyind_h = keylist.index('gamma')
    keyind_vR = keylist.index('rotVelo')

    nhAll = np.zeros((18,numFlies))
    meanSpeeds = np.zeros(numFlies)

    for flyInd, flyID in enumerate(flyIDs):
        querystring = '(moving>0) & (trialtype == "'+trialtype+'") & (trial=='+trial+') & (flyID == "'+flyID+'")'
        tV = mydataframe.query(querystring).iloc[:, keyind_vT:keyind_vT+1].squeeze()
        rV = mydataframe.query(querystring).iloc[:, keyind_vR:keyind_vR+1].squeeze()

        # v trans
        ntV, binEdges=np.histogram(tV, bins=50, range=(vTransTH, 30))
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        vTsubplt.plot(bincenters, ntV, alpha=0.7, color=flyCMap.to_rgba(flyInd))

        # v rot
        nrV, binEdges = np.histogram(rV, bins=50, range=vRotRange)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        vRsubplt.plot(bincenters, nrV, alpha=0.7, color=flyCMap.to_rgba(flyInd))

        # heading
        querystring = '(moving>0)  & (objectDistance>6) & (objectDistance<51) & (trialtype == "'+trialtype+'") & (trial=='+trial+') & (flyID == "'+flyID+'")'
        h = mydataframe.query(querystring).iloc[:, keyind_h:keyind_h+1].squeeze()
        tV = mydataframe.query(querystring).iloc[:, keyind_vT:keyind_vT+1].squeeze()
        nh, binEdges=np.histogram(h, bins=18, range=(0, np.pi))
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        if np.mean(tV) > 2.0:#sum(nh)/(20*6) > 10:
            nhAll[:,flyInd] = nh
            meanSpeeds[flyInd] = np.mean(tV)

    X, Y = np.meshgrid(range(0,numFlies+1), binEdges)
    sortedSpeeds = [i[0] for i in sorted(enumerate(meanSpeeds), key=lambda x:x[1])]
    nhAll_sorted = nhAll[:,sortedSpeeds]
    toPlot = nhAll_sorted/sum(nhAll_sorted)
    toPlot[np.isnan(toPlot)] = 0
    hsubplt.pcolormesh(Y, X, toPlot,cmap='Greys')
    hsubplt.set_title(' 6mm < objectDistance < 51 mm ')
    hsubplt.yaxis.set_visible(False)
    for fly in range(numFlies):
        hsubplt.text(np.pi+0.1,sortedSpeeds[fly]+.5, flyIDs[fly], fontsize=12)

    vTsubplt.set_xlim((vTransTH, 30))
    vRsubplt.set_xlim(vRotRange)
    hsubplt.set_xlim((0, np.pi))

    veloDistFig.suptitle(trialtype+' object, trial '+trial+' (moving > '+str(vTransTH)+' mm/s)', fontsize=12)
    vTsubplt.legend(flyIDs, ncol=2, loc=1, fontsize=8)

    return veloDistFig