"""
Plotting methods using trajectory data in form of a pandas data frame
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as snss
import matplotlib.colors as colors

import seaborn as sns


def myAxisTheme(myax):
    myax.get_xaxis().tick_bottom()
    myax.get_yaxis().tick_left()
    myax.spines['top'].set_visible(False)
    myax.spines['right'].set_visible(False)


# 1D (polar) residency .................................................................................................

def oneDimResidency_df(radResPlt, FODataframe, keyind_xPos, keyind_yPos, movementFilter, visState, numBins, histRange,
                       lineAlpha, plotLog):

    # normalisation factor for cirle area rings
    areaNormA = np.square(np.linspace(histRange[0], histRange[1], numBins+1))*np.pi
    areaNorm = areaNormA[1:]-areaNormA[:-1]

    # colormap for trials (visible object trials in colour, invisible object trials in grey shades)
    numInvTrials = sum(['invisible' in visState[trial] for trial in range(len(visState))])
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
        if plotLog:
            radResPlt.plot(edges[:-1]+np.diff(edges)/2.0, np.log(radresidency/areaNorm),
                           color=trialCMap[trial], alpha=lineAlpha)
        else:
            radResPlt.plot(edges[:-1]+np.diff(edges)/2.0, radresidency/areaNorm,
                           color=trialCMap[trial], alpha=lineAlpha)

        legendtext.append(cond + ' t' + str(trial+1))

    plt.legend(legendtext, loc='best', fontsize=12)
    radResPlt.set_xlabel('object distance [mm]', fontsize=12)
    if plotLog:
        radResPlt.set_ylabel('log(area corrected residency)', fontsize=12)
    else:
        radResPlt.set_ylabel('area corrected residency', fontsize=12)

    return radResPlt


def oneDimResidencyWithVar_df(radResPlt, FODataframe, flyIDs, keyind_xPos, keyind_yPos, movementFilter, visState,
                              numBins, histRange, lineAlpha, plotLog, varstyle, fill, condLegend):

    # normalisation factor for cirle area rings
    areaNormA = np.square(np.linspace(histRange[0], histRange[1], numBins))*np.pi
    areaNorm = areaNormA[1:]-areaNormA[:-1]

    # colormap for trials (visible object trials in colour, invisible object trials in grey shades)
    numInvTrials = sum(['invisible' in visState[trial] for trial in range(len(visState))])
    numVisTrials = len(visState)-numInvTrials

    visTrialCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-2, vmax=numVisTrials), cmap='Reds')
    invTrialCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-2, vmax=numInvTrials), cmap='Greys')
    trialCMap = [visTrialCMap.to_rgba(visTrial) for visTrial in range(numVisTrials)]
    [trialCMap.append(invTrialCMap.to_rgba(invTrial)) for invTrial in range(numInvTrials)]

    numFlies = len(flyIDs)

    legendhand = []

    for trial, cond in enumerate(visState):
        trialRadRes = np.zeros((numFlies, numBins-1))
        for fly in range(numFlies):
            querystring = '(trialtype=="'+cond+'")&(trial=='+str(trial+1)+')&('+movementFilter\
                            +')&(flyID=="'+flyIDs[fly]+'")'

            xPosMA = np.asarray(FODataframe.query(querystring).iloc[:, keyind_xPos:keyind_xPos+1]).squeeze()
            yPosMA = np.asarray(FODataframe.query(querystring).iloc[:, keyind_yPos:keyind_yPos+1]).squeeze()

            # transform trajectory to polar coordinates
            objDist, theta = cartesian2polar(xPosMA, yPosMA)

            radresidency, edges = np.histogram(objDist, bins=np.linspace(histRange[0], histRange[1], numBins))
            trialRadRes[fly, :] = radresidency/areaNorm

            jitterRange = 0.2*np.diff(histRange)/numBins

            if varstyle == 'dotplot':
                if plotLog:
                    toplot = np.log(radresidency/areaNorm)
                else:
                    toplot = radresidency/areaNorm

                radResPlt.plot(edges[:-1]+np.diff(edges)/2.0+np.random.uniform(-jitterRange,jitterRange), toplot,
                               color=trialCMap[trial], linestyle='none', marker='.', alpha=0.5)

        if plotLog:
            if varstyle == 'std':
                toplot = np.log(np.nanmean(trialRadRes,0))
                var1 = np.log(np.nanmean(trialRadRes,0) + np.nanstd(trialRadRes,0))
                var2 = np.log(np.nanmean(trialRadRes,0) - np.nanstd(trialRadRes,0))
            elif varstyle == 'iqr':
                toplot = np.log(np.nanmedian(trialRadRes,0))
                [var1, var2] = np.log(np.nanpercentile(trialRadRes,[25,75],axis=0))
            else:
                toplot = np.log(np.nanmean(trialRadRes,0))

        else:
            if varstyle == 'std':
                toplot = np.nanmean(trialRadRes,0)
                var1 = toplot + np.nanstd(trialRadRes,0)
                var2 = toplot - np.nanstd(trialRadRes,0)
            elif varstyle == 'iqr':
                toplot = np.nanmedian(trialRadRes,0)
                [var1, var2] = np.nanpercentile(trialRadRes,[25,75],axis=0)
            else:
                toplot = np.nanmean(trialRadRes,0)

        if varstyle != 'dotplot':
            lhand, = radResPlt.plot(edges[:-1]+np.diff(edges)/2.0, toplot, color=trialCMap[trial],
                                    alpha=lineAlpha,linewidth=3)
            radResPlt.plot(edges[:-1]+np.diff(edges)/2.0, var1,color=trialCMap[trial], alpha=lineAlpha,linewidth=1)
            radResPlt.plot(edges[:-1]+np.diff(edges)/2.0, var2,color=trialCMap[trial], alpha=lineAlpha,linewidth=1)
            if fill:
                #radResPlt.fill_between(edges[:-1]+np.diff(edges)/2.0, np.maximum(var1,-6), np.maximum(var2,-6), color=trialCMap[trial], alpha=0.2)
                radResPlt.fill_between(edges[:-1]+np.diff(edges)/2.0, var1, var2, color=trialCMap[trial], alpha=0.2)
        else:
            lhand, = radResPlt.plot(edges[:-1]+np.diff(edges)/2.0, toplot, color=trialCMap[trial], alpha=lineAlpha,linewidth=3)

        legendhand.append(lhand)

    plt.legend(legendhand, condLegend, loc='best', fontsize=12)
    radResPlt.set_xlabel('object distance [mm]', fontsize=12)
    if plotLog:
        radResPlt.set_ylabel('log(area corrected residency)', fontsize=12)
    else:
        radResPlt.set_ylabel('area corrected residency', fontsize=12)

    return radResPlt


# Turn count vs. radial distance (from object / arena center) ..........................................................

def getTurnCounts(selectpts, allturns, leftturns, rightturns, objDist, nBins, histDRange):
    pts_turn = np.logical_and(selectpts, allturns)
    pts_turnR = np.logical_and(selectpts, rightturns)
    pts_turnL = np.logical_and(selectpts, leftturns)

    nTL, _ = np.histogram(objDist[pts_turnL], bins=nBins, range=histDRange)
    nTR, _ = np.histogram(objDist[pts_turnR], bins=nBins, range=histDRange)
    nT, _ = np.histogram(objDist[pts_turn], bins=nBins, range=histDRange)
    nDt, edges = np.histogram(objDist[selectpts], bins=nBins, range=histDRange)

    return nTL, nTR, nT, nDt, edges


def getTurnHistCounts(rotMeas, objDist, tTH, tTH_neg, tTH_pos, nBins, histDRange):
    d_objDist = np.hstack((0, np.diff(objDist)))
    pts_apr = d_objDist < 0
    pts_dep = d_objDist > 0

    # approach
    nTL_apr, nTR_apr, nT_apr, nDt_apr, edges = \
        getTurnCounts(pts_apr, abs(rotMeas) > tTH, rotMeas > tTH_pos, rotMeas < tTH_neg, objDist, nBins, histDRange)
    # departure
    nTL_dep, nTR_dep, nT_dep, nDt_dep, edges = \
        getTurnCounts(pts_dep, abs(rotMeas) > tTH, rotMeas > tTH_pos, rotMeas < tTH_neg, objDist, nBins, histDRange)

    return nTL_apr, nTL_dep, nTR_apr, nTR_dep, nT_apr, nT_dep, nDt_apr, nDt_dep, edges


def getTurnChangeHistCounts(rotMeasure, objDist, turnTH, turnTH_neg, turnTH_pos, numBins, histDRange, change):

    d_objDist = np.hstack((0, np.diff(objDist)))
    pts_apr = d_objDist < 0
    pts_dep = d_objDist > 0

    turns = (abs(rotMeasure) > turnTH).astype('int')
    tst = np.zeros(len(turns))
    tst[1:] = np.diff(turns) == change

    turnsL = (rotMeasure > turnTH_pos).astype('int')
    tstL = np.zeros(len(turnsL))
    tstL[1:] = np.diff(turnsL) == change

    turnsR = (rotMeasure < turnTH_neg).astype('int')
    tstR = np.zeros(len(turnsR))
    tstR[1:] = np.diff(turnsR) == change

    # approach
    nTL_apr, nTR_apr, nT_apr, nDt_apr, edges = getTurnCounts(pts_apr, tst, tstL, tstR, objDist, numBins, histDRange)
    # departure
    nTL_dep, nTR_dep, nT_dep, nDt_dep, edges = getTurnCounts(pts_dep, tst, tstL, tstR, objDist, numBins, histDRange)

    return nTL_apr, nTL_dep, nTR_apr, nTR_dep, nT_apr, nT_dep, nDt_apr, nDt_dep, edges


def turnRatePerDistance(Fig, FOAllFlies_df,keylistLong, visState, movementfilt, useTurnIndex, useTurnChange,
                        trialName, ylimrange, numBins, arenaRad):
    keyind_xPos = keylistLong.index('xPosInMiniarena')
    keyind_yPos = keylistLong.index('yPosInMiniarena')
    keyind_vT = keylistLong.index('transVelo')
    keyind_vR = keylistLong.index('rotVelo')

    # Find turnTH over all flies
    querystring = '('+movementfilt+') & (objectDistance>6)'
    vRot = np.asarray(FOAllFlies_df.query(querystring).iloc[:, keyind_vR:keyind_vR+1]).squeeze()

    if useTurnIndex:
        vtrans = np.asarray(FOAllFlies_df.query(querystring).iloc[:, keyind_vT:keyind_vT+1]).squeeze()
        vRot_filt = np.convolve(vRot/vtrans, np.ones((5,))/5, mode='same')
        vRot_filt[np.isinf(abs(vRot_filt))] = 0.0
    else:
        vRot_filt = np.convolve(vRot, np.ones((5,))/5, mode='same')

    turnTH_pos = 3*np.nanstd(vRot_filt[vRot_filt >= 0])
    turnTH_neg = -3*np.nanstd(vRot_filt[vRot_filt <= 0])
    turnTH = 3*np.nanstd(abs(vRot_filt))

    axApr = Fig.add_subplot(121)
    axApr.set_title('Approaches', fontsize=12)
    axDep = Fig.add_subplot(122)
    axDep.set_title('Departures', fontsize=12)

    n_invt = sum(['invisible' in visState[trial] for trial in range(len(visState))])
    n_vist = len(visState)-n_invt

    visAprCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-2, vmax=n_vist), cmap='Blues')
    visDepCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-2, vmax=n_vist), cmap='Greens')
    invAprCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-2, vmax=n_invt), cmap='Greys')
    invDepCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-2, vmax=n_invt), cmap='Greys')

    legendtext = []

    for trial, objecttype in enumerate(visState):
        querystring = '(trialtype=="' + objecttype + '") & (trial==' + str(trial+1) + ') & ('\
                      + movementfilt + ') & (objectDistance>6)'
        xPosFly = np.asarray(FOAllFlies_df.query(querystring).iloc[:, keyind_xPos:keyind_xPos+1]).squeeze()
        yPosFly = np.asarray(FOAllFlies_df.query(querystring).iloc[:, keyind_yPos:keyind_yPos+1]).squeeze()
        vRotFly = np.asarray(FOAllFlies_df.query(querystring).iloc[:, keyind_vR:keyind_vR+1]).squeeze()

        objDist, theta = cartesian2polar(xPosFly, yPosFly)

        if useTurnIndex:
            vtrans = np.asarray(FOAllFlies_df.query(querystring).iloc[:, keyind_vT:keyind_vT+1]).squeeze()
            vRotFly_filt = np.convolve(vRotFly/vtrans, np.ones((5,))/5, mode='same')
        else:
            vRotFly_filt = np.convolve(vRotFly, np.ones((5,))/5, mode='same')

        # Get counts
        if useTurnChange:
            nTL_apr, nTL_dep, nTR_apr, nTR_dep, nT_apr, nT_dep, nDt_apr, nDt_dep, edges\
                = getTurnChangeHistCounts(vRotFly_filt, objDist, turnTH, turnTH_neg, turnTH_pos,
                                          numBins, (6, 56), useTurnChange)
        else:
            nTL_apr, nTL_dep, nTR_apr, nTR_dep, nT_apr, nT_dep, nDt_apr, nDt_dep, edges\
                = getTurnHistCounts(vRotFly_filt, objDist, turnTH, turnTH_neg, turnTH_pos,
                                    numBins, (6, 56))

        binctrs = edges[:-1]+np.mean(np.diff(edges))/2.0

        if objecttype == 'visible':
            axApr.plot(binctrs, 1.0*nT_apr/nDt_apr, color=visAprCMap.to_rgba(0.5+trial % n_vist))
            axDep.plot(binctrs, 1.0*nT_dep/nDt_dep, color=visDepCMap.to_rgba(0.5+trial % n_vist))
        else:
            axApr.plot(binctrs, 1.0*nT_apr/nDt_apr, color=invAprCMap.to_rgba(0.5+trial % n_vist))
            axDep.plot(binctrs, 1.0*nT_dep/nDt_dep, color=invDepCMap.to_rgba(0.5+trial % n_vist))

        legendtext.append(trialName[trial])

    axApr.set_ylabel('normalised turn count')

    for ax in [axApr, axDep]:
        ax.set_xlabel('object distance [mm]')
        ax.set_ylim(ylimrange)
        ax.set_xlim(0, arenaRad)
        ax.legend(legendtext, fontsize=11)
        myAxisTheme(ax)

    Fig.tight_layout()

    return Fig


# Combined plots of velocity distributions + relative heading (sorted by mean walking speed) of multiple flies .........

def plotVeloHeadingDistribution_flyVR_df(mydataframe, trialtype, trial, flyIDs, keylist, vTransTH, minDist, maxDist):
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

    nhAll = np.zeros((18, numFlies))
    meanSpeeds = np.zeros(numFlies)

    flyIDsShort = []
    [flyIDsShort.append(flyIDs[fly][-3:]) for fly in range(numFlies)]

    for flyInd, flyID in enumerate(flyIDs):
        querystring = '(moving>0) & (trialtype == "'+trialtype+'") & (trial=='+trial+') & (flyID == "'+flyID+'")' +\
                      '& (objectDistance>' + str(minDist) + ') & (objectDistance<' + str(maxDist) + ')'
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
        h = mydataframe.query(querystring).iloc[:, keyind_h:keyind_h+1].squeeze()
        nh, binEdges = np.histogram(h, bins=18, range=(0, np.pi))
        if np.nansum(ntV) > 10*60*20*0.05:
            # np.nanmedian(tV) > 2.0:
            nhAll[:, flyInd] = nh
            meanSpeeds[flyInd] = np.nanmedian(tV)

    X, Y = np.meshgrid(range(0, numFlies+1), binEdges)
    sortedSpeeds = sorted(range(len(meanSpeeds)), key=lambda k: meanSpeeds[k])
    nhAll_sorted = np.zeros((len(nh), numFlies))
    for flyInd in range(len(sortedSpeeds)):
        nhAll_sorted[:, flyInd] = nhAll[:, sortedSpeeds[flyInd]]
    toPlot = nhAll_sorted/(np.tile(np.sum(nhAll_sorted, 0), (len(nh), 1)))
    toPlot[np.isnan(toPlot)] = 0
    hsubplt.pcolormesh(Y, X, toPlot, cmap='Greys')
    hsubplt.set_title(str(minDist) + 'mm < objectDistance < ' + str(maxDist) + ' mm ')
    hsubplt.yaxis.set_visible(False)
    for fly in range(numFlies):
        hsubplt.text(np.pi+0.1, sortedSpeeds[fly]+.5, flyIDsShort[fly], fontsize=12)

    vTsubplt.set_xlim((vTransTH, 30))
    vRsubplt.set_xlim(vRotRange)
    hsubplt.set_xlim((0, np.pi))

    veloDistFig.suptitle(trialtype+' object, trial '+trial+' (moving > '+str(vTransTH)+' mm/s)', fontsize=12)
    vTsubplt.legend(flyIDsShort, ncol=2, loc=1, fontsize=8)

    return veloDistFig


def plotVeloHeadingDistribution_freeWalk(flyIDs, vTransTH, vTransAll, vRotAll, objDistAll, gammaFullAll, flyIDAll,
                                         minDist, maxDist):
    """ Plot velocity and relative heading distributions (non-normalised) for a set of flies """

    numFlies = len(flyIDs)
    flyCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=numFlies), cmap='Accent')

    vRotRange = (-10, 10)
    angleRange = (-np.pi, np.pi)

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

    nhAll = np.zeros((18, numFlies))
    meanSpeeds = np.zeros(numFlies)

    flyIDsShort = []
    [flyIDsShort.append(flyIDs[fly][-3:]) for fly in range(numFlies)]

    selectMove = vTransAll > vTransTH
    selectDist = np.logical_and(objDistAll > minDist, objDistAll < maxDist)
    select = np.logical_and(selectMove, selectDist)

    for flyInd, flyID in enumerate(flyIDs):
        selectFly = np.logical_and(select, flyIDAll == flyIDs[flyInd])
        tV = vTransAll[selectFly]
        rV = vRotAll[selectFly]
        # v trans
        ntV, binEdges=np.histogram(tV, bins=50, range=(vTransTH, 60))
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        vTsubplt.plot(bincenters, ntV, alpha=0.7, color=flyCMap.to_rgba(flyInd))

        # v rot
        nrV, binEdges = np.histogram(rV, bins=50, range=vRotRange)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        vRsubplt.plot(bincenters, nrV, alpha=0.7, color=flyCMap.to_rgba(flyInd))

        # heading
        h = gammaFullAll[selectFly]
        nh, binEdges = np.histogram(h, bins=18, range=angleRange)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        nh, binEdges = np.histogram(h, bins=18, range=(0, np.pi))
        if np.nanmean(tV) > 2.0:
            nhAll[:, flyInd] = nh
            meanSpeeds[flyInd] = np.mean(tV)

    X, Y = np.meshgrid(range(0, numFlies+1), binEdges)
    sortedSpeeds = [i[0] for i in sorted(enumerate(meanSpeeds), key=lambda x:x[1])]
    nhAll_sorted = nhAll[:, sortedSpeeds]
    toPlot = nhAll_sorted/sum(nhAll_sorted)
    toPlot[np.isnan(toPlot)] = 0
    hsubplt.pcolormesh(Y, X, toPlot, cmap='Greys')
    hsubplt.set_title(str(minDist) + 'mm < objectDistance < ' + str(maxDist) + ' mm ')
    hsubplt.yaxis.set_visible(False)
    for fly in range(numFlies):
        hsubplt.text(np.pi+0.1, sortedSpeeds[fly]+.5, flyIDsShort[fly], fontsize=12)

    vTsubplt.set_xlim((vTransTH, 60))
    vRsubplt.set_xlim(vRotRange)
    hsubplt.set_xlim((0, np.pi))

    veloDistFig.suptitle('single object in blue light, (moving > '+str(vTransTH)+' mm/s)'+
                         str(minDist) + 'mm < objectDistance < '+str(maxDist)+' mm', fontsize=12)
    vTsubplt.legend(flyIDs, ncol=2, loc=1, fontsize=8)

    return veloDistFig


def plotVeloHeadingDistribution2_flyVR_df(mydataframe, traceframgent, trialtype, trial, flyIDs, keylist, vTransTH,
                                          minDist, maxDist):
    """ Plot velocity and relative heading distributions (non-normalised) for a set of flies """
    # traceframgent can be 'all', 'approach', 'departure'

    numFlies = len(flyIDs)
    flyCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=numFlies), cmap='Accent')

    vRotRange = (-10, 10)
    angleRange = (-np.pi, np.pi)

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
    keyind_h = keylist.index('gammaFull')
    keyind_vR = keylist.index('rotVelo')

    keyind_oD = keylist.index('objectDistance')

    flyIDsShort = []
    [flyIDsShort.append(flyIDs[fly][-3:]) for fly in range(numFlies)]

    for flyInd, flyID in enumerate(flyIDs):
        querystring = '(moving>0) & (trialtype == "'+trialtype+'") & (trial=='+trial+') & (flyID == "'+flyID+'")' + \
                      '& (objectDistance>' + str(minDist) + ') & (objectDistance<' + str(maxDist) + ')'
        tV = mydataframe.query(querystring).iloc[:, keyind_vT:keyind_vT+1].squeeze()
        rV = mydataframe.query(querystring).iloc[:, keyind_vR:keyind_vR+1].squeeze()
        oD = mydataframe.query(querystring).iloc[:, keyind_oD:keyind_oD+1].squeeze()
        oD_d = np.hstack((0, np.diff(oD)))

        if len(tV) == 0:
            continue

        if traceframgent == 'all':
            selecttrace = np.ones(len(oD_d), dtype=bool)
        elif traceframgent == 'approach':
            selecttrace = oD_d < 0
        else:
            selecttrace = oD_d >= 0

        # v trans
        ntV, binEdges=np.histogram(tV[selecttrace], bins=50, range=(vTransTH, 30))
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        vTsubplt.plot(bincenters, ntV, alpha=0.7, color=flyCMap.to_rgba(flyInd))

        # v rot
        nrV, binEdges = np.histogram(rV[selecttrace], bins=50, range=vRotRange)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        vRsubplt.plot(bincenters, nrV, alpha=0.7, color=flyCMap.to_rgba(flyInd))

        # heading
        querystring = '(moving>0)  & (objectDistance>' + str(minDist) + ') & (objectDistance<' + str(maxDist) + \
                      ') & (trialtype == "' + trialtype + '") & (trial==' + trial + \
                      ') & (flyID == "' + flyID + '")'
        h = mydataframe.query(querystring).iloc[:, keyind_h:keyind_h+1].squeeze()
        nh, binEdges = np.histogram(h[selecttrace], bins=18, range=angleRange)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        if np.nansum(ntV) > 10*60*20*0.05:
            # np.nanmean(tV[selecttrace]) > 2.0:
            hsubplt.plot(bincenters, nh, alpha=0.7, color=flyCMap.to_rgba(flyInd))

    hsubplt.set_title(str(minDist) + 'mm < objectDistance < ' + str(maxDist) + ' mm ')

    vTsubplt.set_xlim((vTransTH, 30))
    vRsubplt.set_xlim(vRotRange)
    hsubplt.set_xlim(angleRange)

    veloDistFig.suptitle(trialtype+' object, trial '+trial+' (moving > '+str(vTransTH)+' mm/s)', fontsize=12)
    vTsubplt.legend(flyIDsShort, ncol=2, loc=1, fontsize=8)

    return veloDistFig


def plotVeloHeadingDistribution2_freeWalk(flyIDs, vTransTH, vTransAll, vRotAll, objDistAll, gammaFullAll, flyIDAll,\
                                         minDist, maxDist):
    """ Plot velocity and relative heading distributions (non-normalised) for a set of flies """

    numFlies = len(flyIDs)
    flyCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=numFlies), cmap='Accent')

    vRotRange = (-10, 10)
    angleRange = (-np.pi, np.pi)

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

    selectMove = vTransAll > vTransTH
    selectDist = np.logical_and(objDistAll>minDist,objDistAll<maxDist)
    select = np.logical_and(selectMove,selectDist)

    for flyInd, flyID in enumerate(flyIDs):
        selectFly = np.logical_and(select,flyIDAll == flyIDs[flyInd])
        tV = vTransAll[selectFly]
        rV = vRotAll[selectFly]
        # v trans
        ntV, binEdges=np.histogram(tV, bins=50, range=(vTransTH, 60))
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        vTsubplt.plot(bincenters, ntV, alpha=0.7, color=flyCMap.to_rgba(flyInd))

        # v rot
        nrV, binEdges = np.histogram(rV, bins=50, range=vRotRange)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        vRsubplt.plot(bincenters, nrV, alpha=0.7, color=flyCMap.to_rgba(flyInd))

        # heading
        h = gammaFullAll[selectFly]
        nh, binEdges = np.histogram(h, bins=18, range=angleRange)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        if np.mean(tV) > 2.0:
            hsubplt.plot(bincenters, nh, alpha=0.7, color=flyCMap.to_rgba(flyInd))

    vTsubplt.set_xlim((vTransTH, 60))
    vRsubplt.set_xlim(vRotRange)
    hsubplt.set_xlim(angleRange)

    veloDistFig.suptitle('single object in blue light, (moving > '+str(vTransTH)+' mm/s)'+
                         str(minDist) + 'mm < objectDistance < '+str(maxDist)+' mm', fontsize=12)
    vTsubplt.legend(flyIDs, ncol=2, loc=1, fontsize=8)

    return veloDistFig

# Utilitiy funtions  ...................................................................................................

def cartesian2polar(xPosFly, yPosFly):
    raddist = np.hypot(xPosFly, yPosFly)
    theta = np.arctan2(yPosFly, xPosFly) + np.pi

    return raddist, theta