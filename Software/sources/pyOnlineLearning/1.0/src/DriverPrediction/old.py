__author__ = 'vlosing'

import logging
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
import sys
import os
from JunctionGPS import JunctionGPSInformation, JunctionLabelData, Route
import math

import matplotlib.pyplot as plt
from matplotlib import cm


def plotAllData(gpsStreams, valueStreams, title, titles, junctionStopLinesGPS, junctionTrafficLightsGPS, size=1):
    totMin = sys.maxint
    totMax = 0
    for valueStream in valueStreams:
        minVal = np.min(valueStream)
        if minVal < totMin:
            totMin = minVal
        maxVal = np.max(valueStream)
        if maxVal > totMax:
            totMax = maxVal

    fig, ax = plt.subplots(1, len(valueStreams), sharey=True, sharex=True)
    fig.suptitle(title)
    plt.tight_layout()
    for i in range(len(valueStreams)):
        if len(valueStreams) > 1:
            tmpAx = ax[i]
        else:
            tmpAx = ax

        #tmpAx.hold(True)
        tmpAx.set_title(titles[i])
        cax = plotGPSValues(tmpAx, gpsStreams[i], valueStreams[i], totMax, totMin, cmap=cm.get_cmap('jet'), size=size)
        plotJunctionInformation(tmpAx, junctionStopLinesGPS, junctionTrafficLightsGPS)

    cbar = fig.colorbar(cax)
    ticks = cbar.ax.get_yticks()
    cbar.set_ticks(ticks)
    newTicks = ticks*totMax+totMin
    newTickLabels = []
    for tick in newTicks:
        newTickLabels.append(str(tick))
    cbar.set_ticklabels(newTickLabels)


def plotAllData2(lowVelGPSStreams, lowVelValueStreams, highVelGPSStreams, highVelValueStreams, title, titles, junctionStopLinesGPS, junctionTrafficLightsGPS, size=1):
    totMin = sys.maxint
    totMax = 0
    for valueStream in lowVelValueStreams:
        minVal = np.min(valueStream)
        if minVal < totMin:
            totMin = minVal
        maxVal = np.max(valueStream)
        if maxVal > totMax:
            totMax = maxVal

    fig, ax = plt.subplots(1, len(lowVelValueStreams), sharey=True, sharex=True)
    fig.suptitle(title)
    plt.tight_layout()
    for i in range(len(lowVelValueStreams)):
        if len(lowVelValueStreams) > 1:
            tmpAx = ax[i]
        else:
            tmpAx = ax

        tmpAx.set_title(titles[i])
        cax = plotGPSValues(tmpAx, lowVelGPSStreams[i], lowVelValueStreams[i], totMax, totMin, cmap=cm.get_cmap('jet'), size=size)
        tmpAx.scatter(highVelGPSStreams[i][:, 1], highVelGPSStreams[i][:, 0], c='gray', s=0.1, linewidth=0)
        plotJunctionInformation(tmpAx, junctionStopLinesGPS, junctionTrafficLightsGPS)

    cbar = fig.colorbar(cax)
    ticks = cbar.ax.get_yticks()
    cbar.set_ticks(ticks)
    newTicks = ticks*totMax+totMin
    newTickLabels = []
    for tick in newTicks:
        newTickLabels.append(str(tick))
    cbar.set_ticklabels(newTickLabels)

def plotVelocityProfiles(streamsData, title, junctionGPS):
    gpsStreamsData = []
    valueStreamsData = []
    titles = []
    for streamData in streamsData:
        gpsStreamsData.append(streamData[2][:, [GPSLAT_IDX, GPSLONG_IDX]])
        valueStreamsData.append(streamData[2][:, VELOCITY_IDX])
        titles.append(os.path.basename(streamData[3]))
    plotAllData(gpsStreamsData, valueStreamsData, title, titles, junctionGPS.stopLines, junctionGPS.trafficLights)

def plotLowVelocityPoints(streamsData, title, junctionGPS):
    lowVelGPSStreamsData = []
    lowVelValueStreamsData = []

    highVelGPSStreamsData = []
    highVelValueStreamsData = []

    titles = []
    threshold = 5
    for streamData in streamsData:
        gpsData = streamData[0][:, [GPSLAT_IDX, GPSLONG_IDX]]
        velocityValues = streamData[0][:, VELOCITY_IDX]
        lowVelIndices = np.where(velocityValues <= threshold)[0]
        highVelIndices = np.where(velocityValues > threshold)[0]

        lowVelGPSStreamsData.append(gpsData[lowVelIndices, :])
        lowVelValueStreamsData.append(velocityValues[lowVelIndices])

        highVelGPSStreamsData.append(gpsData[highVelIndices, :])
        highVelValueStreamsData.append(velocityValues[highVelIndices])

        titles.append(streamData[1])


    plotAllData2(lowVelGPSStreamsData, lowVelValueStreamsData, highVelGPSStreamsData, highVelValueStreamsData, title, titles, junctionGPS.stopLines, junctionGPS.trafficLights, size=10)

def plotPredictionDifferences(streamsData, streamsPred, title, junctionGPS):
    gpsStreamsData = []
    valueStreamsData = []
    titles = []
    for i in range(len(streamsData)):
        streamData = streamsData[i]
        streamPred = streamsPred[i]
        gpsStreamsData.append(streamData[2][:, [GPSLAT_IDX, GPSLONG_IDX]])
        valueStreamsData.append(np.abs(streamData[1] - streamPred))
        titles.append(os.path.basename(streamData[3]))
    plotAllData(gpsStreamsData, valueStreamsData, title, titles, junctionGPS.stopLines, junctionGPS.trafficLights)

def plotPredictionValues(streamsData, streamsPred, title, junctionGPS):
    gpsStreamsData = []
    valueStreamsData = []
    titles = []
    for i in range(len(streamsData)):
        streamData = streamsData[i]
        streamPred = streamsPred[i]
        gpsStreamsData.append(streamData[2][:, [GPSLAT_IDX, GPSLONG_IDX]])
        valueStreamsData.append(streamPred)
        titles.append(os.path.basename(streamData[3]))
    plotAllData(gpsStreamsData, valueStreamsData, title, titles, junctionGPS.stopLines, junctionGPS.trafficLights)

def plotGroundTruthValues(streamsData, title, junctionGPS):
    gpsStreamsData = []
    valueStreamsData = []
    titles = []
    for i in range(len(streamsData)):
        streamData = streamsData[i]
        gpsStreamsData.append(streamData[2][:, 1:3])
        valueStreamsData.append(streamData[1])
        titles.append(os.path.basename(streamData[3]))
    plotAllData(gpsStreamsData, valueStreamsData, title, titles, junctionGPS.stopLines, junctionGPS.trafficLights)