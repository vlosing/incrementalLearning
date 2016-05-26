__author__ = 'viktor'
import predCommon
import matplotlib.pyplot as plt
from Visualization import GLVQPlot
from matplotlib import cm
import numpy as np

def plotGPSValues(ax, fig, gpsStream, valueStream, maxVal, minVal, cmap, size):
    colors = valueStream - minVal
    colors /= maxVal
    cax = ax.scatter(gpsStream[:, 1], gpsStream[:, 0], c=colors, s=size, cmap=cmap, linewidth=0)
    ax.scatter(gpsStream[0, 1], gpsStream[0, 0], c='green', s=40)
    ax.scatter(gpsStream[-1, 1], gpsStream[-1, 0], c='black', s=40)

    cbar = fig.colorbar(cax)
    ticks = cbar.ax.get_yticks()
    cbar.set_ticks(ticks)
    newTicks = ticks * maxVal + minVal
    newTickLabels = []
    for tick in newTicks:
        newTickLabels.append('%.2f '% tick)
    cbar.set_ticklabels(newTickLabels)
    return cax

def plotJunctionInformation(ax, junctionStopLinesGPS, junctionTrafficLightsGPS, stoppingJunctionApproaches=None, junctionDirections=None):
    junctionsWithTrafficLights = junctionStopLinesGPS[junctionTrafficLightsGPS[:, 0] != -1]
    junctionsWithoutTrafficLights = junctionStopLinesGPS[junctionTrafficLightsGPS[:, 0] == -1]

    ax.scatter(junctionsWithTrafficLights[:, 1], junctionsWithTrafficLights[:, 0], c='magenta', s=80, marker='+')
    ax.scatter(junctionsWithoutTrafficLights[:, 1], junctionsWithoutTrafficLights[:, 0], c='black', s=80, marker='+')

    for i in range(len(junctionStopLinesGPS)):
        junctionStopLine = junctionStopLinesGPS[i]
        text = str(i
                   )
        if stoppingJunctionApproaches!= None:
            for junctionApproach in stoppingJunctionApproaches:
                if all(junctionApproach[0] == junctionStopLine):
                    if junctionApproach[2] == predCommon.STOP_LABEL:
                        text += 's'
                    break

        if junctionDirections != None:
            for junctionDirection in junctionDirections:
                if all(junctionDirection[0] == junctionStopLine):
                    text += predCommon.TURNING_LABELS_SYMBOLS[junctionDirection[2]]
                    break
        ax.text(junctionStopLine[1]-0.0002, junctionStopLine[0]+0.0005, text)

def plotStoppingVelProfilesPerJunction(stoppingJunctionApproaches, junctionGPS):
    approachesOrderedByJunction = []
    for stopLine in junctionGPS.stopLines:
        approaches = []
        for junctionApproach in stoppingJunctionApproaches:
            if all(junctionApproach[0] == stopLine):
                approaches.append([junctionApproach[1], junctionApproach[2]])
        approachesOrderedByJunction.append([stopLine, approaches])

    for junctionApproaches in approachesOrderedByJunction:
        fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
        junctionIdx = np.where(np.all(junctionGPS.stopLines == junctionApproaches[0], axis=1))[0][0]
        fig.suptitle('junction ' + str(junctionIdx))
        plt.tight_layout()
        for junctionApproach in junctionApproaches[1]:
            #timeValues = junctionApproach[0][:, 0] - np.min(junctionApproach[0][:,0])
            if junctionApproach[1] == predCommon.STOP_LABEL:
                color = 'r'
            else:
                color = 'b'
            ax.plot(junctionApproach[0][:, 0], junctionApproach[0][:, predCommon.VELOCITY_IDX], c=color)
        ax.set_xlabel('time (s) to intersection')
        ax.set_ylabel('velocity (m/s)')

def plotAllJunctionApproaches(junctionApproaches, singlePlotting=True):
    if singlePlotting:
        fig, ax1 = plt.subplots(1, 1, sharex=True)
        fig, ax2 = plt.subplots(1, 1, sharex=True)
        fig, ax3 = plt.subplots(1, 1, sharex=True)
        ax3.set_xlabel('time [s] to intersection')
    else:
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax1=ax[0]
        ax2=ax[0]
        ax3=ax[0]
    fig.suptitle('all junctions')
    plt.tight_layout()
    for junctionApproach in junctionApproaches:
        ax1.plot(junctionApproach[1][:, 0], junctionApproach[1][:, predCommon.VELOCITY_IDX], c=GLVQPlot.getDefColors()[junctionApproach[2]], label=predCommon.LABEL_NAMES[junctionApproach[2]])
        ax2.plot(junctionApproach[1][:, 0], junctionApproach[1][:, predCommon.ACCEL_IDX], c=GLVQPlot.getDefColors()[junctionApproach[2]], label=predCommon.LABEL_NAMES[junctionApproach[2]])
        ax3.plot(junctionApproach[1][:, 0], junctionApproach[1][:, predCommon.AVS_IDX], c=GLVQPlot.getDefColors()[junctionApproach[2]], label=predCommon.LABEL_NAMES[junctionApproach[2]])
    handles, labels = ax1.get_legend_handles_labels()
    labels, indices = np.unique(labels, return_index=True)
    handles = np.array(handles)
    ax1.legend(handles[indices], labels)


    ax1.set_ylabel('velocity (m/s)')
    ax2.set_ylabel('acceleration (m/s^2)')
    ax3.set_ylabel('AVS')
    if singlePlotting:
        ax1.set_xlabel('time [s] to intersection')
        ax2.set_xlabel('time [s] to intersection')
    ax3.set_xlabel('time [s] to intersection')


def plotStream(streamData, junctionGPS, stopIndices=None, stoppingJunctionApproaches=None, junctionDirections=None, maxVel=None, minVel=None):
    gpsStream = streamData[0][:, [predCommon.GPSLAT_IDX, predCommon.GPSLONG_IDX]]
    velStream = streamData[0][:, predCommon.VELOCITY_IDX]
    fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
    fig.suptitle(streamData[1])
    plt.tight_layout()

    if maxVel == None:
        maxVel = np.max(velStream)
    if minVel == None:
        minVel = np.min(velStream)

    plotGPSValues(ax, fig, gpsStream, velStream, maxVel, minVel, cmap=cm.get_cmap('jet'), size=1)
    plotJunctionInformation(ax, junctionGPS.stopLines, junctionGPS.trafficLights, stoppingJunctionApproaches=stoppingJunctionApproaches, junctionDirections=junctionDirections)
    if stopIndices != None:
        ax.scatter(gpsStream[stopIndices, 1], gpsStream[stopIndices, 0], c='black', s=20)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

def plotFitting(firstPoints, secondPoints, vec1, vec2, m1, m2, c1, c2):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(firstPoints[:,1], firstPoints[:,0], c='r', linewidths=0.1)
    ax.scatter(secondPoints[:,1], secondPoints[:,0], c='b', linewidths=0.1)
    ax.scatter(firstPoints[0,1], firstPoints[0,0], c='y', linewidths=0.1)
    ax.scatter(firstPoints[-1,1], firstPoints[-1,0], c='g', linewidths=0.1)
    ax.scatter(secondPoints[-1,1], secondPoints[-1,0], c='black', linewidths=0.1)

    plotX = np.append(firstPoints[0,0], firstPoints[-1,0])
    y = m1 * plotX + c1
    ax.plot(y, plotX, c='gray')
    plotX = np.append(secondPoints[0,0], secondPoints[-1,0])
    y = m2 * plotX + c2
    ax.plot(y, plotX, c='red')

def featureImportancesPlot(allFeatureImportances, predictionHorizons, featureNames):
    fig, ax = plt.subplots(1, 1)
    legendPlotList = []
    currentSum = 0
    for i in range(allFeatureImportances.shape[1]):
        p = ax.bar(predictionHorizons, allFeatureImportances[:, i], color=GLVQPlot.getDefColors()[i], bottom=currentSum)
        currentSum += allFeatureImportances[:, i]
        legendPlotList.append(p[0])
    ax.legend(legendPlotList, featureNames)
    fig.suptitle('Features importance')
    ax.set_xlabel('prediction time(s)')
    ax.set_ylabel('relative importance')
