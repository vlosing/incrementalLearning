__author__ = 'vlosing'

import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import sys
import os
from JunctionGPS import JunctionGPSInformation, JunctionLabelData, Route
import math
import copy

from Base import Paths
import predCommon
import predPlotting
import matplotlib.pyplot as plt
from ClassifierCommon.BaseClassifier import BaseClassifier

def haversineDistInMeter(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)


    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return (c * r)*1000.


def getMinMaxVelocity(streamsData):
    minVel = sys.maxint
    maxVel = 0
    for streamData in streamsData:
        velStream = streamData[0][:, predCommon.VELOCITY_IDX]
        minVal = np.min(velStream)
        if minVal < minVel:
            minVel = minVal
        maxVal = np.max(velStream)
        if maxVal > maxVel:
            maxVel = maxVal
    return minVel, maxVel

def determStopIndices(streamData, stopVelocity=1.38):
    stopIndices = np.where(streamData[0][:, predCommon.VELOCITY_IDX] < stopVelocity)[0]
    seperatedStopIndices = []
    lastIdx = 0
    for stopIdx in stopIndices:
        if stopIdx - lastIdx > 100:
            seperatedStopIndices.append(stopIdx)
        lastIdx = stopIdx
    return seperatedStopIndices

def getFittedLineVector(pointsToFit):
    x = pointsToFit[:, 0]
    y = pointsToFit[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    firstPointCalcy = m * x[0] + c
    lastPointCalcy = m *x[-1] + c
    vec = np.array([x[-1], lastPointCalcy]) - np.array([x[01], firstPointCalcy])
    vec /= np.linalg.norm(vec)
    return vec, m, c

def getStreamJunctionDirections(streamData, junctionGPS, maxJunctionDistanceInMeter=40, fittingDistanceInMeter=70, turnAngleThresh=35, minVelocity=5):
    gpsStream = streamData[0][:, [predCommon.GPSLAT_IDX, predCommon.GPSLONG_IDX]]
    junctionDirections = []
    i=0
    for stopLineGPS in junctionGPS.stopLines:

        minDistIdx, junctionDistance = findMinHarvDistIdx(gpsStream, stopLineGPS)
        if junctionDistance <= maxJunctionDistanceInMeter:
            distances = haversineDistancesInMeter(gpsStream[:minDistIdx+1, :], stopLineGPS)
            #fittingIndices1 = np.where((distances <= fittingDistanceInMeter) & (streamData[0][:minDistIdx+1, VELOCITY_IDX]>=minVelocity))[0]
            fittingIndices1 = np.where(distances <= fittingDistanceInMeter)[0]
            vec1, m1, c1 = getFittedLineVector(gpsStream[fittingIndices1, :])

            distances = haversineDistancesInMeter(gpsStream[minDistIdx:, :], stopLineGPS)
            #fittingIndices2 = np.where((distances <= fittingDistanceInMeter) & (streamData[0][minDistIdx:, VELOCITY_IDX]>=minVelocity))[0] + minDistIdx
            fittingIndices2 = np.where(distances <= fittingDistanceInMeter)[0] + minDistIdx
            vec2, m2, c2 = getFittedLineVector(gpsStream[fittingIndices2, :])

            angle= np.arccos(np.dot(vec1, vec2)) * 100
            turningLabel = predCommon.TURNING_LABELS[predCommon.STRAIGHT_LABEL_IDX]
            if np.abs(angle) > turnAngleThresh:
                if np.sign(np.cross(vec1, vec2)) == 1:
                    turningLabel = predCommon.TURNING_LABELS[predCommon.RIGHT_TURN_LABEL_IDX]
                else:
                    turningLabel = predCommon.TURNING_LABELS[predCommon.LEFT_TURN_LABEL_IDX]
            #print i, np.dot(vec1, vec2), angle, np.cross(vec1, vec2), turningLabel
            junctionDirections.append([stopLineGPS, minDistIdx, turningLabel])
            #plotFitting(gpsStream[fittingIndices1, :], gpsStream[fittingIndices2, :], vec1, vec2, m1, m2, c1, c2)
        i+=1
    return junctionDirections


def getStreamStoppingJunctionApproaches(streamData, seperatedStopIndices, predictionHorizon, junctionGPS, junctionLabelData, maxPreMappingDistanceInMeter=70, maxPostMappingDistanceInMeter=20):
    gpsStream = streamData[0][:, [predCommon.GPSLAT_IDX, predCommon.GPSLONG_IDX]]
    junctionApproaches = []
    for stopLineGPS in junctionGPS.stopLines:
        approachData = []
        minDistIdx, junctionDistance = findMinHarvDistIdx(gpsStream[seperatedStopIndices, :], stopLineGPS)
        junctionStopIdx = seperatedStopIndices[minDistIdx]

        closestIdx, closestDistance = findMinHarvDistIdx(gpsStream, stopLineGPS)
        stopping = predCommon.STOP_LABEL
        if closestIdx >= junctionStopIdx and junctionDistance > maxPreMappingDistanceInMeter or closestIdx < junctionStopIdx and junctionDistance > maxPostMappingDistanceInMeter:
            if closestDistance > min(maxPreMappingDistanceInMeter, maxPostMappingDistanceInMeter):
                stopping = -1
            else:
                junctionStopIdx = closestIdx
                stopping = predCommon.PASS_LABEL
        if stopping > -1:
            junctionStopPoint = gpsStream[junctionStopIdx, :]
            timeJunctionStop = streamData[0][junctionStopIdx, 0]
            targetStartTime = max(timeJunctionStop - predictionHorizon, 0)
            startIdx = np.argmin(np.abs(streamData[0][:junctionStopIdx, 0] - targetStartTime))
            for datum in streamData[0][startIdx:junctionStopIdx + 1, :]:
                stopDistance = haversineDistInMeter(junctionStopPoint[0], junctionStopPoint[1], datum[predCommon.GPSLAT_IDX], datum[predCommon.GPSLONG_IDX])
                avs = datum[predCommon.VELOCITY_IDX]**2 + 2 * stopDistance * datum[predCommon.ACCEL_IDX]

                approachData.append(np.append(datum, [avs, stopDistance]))
            approachData = np.array(approachData)
            junctionApproaches.append([stopLineGPS, approachData, stopping])
    for junctionApproach in junctionApproaches:
        stopTime = junctionApproach[1][-1][0]
        junctionApproach[1][:,0] = np.abs(junctionApproach[1][:,0] - stopTime)
    return junctionApproaches

def getFeaturesFromJunctionApproaches(junctionApproaches, featureColumns, timeStamp = None, maxDeltaTime=0.1):
    X = np.empty(shape=(0, len(featureColumns)))
    Y = np.array([])
    for junctionApproach in junctionApproaches:
        approachData = junctionApproach[1]
        if timeStamp != None:
            minIdx = np.argmin(np.abs(approachData[:,0] - timeStamp))
            foundTime = approachData[minIdx, 0]
            if np.abs(foundTime - timeStamp) <= maxDeltaTime:
                X = np.vstack([X, approachData[minIdx, featureColumns]])
                Y = np.append(Y, junctionApproach[2])
        else:
            X = np.vstack([X, approachData[:, featureColumns]])
            Y = np.append(Y, np.ones(shape=(len(approachData))) * junctionApproach[2])
    return X, Y

def getAllCSVFilePaths(directory):
    allFiles = []
    for fileName in os.listdir(directory):
        if fileName.endswith(".csv"):
            allFiles.append(os.path.join(directory, fileName))
    return allFiles

def getAllStreamData(route):
    filePaths = getAllCSVFilePaths(route.featuresDir)

    streamsData = []
    for filePath in filePaths:
        data = np.loadtxt(filePath, delimiter=',', skiprows=1)[:, 0:4]
        streamDate = os.path.basename(filePath).split('.')[0]
        streamsData.append([data, streamDate])
    return streamsData

def preprocessStreamData(streamsData):
    for i in range(len(streamsData)):
        streamsData[i] = smoothVelocity(streamsData[i])
        streamsData[i] = addAcceleration(streamsData[i])
        streamsData[i] = smoothAcceleartion(streamsData[i])
    return streamsData

def addAcceleration(streamData):
    velocitys1 = streamData[0][:-1,predCommon.VELOCITY_IDX]
    velocitys2 = streamData[0][1:,predCommon.VELOCITY_IDX]
    time1 = streamData[0][:-1,predCommon.TIME_IDX]
    time2 = streamData[0][1:,predCommon.TIME_IDX]
    acceleartion = np.append(0., (velocitys2 - velocitys1) / (time2 - time1))
    streamData[0] = np.hstack([streamData[0], np.atleast_2d(acceleartion).T])
    return streamData

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def smoothVelocity(streamData, window_size=10):
    smoothedVel = movingaverage(streamData[0][:,predCommon.VELOCITY_IDX], window_size)
    streamData[0][:,predCommon.VELOCITY_IDX] = smoothedVel/3.6
    return streamData

def smoothAcceleartion(streamData, window_size=10):
    smoothedAcc = movingaverage(streamData[0][:,predCommon.ACCEL_IDX], window_size)
    streamData[0][:,predCommon.ACCEL_IDX] = smoothedAcc
    return streamData

def getFutureDataToStreamData(streamData, targetDeltaTime, maxDeltaTime=0.1):
    inputData = []
    mappedFutureData = []
    for i in range(len(streamData)):
        timeStart = streamData[i][predCommon.TIME_IDX]
        targetTime = timeStart + targetDeltaTime
        minIdx = np.argmin(np.abs(streamData[:, predCommon.TIME_IDX] - targetTime))
        closestTime = streamData[minIdx, predCommon.TIME_IDX]
        if np.abs(closestTime - targetTime) <= maxDeltaTime:
            inputData.append(streamData[i])
            mappedFutureData.append(streamData[minIdx])
    inputData = np.array(inputData)
    mappedFutureData = np.array(mappedFutureData)
    return inputData, mappedFutureData

def getStreamVelocityMapping(streamData, predictionHorizon, minVelocity=0.5):
    streamData = streamData[streamData[:, predCommon.VELOCITY_IDX] >= minVelocity]
    inputData, mappedData = getFutureDataToStreamData(streamData, predictionHorizon)
    return inputData, mappedData

def getVelocityMappingDataGroupedByStream(streamsData, predictionHorizon):
    velocityMappingDataGroupedByStream = []
    for streamData in streamsData:
        inputData, mappedData = getStreamVelocityMapping(streamData[0], predictionHorizon)
        velocityMappingDataGroupedByStream.append([inputData, mappedData, streamData[1]])
    return velocityMappingDataGroupedByStream

def getVelocityMappingData(streamsData, predictionHorizon):
    allInputData = np.empty(shape=(0, initialColumnCount))
    allMappedData = np.empty(shape=(0, initialColumnCount))
    for streamData in streamsData:
        inputData, mappedData = getStreamVelocityMapping(streamData[0], predictionHorizon)
        allInputData = np.vstack([allInputData, inputData])
        allMappedData = np.vstack([allMappedData, mappedData])
    return allInputData, allMappedData

def getVelPredictionDataFromMappedData(inputData, mappedData):
    X = inputData[:, VEL_FEATURE_COLUMNS]
    Y = mappedData[:, predCommon.VELOCITY_IDX]
    return X, Y

def getStoppingJunctionApproachesGroupedByStream(streamsData, predictionHorizon, junctionGPS):
    junctionApproachesGroupedByStream = []
    for streamData in streamsData:
        print streamData[1]
        #junctionLabelData = JunctionLabelData(route, streamData[1])
        stopIndices = determStopIndices(streamData)
        junctionApproaches = getStreamStoppingJunctionApproaches(streamData, stopIndices, predictionHorizon, junctionGPS, None)
        junctionApproachesGroupedByStream.append([junctionApproaches, streamData[1]])
    return junctionApproachesGroupedByStream

def getStoppingJunctionApproaches(streamsData, predictionHorizon, junctionGPS):
    allJunctionApproaches = []
    for streamData in streamsData:
        #junctionLabelData = JunctionLabelData(route, streamData[1])
        stopIndices = determStopIndices(streamData)
        junctionApproaches = getStreamStoppingJunctionApproaches(streamData, stopIndices, predictionHorizon, junctionGPS, None)
        allJunctionApproaches = allJunctionApproaches + junctionApproaches
    return allJunctionApproaches

def seperateStreamsData(streamsData, trainProportion=0.5):
    streamsData = np.array(streamsData)
    streamsData = streamsData[np.random.permutation(len(streamsData))]
    return streamsData[:int(len(streamsData)*trainProportion)], streamsData[len(streamsData) / 2:]

def findMinDistIdx(gpsStream, target):
    target = np.atleast_2d(target)
    refData = np.ones(shape=(len(gpsStream), 1)) * target
    differences = np.linalg.norm(refData - gpsStream, axis=1)
    return np.argmin(differences)


def haversineDistancesInMeter(gpsStream, target):
    targets = target * np.ones(shape=(len(gpsStream), 1))
    distances2 = haversineDistInMeter(gpsStream[:, 0], gpsStream[:, 1], targets[:, 0], targets[:, 1])
    return np.array(distances2)

def findMinHarvDistIdx(gpsStream, target):
    distances = haversineDistancesInMeter(gpsStream, target)
    minIdx = np.argmin(distances)
    return minIdx, distances[minIdx]

def determCorrespondingStreamIndicesByMinQuadDistance(refStream, compareStreams, maxDistanceInMeter=25, minVelocity=0.5):
    allVelocityDifferences = []
    print 'refStreamName', refStream[1]
    refStream[0] = refStream[0][refStream[0][:, predCommon.VELOCITY_IDX] >= minVelocity]
    for i in range(len(compareStreams)):
        compStream = compareStreams[i]
        print 'compStreamName', compStream[1]
        compStream[0] = compStream[0][compStream[0][:, predCommon.VELOCITY_IDX] >= minVelocity]

        usedIndices = []
        assignedIndices = []
        distances = []
        for i in range(len(refStream[0][:, [predCommon.GPSLAT_IDX, predCommon.GPSLONG_IDX]])):
            refData = refStream[0][i, [predCommon.GPSLAT_IDX, predCommon.GPSLONG_IDX]]
            minIdx, minDistance = findMinHarvDistIdx(compStream[0][:, [predCommon.GPSLAT_IDX, predCommon.GPSLONG_IDX]], refData)
            if minDistance < maxDistanceInMeter:
                usedIndices.append(i)
                assignedIndices.append(minIdx)
                distances.append(minDistance)


        print 'mean spacediff', np.mean(distances)
        print 'median spacediff',np.median(distances)
        print 'max spacediff', np.max(distances)
        volictyDifferences = np.abs(refStream[0][usedIndices, :][:, predCommon.VELOCITY_IDX] - compStream[0][assignedIndices, predCommon.VELOCITY_IDX])
        allVelocityDifferences = np.append(allVelocityDifferences, volictyDifferences)
        #ax[1].hist(volictyDifferences, bins=20)

        print 'mean velDiff', np.mean(volictyDifferences)
        print 'med velDiff', np.median(volictyDifferences)
        print 'std velDiff', np.std(volictyDifferences)
    print 'totalMean', np.mean(allVelocityDifferences)
    print 'totalMed', np.median(allVelocityDifferences)
    print 'totalStd', np.std(allVelocityDifferences)

def getLabelDistributions(Y):
    distribution = []
    uniqueLabels = np.unique(Y)
    for label in uniqueLabels:
        numberOfSamples = len(np.where(Y == label)[0])
        distribution.append([label, numberOfSamples])
    return distribution

def velocityPrediction(trainStreamsData, testStreamsData, predictionHorizon, FEATURE_COLUMNS, randomState=None):
    logging.info('vel prediction')
    logging.info('%d seconds horizon' % predictionHorizon)
    logging.info('%s feature columns' % str(predCommon.FEATURE_COLUMN_NAMES[FEATURE_COLUMNS]))


    predictionHorizons = np.arange(predictionHorizon, 0, -3)
    meanAbsoluteErrors = []
    maxAbsoluteErrors = []
    allFeatureImportances = np.empty(shape=(0, len(FEATURE_COLUMNS)))


    for predHorizon in predictionHorizons:
        inputData, mappedData = getVelocityMappingData(trainStreamsData, predHorizon)
        trainX, trainY = getVelPredictionDataFromMappedData(inputData, mappedData)
        #logging.info('%d train samples' % len(trainY))

        inputData, mappedData = getVelocityMappingData(testStreamsData, predHorizon)
        testX, testY = getVelPredictionDataFromMappedData(inputData, mappedData)
        #logging.info('%d test samples' % len(testY))
        clf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=randomState)
        clf.fit(trainX, trainY)
        #print 'feature Importances', clf.feature_importances_
        allFeatureImportances = np.vstack([allFeatureImportances, clf.feature_importances_])
        #predTrain = clf.predict(trainX)
        predTest = clf.predict(testX)
        meanAbsoluteErrors.append(mean_absolute_error(testY, predTest))
        maxAbsoluteErrors.append(np.max(np.abs(testY - predTest)))
        '''print 'mae train', mean_absolute_error(trainY, predTrain),'m/s'
        print 'mae test', mean_absolute_error(testY, predTest),'m/s'
        print 'max error train', np.max(np.abs(trainY - predTrain)),'m/s'
        print 'max error test', np.max(np.abs(testY - predTest)),'m/s'''

    predPlotting.featureImportancesPlot(allFeatureImportances, predictionHorizons, predCommon.FEATURE_COLUMN_NAMES[FEATURE_COLUMNS])

    fig, ax = plt.subplots(1, 1)
    ax.plot(predictionHorizons, meanAbsoluteErrors)
    fig.suptitle('RMSE velocity-prediction')
    ax.set_xlabel('prediction time (s)')
    ax.set_ylabel('RMSE (m/s^2)')
    fig, ax = plt.subplots(1, 1)
    ax.plot(predictionHorizons, maxAbsoluteErrors)
    fig.suptitle('max absolut error velocity-prediction')
    ax.set_xlabel('prediction time(s)')
    ax.set_ylabel('max absolut error (m/s^2)')


    '''predTestStreams = []
    for streamData in testStreamsData:
        predTestStreams.append(clf.predict(streamData[0]))


    #plotLowVelocityPoints(streamsData, 'Low velocity', junctionGPS)
    plotVelocityProfiles(trainStreamsData, 'TrainStreams', junctionGPS)
    plotVelocityProfiles(testStreamsData, 'TestStreams', junctionGPS)
    plotLowVelocityPoints(trainStreamsData, 'Low velocity', junctionGPS)
    plotLowVelocityPoints(testStreamsData, 'Low velocity', junctionGPS)
    plotPredictionValues(testStreamsData, predTestStreams, 'predictions', junctionGPS)
    plotGroundTruthValues(testStreamsData, 'groundTruth', junctionGPS)
    plotPredictionDifferences(testStreamsData, predTestStreams, 'differences', junctionGPS)'''

def junctionPredictionAccuracyForDifferentTimeStamps(predictionHorizon, testJunctionApproaches, clf, featureColumns):
    timeStamps = np.arange(predictionHorizon, 0, -0.1)
    accuracies = []
    for timeStamp in timeStamps:
        testX, testY = getFeaturesFromJunctionApproaches(testJunctionApproaches, featureColumns, timeStamp=timeStamp)
        predTest = clf.predict(testX)
        accuracies.append(accuracy_score(testY, predTest))
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('accuracy of stopping prediction')
    ax.plot(timeStamps, accuracies)
    ax.set_xlabel('time to intersection (s)')
    ax.set_ylabel('accuracy')


def junctionPredictionAccuracyForDifferentPredictionHorizons(featureColumns, randomState=None):
    predictionHorizons = np.arange(10, 0, -1)
    allFeatureImportances = np.empty(shape=(0, len(featureColumns)))
    accuracies = []
    for predHorizon in predictionHorizons:
        #trainJunctionApproaches = getStoppingJunctionApproaches(trainStreamsData, predHorizon, junctionGPS)
        trainJunctionApproaches = getTurnStopPassApproaches(trainStreamsData, predHorizon, junctionGPS)
        trainX, trainY = getFeaturesFromJunctionApproaches(trainJunctionApproaches, featureColumns)

        clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=randomState)
        clf.fit(trainX, trainY)
        #print 'feature Importances', clf.feature_importances_

        #testJunctionApproaches = getStoppingJunctionApproaches(testStreamsData, predHorizon, junctionGPS)
        testJunctionApproaches = getTurnStopPassApproaches(testStreamsData, predHorizon, junctionGPS)

        testX, testY = getFeaturesFromJunctionApproaches(testJunctionApproaches, featureColumns)
        predTest = clf.predict(testX)
        accuracies.append(accuracy_score(testY, predTest))
        allFeatureImportances = np.vstack([allFeatureImportances, clf.feature_importances_])
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('accuracy for different prediction horizons')
    ax.plot(predictionHorizons, accuracies)

    predPlotting.featureImportancesPlot(allFeatureImportances, predictionHorizons, predCommon.FEATURE_COLUMN_NAMES[featureColumns])

def stoppingPrediction(trainStreamsData, testStreamsData, predictionHorizon, featureColumns, junctionGPS, randomState=None):
    logging.info('stopping prediction')
    logging.info('%d seconds horizon' % predictionHorizon)
    logging.info('%s feature columns' % str(predCommon.FEATURE_COLUMN_NAMES[featureColumns]))

    trainJunctionApproaches = getStoppingJunctionApproaches(trainStreamsData, predictionHorizon, junctionGPS)
    trainX, trainY = getFeaturesFromJunctionApproaches(trainJunctionApproaches, featureColumns)

    clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=randomState)
    clf.fit(trainX, trainY)
    print 'feature Importances', clf.feature_importances_

    testJunctionApproaches = getStoppingJunctionApproaches(testStreamsData, predictionHorizon, junctionGPS)
    testX, testY = getFeaturesFromJunctionApproaches(testJunctionApproaches, featureColumns)
    predTest = clf.predict(testX)
    predTrain = clf.predict(trainX)

    print 'distribution test', getLabelDistributions(testY)
    print 'distribution train', getLabelDistributions(trainY)
    print 'acc train', accuracy_score(trainY, predTrain)
    print 'acc test', accuracy_score(testY, predTest)

    BaseClassifier.getLabelAccuracy(predTest, testY)

    #stoppingPredictionAccuracyForDifferentTimeStamps(predictionHorizon, testJunctionApproaches, clf)
    #stoppingPredictionAccuracyForDifferentPredictionHorizons(randomState=randomState)

def turnStopPassPrediction(trainStreamsData, testStreamsData, predictionHorizon, featureColumns, junctionGPS, randomState=None):
    logging.info('turn stop pass prediction')
    logging.info('%d seconds horizon' % predictionHorizon)
    logging.info('%s feature columns' % str(predCommon.FEATURE_COLUMN_NAMES[featureColumns]))

    trainJunctionApproaches = getTurnStopPassApproaches(trainStreamsData, predictionHorizon, junctionGPS)
    trainX, trainY = getFeaturesFromJunctionApproaches(trainJunctionApproaches, featureColumns)

    clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=randomState)
    clf.fit(trainX, trainY)
    print 'feature Importances', clf.feature_importances_

    testJunctionApproaches = getTurnStopPassApproaches(testStreamsData, predictionHorizon, junctionGPS)
    testX, testY = getFeaturesFromJunctionApproaches(testJunctionApproaches, featureColumns)
    predTest = clf.predict(testX)
    predTrain = clf.predict(trainX)

    print 'distribution test', getLabelDistributions(testY)
    print 'distribution train', getLabelDistributions(trainY)
    print 'acc train', accuracy_score(trainY, predTrain)
    print 'acc test', accuracy_score(testY, predTest)

    BaseClassifier.getLabelAccuracy(predTest, testY)

    junctionPredictionAccuracyForDifferentTimeStamps(predictionHorizon, testJunctionApproaches, clf, featureColumns)
    junctionPredictionAccuracyForDifferentPredictionHorizons(featureColumns, randomState=randomState)

def getTurnStopPassApproaches(streamsData, predictionHorizon, junctionGPS):
    allJunctionApproaches = []
    for streamData in streamsData:
        allJunctionApproaches = allJunctionApproaches + getStreamTurnStopPassApproaches(streamData, predictionHorizon, junctionGPS)
    return allJunctionApproaches

def getStreamTurnStopPassApproaches(streamData, predictionHorizon, junctionGPS):
    stopIndices = determStopIndices(streamData)
    stoppingJunctionApproaches = getStreamStoppingJunctionApproaches(streamData, stopIndices, predictionHorizon, junctionGPS, None)
    junctionDirections = getStreamJunctionDirections(streamData, junctionGPS)

    stopPassTurnApproaches = []
    for junctionApproach in stoppingJunctionApproaches:
        for junctionDirection in junctionDirections:
            if np.all(junctionApproach[0] == junctionDirection[0]):
                if junctionApproach[2] == predCommon.STOP_LABEL:
                    stopPassTurnApproaches.append(junctionApproach)
                elif junctionApproach[2] == predCommon.PASS_LABEL:
                    '''if junctionDirection[2] == predCommon.TURNING_LABELS[predCommon.LEFT_TURN_LABEL_IDX] or junctionDirection[2] == predCommon.TURNING_LABELS[predCommon.RIGHT_TURN_LABEL_IDX]:
                        tmpApproach = copy.deepcopy(junctionApproach)
                        tmpApproach[2] = predCommon.TURNING_LABEL
                        stopPassTurnApproaches.append(tmpApproach)'''
                    if junctionDirection[2] == predCommon.TURNING_LABELS[predCommon.LEFT_TURN_LABEL_IDX]:
                        tmpApproach = copy.deepcopy(junctionApproach)
                        tmpApproach[2] = predCommon.TURNING_LEFT_LABEL
                        stopPassTurnApproaches.append(tmpApproach)
                    elif junctionDirection[2] == predCommon.TURNING_LABELS[predCommon.RIGHT_TURN_LABEL_IDX]:
                        tmpApproach = copy.deepcopy(junctionApproach)
                        tmpApproach[2] = predCommon.TURNING_RIGHT_LABEL
                        stopPassTurnApproaches.append(tmpApproach)
                    elif junctionDirection[2] == predCommon.TURNING_LABELS[predCommon.STRAIGHT_LABEL_IDX]:
                        stopPassTurnApproaches.append(junctionApproach)
                    else:
                        raise RuntimeError('unexpected direction ' + str(junctionDirection[2]))
                else:
                    raise RuntimeError('stopping label ' + str(junctionApproach[2]))
                break
    return stopPassTurnApproaches

def plotStoppingData(streamsData, predictionHorizon, junctionGPS):
    minVel, maxVel = getMinMaxVelocity(streamsData)
    allStoppingJunctionApproaches = []
    for streamData in streamsData:
        print streamData[1]
        stopIndices = determStopIndices(streamData)
        stoppingJunctionApproaches = getStreamStoppingJunctionApproaches(streamData, stopIndices, predictionHorizon, junctionGPS, None)
        allStoppingJunctionApproaches = allStoppingJunctionApproaches + stoppingJunctionApproaches
        #predPlotting.plotStream(streamData, junctionGPS, stopIndices=stopIndices, stoppingJunctionApproaches=stoppingJunctionApproaches, junctionDirections=junctionDirections, minVel=minVel, maxVel=maxVel)

    #stoppingJunctionApproaches = getStoppingJunctionApproaches(streamsData, predictionHorizon, junctionGPS)
    #predPlotting.plotStoppingVelProfilesPerJunction(stoppingJunctionApproaches, junctionGPS)
    predPlotting.plotAllJunctionApproaches(allStoppingJunctionApproaches)

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    randomState = 0
    np.random.seed(randomState)

    VEL_FEATURE_COLUMNS = [predCommon.GPSLAT_IDX, predCommon.GPSLONG_IDX, predCommon.VELOCITY_IDX, predCommon.ACCEL_IDX]
    STOP_FEATURE_COLUMNS = [predCommon.GPSLAT_IDX, predCommon.GPSLONG_IDX, predCommon.VELOCITY_IDX, predCommon.ACCEL_IDX, predCommon.AVS_IDX, predCommon.STOP_DISTANCE_IDX]
    TURN_STOP_PASS_FEATURE_COLUMNS = [predCommon.VELOCITY_IDX, predCommon.ACCEL_IDX, predCommon.AVS_IDX, predCommon.STOP_DISTANCE_IDX]
    #FEATURE_COLUMNS = [3]

    predictionHorizonVelocity = 30
    predictionHorizonStopping = 6
    route = Route('martina', 'fromHonda', Paths.trackAddictDir())

    junctionGPS = JunctionGPSInformation(route)
    streamsData = getAllStreamData(route)
    streamsData = preprocessStreamData(streamsData)
    initialColumnCount = streamsData[0][0].shape[1]

    velPredData = getVelocityMappingDataGroupedByStream(streamsData, predictionHorizonVelocity)

    trainStreamsData, testStreamsData = seperateStreamsData(streamsData, trainProportion=0.8)
    logging.info('%d streams' % len(streamsData))
    logging.info('%d training streams' % len(trainStreamsData))

    turnPassStopApproaches = getTurnStopPassApproaches(streamsData, predictionHorizonStopping, junctionGPS)
    predPlotting.plotAllJunctionApproaches(turnPassStopApproaches)

    #plotStoppingData(streamsData, predictionHorizonStopping, junctionGPS)
    if len(trainStreamsData) > 0:
        stoppingPrediction(trainStreamsData, testStreamsData, predictionHorizonStopping, STOP_FEATURE_COLUMNS, junctionGPS, randomState=randomState)
        velocityPrediction(trainStreamsData, testStreamsData, predictionHorizonVelocity, VEL_FEATURE_COLUMNS, randomState=randomState)
        turnStopPassPrediction(trainStreamsData, testStreamsData, predictionHorizonStopping, TURN_STOP_PASS_FEATURE_COLUMNS, junctionGPS, randomState=randomState)
    #predPlotting.plotStream(streamsData[0], junctionGPS, None)
    #determCorrespondingStreamIndicesByMinQuadDistance(streamsData[0], streamsData[1:])
    plt.show()















