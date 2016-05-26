from LVQPY import DistanceMatrix

__author__ = 'viktor'
import logging
import numpy as np
from sklearn import cross_validation
from sklearn.cluster import KMeans
import math
from LVQCommon import LVQCommon
from DistanceMatrix import DistanceMatrix
import matplotlib.pyplot as plt
from Visualization import GLVQPlot
import time
from scipy.stats import binom
from sklearn.cross_validation import train_test_split

class InsertionStrategies(object):
    #fig, ax = plt.subplots(1, 4, figsize=(12, 6))
    samples = None
    samplesLabels = None
    prototypes = None
    prototypesLabels = None
    distmatrix = None

    @staticmethod
    def getCandidates(insertionStrategy, distMatrix, samples, labels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct, protoAdds):
        #numTries = int(self.windowSize * 0.2)
        numTries = 50
        if insertionStrategy is None:
            candidates = []
            candidatesLabels = []
        elif insertionStrategy == 'Closest':
            InsertionStrategies._insertionStrategyClosest(distMatrix)
        elif insertionStrategy == 'Cluster':
            InsertionStrategies._insertionStrategyClustering(distMatrix, 'kMeans', 2)
        elif insertionStrategy == 'Voronoi':
            InsertionStrategies.insertionStrategyVoronoi(distMatrix)
        elif insertionStrategy == 'samplingCost':
            candidates, candidatesLabels = InsertionStrategies.samplingCost(protoAdds, distMatrix, numTries,
                                                                            samples, labels,
                                                                            prototypes, prototypesLabels, insStatistics,
                                                                            samplingFct, activFct, logisticFactor, getDistanceFct)
        elif insertionStrategy == 'samplingAcc':
            candidates, candidatesLabels, dummy, dummy2 = InsertionStrategies.samplingAcc(protoAdds, distMatrix, numTries,
                                                                            samples, labels,
                                                                            prototypes, prototypesLabels, insStatistics,
                                                                            samplingFct, activFct, logisticFactor, getDistanceFct)

        elif insertionStrategy == 'samplingAccReg':
            candidates, candidatesLabels = InsertionStrategies.samplingAccReg2(protoAdds, distMatrix, numTries,
                                                                            samples, labels,
                                                                            prototypes, prototypesLabels, insStatistics,
                                                                            samplingFct, activFct, logisticFactor, getDistanceFct)

        elif insertionStrategy == 'samplingAccMax':
            candidates, candidatesLabels = InsertionStrategies.samplingAccMax(distMatrix, numTries,
                                                                            samples, labels,
                                                                            prototypes, prototypesLabels, insStatistics,
                                                                            samplingFct, activFct, logisticFactor, getDistanceFct)
        elif insertionStrategy == 'samplingAcc':
            candidates, candidatesLabels = InsertionStrategies.samplingAccMax(distMatrix, numTries,
                                                                            samples, labels,
                                                                            prototypes, prototypesLabels, insStatistics,
                                                                            samplingFct, activFct, logisticFactor, getDistanceFct)
        return candidates, candidatesLabels



    @staticmethod
    def _getRandomSamplingIndices(indices, amount):
        permutation = np.random.permutation(len(indices))
        return indices[permutation][:amount]

    @staticmethod
    def weighted_sample(weights, sample_size):
        """
        Returns a weighted sample without replacement. Note: weights.dtype should be float for speed, see: http://sociograph.blogspot.nl/2011/12/gotcha-with-numpys-searchsorted.html
        """
        totals = np.cumsum(weights)
        sample = []
        for i in xrange(sample_size):
            rnd = np.random.sample() * totals[-1]
            idx = np.searchsorted(totals, rnd ,'right')
            sample.append(idx)
            totals[idx:] -= weights[idx]
        return sample

    @staticmethod
    def _getRouletteSamplingIndices(amount, windowSize):
        weights = np.logspace(0, 1, windowSize + 1, base=windowSize+1)[1:]
        return InsertionStrategies.weighted_sample(weights, min(amount, windowSize))

    @staticmethod
    def _getRecentSamplingIndices(indices, amount):
        return indices[-amount:]

    @staticmethod
    def _getSamplingIndices(samplingFct, numTries, numSamples):
        if numSamples <= numTries:
            return np.arange(numSamples)
        elif samplingFct == 'random':
            return InsertionStrategies._getRandomSamplingIndices(np.arange(numSamples), numTries)
        elif samplingFct == 'roulette':
            return InsertionStrategies._getRouletteSamplingIndices(numTries, numSamples)
        else:
            return InsertionStrategies._getRecentSamplingIndices(np.arange(numSamples), numTries)

    @staticmethod
    def samplingAcc(protoAdds, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct):
        #figRef, subplotRef = GLVQPlot.plotAll(None, prototypes, prototypesLabels, samples=windowSamples, samplesLabels=windowSamplesLabels,
        #                                                colors=GLVQPlot.getDefColors(), title='before', plotBoundary=False, XRange=GLVQPlot.getDefXRange2(), YRange=GLVQPlot.getDefYRange2())
        #plt.show()

        distMatrixTmp = np.copy(distMatrix)
        candidates = np.empty(shape=(0, windowSamples.shape[1]))
        candidatesLabels = np.empty(shape=(0, 1))
        tmpPrototypesLabels = np.copy(prototypesLabels)
        tmpPrototypes = np.copy(prototypes)
        addedProtos = 0
        for i in range(protoAdds):
            insStatistics._lastTriedInsertionWindowPrototypeCount = len(np.unique(distMatrixTmp[:, 0]))
            insStatistics._totalTriedInsertionWindowPrototypeCount += insStatistics._lastTriedInsertionWindowPrototypeCount
            insStatistics._lastTriedInsertionWindowPrototypeDensity = insStatistics._lastTriedInsertionWindowPrototypeCount / float(len(windowSamplesLabels))
            insStatistics._totalTriedInsertionWindowPrototypeDensity += insStatistics._lastTriedInsertionWindowPrototypeDensity
            insStatistics._triedInsertionCount += 1
            bestCandidateIdx, bestDistMatrix, newAcc, initialAcc = InsertionStrategies.getCandidateMaxAccuracy(distMatrixTmp, windowSamples,
                                                                                    windowSamplesLabels, tmpPrototypes, tmpPrototypesLabels, numTries,
                                                                                    samplingFct, activFct, logisticFactor, getDistanceFct)
            if bestCandidateIdx is not None:
                insStatistics._lastTriedInsertionWindowDeltaCost = newAcc - initialAcc
                insStatistics._totalTriedInsertionWindowDeltaCost += insStatistics._lastTriedInsertionWindowDeltaCost

                if newAcc > (initialAcc):
                    insStatistics._lastInsertionWindowPrototypeCount = insStatistics._lastTriedInsertionWindowPrototypeCount
                    insStatistics._totalInsertionWindowPrototypeCount += insStatistics._lastInsertionWindowPrototypeCount
                    insStatistics._lastInsertionWindowPrototypeDensity = insStatistics._lastTriedInsertionWindowPrototypeDensity
                    insStatistics._totalInsertionWindowPrototypeDensity += insStatistics._lastInsertionWindowPrototypeDensity
                    insStatistics._lastInsertionWindowDeltaCost = insStatistics._lastTriedInsertionWindowDeltaCost
                    insStatistics._totalInsertionWindowDeltaCost += insStatistics._lastInsertionWindowDeltaCost
                    candidates = np.vstack([candidates, windowSamples[bestCandidateIdx]])
                    candidatesLabels = np.append(candidatesLabels, windowSamplesLabels[bestCandidateIdx])
                    tmpPrototypes = np.vstack([tmpPrototypes, windowSamples[bestCandidateIdx]])
                    tmpPrototypesLabels = np.append(tmpPrototypesLabels, windowSamplesLabels[bestCandidateIdx])
                    distMatrixTmp = bestDistMatrix
                    addedProtos += 1
                else:
                    1#logging.info('not found')
        return candidates, candidatesLabels

    @staticmethod
    def samplingAccMax(distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct):
        distMatrixTmp = np.copy(distMatrix)
        candidates = np.empty(shape=(0, windowSamples.shape[1]))
        candidatesLabels = np.empty(shape=(0, 1))
        tmpPrototypesLabels = np.copy(prototypesLabels)
        tmpPrototypes = np.copy(prototypes)
        addedProtos = 0
        while True:
            bestCandidateIdx, bestDistMatrix, newAcc, initialAcc = InsertionStrategies.getCandidateMaxAccuracy(distMatrixTmp, windowSamples,
                                                                                    windowSamplesLabels, tmpPrototypes, tmpPrototypesLabels, numTries,
                                                                                    samplingFct, activFct, logisticFactor, getDistanceFct)
            if bestCandidateIdx is not None:
                if newAcc > initialAcc:
                    candidates = np.vstack([candidates, windowSamples[bestCandidateIdx]])
                    candidatesLabels = np.append(candidatesLabels, windowSamplesLabels[bestCandidateIdx])
                    tmpPrototypes = np.vstack([tmpPrototypes, windowSamples[bestCandidateIdx]])
                    tmpPrototypesLabels = np.append(tmpPrototypesLabels, windowSamplesLabels[bestCandidateIdx])
                    distMatrixTmp = bestDistMatrix
                    addedProtos += 1
                else:
                    break
        return candidates, candidatesLabels

    @staticmethod
    def samplingCostReg(protoAdds, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct):
    #def getCandidateSamplingCost2(windowSamples, windowSamplesLabels, prototypes, prototypesLabels, numTries, samplingFct, activFct, logisticFactor, getDistanceFct):
        stopIndices = []
        for k in range(1):
            X_train, X_test, y_train, y_test = train_test_split(windowSamples, windowSamplesLabels, test_size=0.33)
            distMatrixTrain = DistanceMatrix.getDistanceMatrix(X_train, y_train, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct)
            distMatrixTest = DistanceMatrix.getDistanceMatrix(X_test, y_test, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct)
            trainAcc = [np.average(LVQCommon.getAvgCostValueByDistanceMatrix(distMatrixTrain))]
            testAcc = [np.average(LVQCommon.getAvgCostValueByDistanceMatrix(distMatrixTest))]

            samplingIndices = InsertionStrategies._getSamplingIndices(samplingFct, numTries, len(windowSamplesLabels))
            tmpPrototypes = np.copy(prototypes)
            tmpPrototypesLabels = np.copy(prototypesLabels)

            iteration = 0
            while iteration < protoAdds or testAcc[-1] < testAcc[-2]:
                bestCandidateIdx = None
                bestDistMatrix = None
                maxTrainAcc = 1
                for idx in samplingIndices:
                    protoCandidate = windowSamples[idx]
                    protoCandidateLabel = windowSamplesLabels[idx]
                    if not LVQCommon.doesProtoExist(protoCandidate, tmpPrototypes):
                        newDistMatrix, dummy = DistanceMatrix.addProtoToDistanceMatrix(distMatrixTrain, X_train, protoCandidate,
                                                                                       protoCandidateLabel, len(tmpPrototypesLabels),
                                                                                       tmpPrototypesLabels, activFct, logisticFactor, getDistanceFct)
                        newAcc = LVQCommon.getAvgCostValueByDistanceMatrix(newDistMatrix)
                        if newAcc < maxTrainAcc:
                            maxTrainAcc = newAcc
                            bestCandidateIdx = idx
                            bestDistMatrix = np.copy(newDistMatrix)
                distMatrixTrain = bestDistMatrix
                trainAcc.append(maxTrainAcc)
                distMatrixTest, dummy = DistanceMatrix.addProtoToDistanceMatrix(distMatrixTest, X_test, windowSamples[bestCandidateIdx],
                                                                                windowSamplesLabels[bestCandidateIdx], len(tmpPrototypesLabels),
                                                                                tmpPrototypesLabels, activFct, logisticFactor, getDistanceFct)
                testAcc.append(LVQCommon.getAvgCostValueByDistanceMatrix(distMatrixTest))
                tmpPrototypesLabels = np.append(tmpPrototypesLabels, windowSamplesLabels[bestCandidateIdx])
                tmpPrototypes = np.vstack([tmpPrototypes, windowSamples[bestCandidateIdx]])
                iteration += 1
            '''deltaAcc =  np.array(testAcc[1:]) - np.array(testAcc[:-1])
            minDeltaAcc = 1./len(windowSamples)
            remainingDelta = deltaAcc - minDeltaAcc

            posIndices = np.where(remainingDelta > 0)[0][::-1]
            stopIdx = 0
            for posIdx in posIndices:
                if np.sum(remainingDelta[:posIdx + 1]) > 0:
                    stopIdx = posIdx + 1
                    break'''


            #print trainAcc
            #print testAcc
            stopIdx = np.argmin(testAcc)
            stopIndices.append(stopIdx)


        protosToAdd = int(math.floor(np.mean(stopIndices)))
        if protosToAdd > 0:
            return InsertionStrategies.samplingCost(protosToAdd, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct)
        else:
            return [], []

    @staticmethod
    def samplingAccReg(protoAdds, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct):
        stopIndices = []
        for k in range(1):
            X_train, X_test, y_train, y_test = train_test_split(windowSamples, windowSamplesLabels, test_size=0.33)
            distMatrixTrain = DistanceMatrix.getDistanceMatrix(X_train, y_train, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct)
            distMatrixTest = DistanceMatrix.getDistanceMatrix(X_test, y_test, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct)
            trainAcc = [LVQCommon.getAccuracyByDistanceMatrix(distMatrixTrain)]
            testAcc = [LVQCommon.getAccuracyByDistanceMatrix(distMatrixTest)]

            samplingIndices = InsertionStrategies._getSamplingIndices(samplingFct, numTries, len(windowSamplesLabels))
            tmpPrototypes = np.copy(prototypes)
            tmpPrototypesLabels = np.copy(prototypesLabels)

            iteration = 0
            while iteration < protoAdds or testAcc[-1] > testAcc[-2]:
                bestCandidateIdx = None
                bestDistMatrix = None
                maxTrainAcc = 0
                for idx in samplingIndices:
                    protoCandidate = windowSamples[idx]
                    protoCandidateLabel = windowSamplesLabels[idx]
                    if not LVQCommon.doesProtoExist(protoCandidate, tmpPrototypes):
                        newDistMatrix, dummy = DistanceMatrix.addProtoToDistanceMatrix(distMatrixTrain, X_train, protoCandidate,
                                                                                       protoCandidateLabel, len(tmpPrototypesLabels),
                                                                                       tmpPrototypesLabels, activFct, logisticFactor, getDistanceFct)
                        newAcc = LVQCommon.getAccuracyByDistanceMatrix(newDistMatrix)
                        if newAcc > maxTrainAcc:
                            maxTrainAcc = newAcc
                            bestCandidateIdx = idx
                            bestDistMatrix = np.copy(newDistMatrix)
                distMatrixTrain = bestDistMatrix
                trainAcc.append(maxTrainAcc)
                distMatrixTest, dummy = DistanceMatrix.addProtoToDistanceMatrix(distMatrixTest, X_test, windowSamples[bestCandidateIdx],
                                                                                windowSamplesLabels[bestCandidateIdx], len(tmpPrototypesLabels),
                                                                                tmpPrototypesLabels, activFct, logisticFactor, getDistanceFct)
                testAcc.append(LVQCommon.getAccuracyByDistanceMatrix(distMatrixTest))
                tmpPrototypesLabels = np.append(tmpPrototypesLabels, windowSamplesLabels[bestCandidateIdx])
                tmpPrototypes = np.vstack([tmpPrototypes, windowSamples[bestCandidateIdx]])
                iteration += 1
            stopIdx = np.argmax(testAcc)
            stopIndices.append(stopIdx)

        protosToAdd = int(math.floor(np.mean(stopIndices)))
        if protosToAdd > 0:
            candidates, candidatesLabels, dumy, dummy2 = InsertionStrategies.samplingAcc(protosToAdd, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct)
            return candidates, candidatesLabels
        else:
            return [], []


    @staticmethod
    def samplingAccReg2(protoAdds, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct):
    #def getCandidateSamplingCost2(windowSamples, windowSamplesLabels, prototypes, prototypesLabels, numTries, samplingFct, activFct, logisticFactor, getDistanceFct):
        stopIndices = []
        X_train, X_test, y_train, y_test = train_test_split(windowSamples, windowSamplesLabels, test_size=0.33)


        distMatrixTrain = DistanceMatrix.getDistanceMatrix(X_train, y_train, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct)
        distMatrixTest = DistanceMatrix.getDistanceMatrix(X_test, y_test, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct)
        trainAcc = [np.average(LVQCommon.getAccuracyByDistanceMatrix(distMatrixTrain))]
        testAcc = [np.average(LVQCommon.getAccuracyByDistanceMatrix(distMatrixTest))]

        samplingIndices = InsertionStrategies._getSamplingIndices(samplingFct, numTries, len(windowSamplesLabels))
        tmpPrototypes = np.copy(prototypes)
        tmpPrototypesLabels = np.copy(prototypesLabels)
        candidates = np.empty(shape=(0, windowSamples.shape[1]))
        candidatesLabels = np.empty(shape=(0, 1))
        iteration = 0

        while iteration < protoAdds or testAcc[-1] > testAcc[-2]:
            bestCandidateIdx = None
            bestDistMatrix = None
            maxTrainAcc = 0
            for idx in samplingIndices:
                protoCandidate = windowSamples[idx]
                protoCandidateLabel = windowSamplesLabels[idx]
                if not LVQCommon.doesProtoExist(protoCandidate, tmpPrototypes):
                    newDistMatrix, dummy = DistanceMatrix.addProtoToDistanceMatrix(distMatrixTrain, X_train, protoCandidate,
                                                                   protoCandidateLabel, len(tmpPrototypesLabels),
                                                                   tmpPrototypesLabels, activFct, logisticFactor, getDistanceFct)
                    newAcc = LVQCommon.getAccuracyByDistanceMatrix(newDistMatrix)
                    if newAcc > maxTrainAcc:
                        maxTrainAcc = newAcc
                        bestCandidateIdx = idx
                        bestDistMatrix = np.copy(newDistMatrix)
            distMatrixTrain = bestDistMatrix
            trainAcc.append(maxTrainAcc)
            distMatrixTest, dummy = DistanceMatrix.addProtoToDistanceMatrix(distMatrixTest, X_test, windowSamples[bestCandidateIdx],
                                                                   windowSamplesLabels[bestCandidateIdx], len(tmpPrototypesLabels),
                                                                   tmpPrototypesLabels, activFct, logisticFactor, getDistanceFct)
            testAcc.append(LVQCommon.getAccuracyByDistanceMatrix(distMatrixTest))
            tmpPrototypes = np.vstack([tmpPrototypes, windowSamples[bestCandidateIdx]])
            tmpPrototypesLabels = np.append(tmpPrototypesLabels, windowSamplesLabels[bestCandidateIdx])

            candidates = np.vstack([candidates, windowSamples[bestCandidateIdx]])
            candidatesLabels = np.append(candidatesLabels, windowSamplesLabels[bestCandidateIdx])
            iteration += 1
        stopIdx = np.argmax(testAcc)
        candidates = candidates[:stopIdx, :]
        candidatesLabels = candidatesLabels[:stopIdx]
        return candidates, candidatesLabels


    @staticmethod
    def getCandidateSamplingCost(distMatrix, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, numTries, samplingFct, activFct, logisticFactor, getDistanceFct):
        minProtoCandidateIdx = None
        minDistMatrix = None
        initialAvgCost = np.average(LVQCommon.getCostFunctionValuesByMatrix(distMatrix))
        minAvgCost = np.finfo(np.float).max
        allDeltas = []
        samplingIndices = InsertionStrategies._getSamplingIndices(samplingFct, numTries, len(windowSamplesLabels))

        #print self.sampling, len(windowSamples), samplingIndices
        for idx in samplingIndices:
            protoCandidate = windowSamples[idx]
            protoCandidateLabel = windowSamplesLabels[idx]
            if not LVQCommon.doesProtoExist(protoCandidate, prototypes):
                newDistMatrix, dummy = DistanceMatrix.addProtoToDistanceMatrix(distMatrix, windowSamples, protoCandidate,
                                                                               protoCandidateLabel, len(prototypesLabels),
                                                                               prototypesLabels, activFct, logisticFactor, getDistanceFct)
                avgCost = np.average(LVQCommon.getCostFunctionValuesByMatrix(newDistMatrix))
                allDeltas.append(initialAvgCost - avgCost)
                if avgCost < minAvgCost:
                    minAvgCost = avgCost
                    minProtoCandidateIdx = idx
                    minDistMatrix = np.copy(newDistMatrix)
        return minProtoCandidateIdx, minDistMatrix, minAvgCost, initialAvgCost, allDeltas

    @staticmethod
    def getCandidateMaxAccuracy(distMatrix, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, numTries, samplingFct, activFct, logisticFactor, getDistanceFct):
        bestCandidateIdx = None
        bestDistMatrix = None
        initialAcc = LVQCommon.getAccuracyByDistanceMatrix(distMatrix)
        #initialAcc = LVQCommon.getWeightedAccuracyByDistanceMatrix(distMatrix)
        maxAcc = 0
        allDeltas = []
        samplingIndices = InsertionStrategies._getSamplingIndices(samplingFct, numTries, len(windowSamplesLabels))

        #print self.sampling, len(windowSamples), samplingIndices
        for idx in samplingIndices:
            protoCandidate = windowSamples[idx]
            protoCandidateLabel = windowSamplesLabels[idx]
            if not LVQCommon.doesProtoExist(protoCandidate, prototypes):
                newDistMatrix, dummy = DistanceMatrix.addProtoToDistanceMatrix(distMatrix, windowSamples, protoCandidate,
                                                                               protoCandidateLabel, len(prototypesLabels),
                                                                               prototypesLabels, activFct, logisticFactor, getDistanceFct)
                newAcc = LVQCommon.getAccuracyByDistanceMatrix(newDistMatrix)
                #newAcc = LVQCommon.getWeightedAccuracyByDistanceMatrix(newDistMatrix)
                if newAcc > maxAcc:
                    maxAcc = newAcc
                    bestCandidateIdx = idx
                    bestDistMatrix = np.copy(newDistMatrix)
        return bestCandidateIdx, bestDistMatrix, maxAcc, initialAcc


    @staticmethod
    def samplingAccBinom(protoAdds, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct):
        distMatrixTmp = np.copy(distMatrix)
        candidates = np.empty(shape=(0, windowSamples.shape[1]))
        candidatesLabels = np.empty(shape=(0, 1))
        tmpPrototypesLabels = np.copy(prototypesLabels)
        tmpPrototypes = np.copy(prototypes)
        for i in range(protoAdds):
            bestCandidateIdx, bestDistMatrix, newAcc, initialAcc = InsertionStrategies.getCandidateMaxAccuracy(distMatrixTmp, windowSamples,
                                                                                    windowSamplesLabels, tmpPrototypes, tmpPrototypesLabels, numTries,
                                                                                   samplingFct, activFct, logisticFactor, getDistanceFct)
            if bestCandidateIdx is not None:
                deltaAcc = binom.std(len(windowSamplesLabels), initialAcc)/len(windowSamplesLabels)
                print initialAcc, deltaAcc, newAcc
                if newAcc >= (initialAcc + deltaAcc):
                    candidates = np.vstack([candidates, windowSamples[bestCandidateIdx]])
                    candidatesLabels = np.append(candidatesLabels, windowSamplesLabels[bestCandidateIdx])
                    tmpPrototypes = np.vstack([tmpPrototypes, windowSamples[bestCandidateIdx]])
                    tmpPrototypesLabels = np.append(tmpPrototypesLabels, windowSamplesLabels[bestCandidateIdx])
                    distMatrixTmp = bestDistMatrix
                else:
                    logging.debug('delta %f too small' % (newAcc - initialAcc))
        return candidates, candidatesLabels

    @staticmethod
    def samplingMultipleSizes(protoAdds, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct):
        if len(windowSamplesLabels) > 100:
            stepSize = len(windowSamplesLabels) / 5
            initialAccs = []
            finalAccs = []
            for windowLength in np.arange(stepSize, len(windowSamplesLabels), stepSize):
                startIdx = (len(windowSamplesLabels) - windowLength)
                candidates, candidatesLabels, initialAcc, finalAcc = InsertionStrategies.samplingAcc(protoAdds, distMatrix[startIdx:], numTries, windowSamples[startIdx:, :], windowSamplesLabels[startIdx:], prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct)
                initialAccs.append(initialAcc)
                finalAccs.append(finalAcc)
            print 'in', initialAccs
            print 'af', finalAccs
        candidates, candidatesLabels, initialAcc, finalAcc = InsertionStrategies.samplingAcc(protoAdds, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct)
        return candidates, candidatesLabels

    @staticmethod
    def samplingAcc(protoAdds, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct):
        distMatrixTmp = np.copy(distMatrix)
        candidates = np.empty(shape=(0, windowSamples.shape[1]))
        candidatesLabels = np.empty(shape=(0, 1))
        tmpPrototypesLabels = np.copy(prototypesLabels)
        tmpPrototypes = np.copy(prototypes)
        for i in range(protoAdds):
            bestCandidateIdx, bestDistMatrix, newAcc, initialAcc = InsertionStrategies.getCandidateMaxAccuracy(distMatrixTmp, windowSamples,
                                                                                    windowSamplesLabels, tmpPrototypes, tmpPrototypesLabels, numTries,
                                                                                   samplingFct, activFct, logisticFactor, getDistanceFct)
            finalAcc = initialAcc
            if bestCandidateIdx is not None:

                if newAcc >= (initialAcc):
                    candidates = np.vstack([candidates, windowSamples[bestCandidateIdx]])
                    candidatesLabels = np.append(candidatesLabels, windowSamplesLabels[bestCandidateIdx])
                    tmpPrototypes = np.vstack([tmpPrototypes, windowSamples[bestCandidateIdx]])
                    tmpPrototypesLabels = np.append(tmpPrototypesLabels, windowSamplesLabels[bestCandidateIdx])
                    distMatrixTmp = bestDistMatrix
                    finalAcc = newAcc
                else:
                    logging.debug('delta %f too small' % (newAcc - initialAcc))
        return candidates, candidatesLabels, initialAcc, finalAcc

    #XXVL for performance: the distMatrix could be used directly, instead of recalculating at addPrototype
    @staticmethod
    def samplingCost(protoAdds, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct):
        #figRef, subplotRef = GLVQPlot.plotAll(None, prototypes, prototypesLabels, samples=windowSamples, samplesLabels=windowSamplesLabels,
        #                                                colors=GLVQPlot.getDefColors(), title='before', plotBoundary=False, XRange=GLVQPlot.getDefXRange2(), YRange=GLVQPlot.getDefYRange2())
        #plt.show()

        distMatrixTmp = np.copy(distMatrix)
        candidates = np.empty(shape=(0, windowSamples.shape[1]))
        candidatesLabels = np.empty(shape=(0, 1))
        tmpPrototypesLabels = np.copy(prototypesLabels)
        tmpPrototypes = np.copy(prototypes)
        addedProtos = 0
        windowCosts = []
        windowDeltaCosts = []
        for i in range(protoAdds):
            insStatistics._lastTriedInsertionWindowPrototypeCount = len(np.unique(distMatrixTmp[:, 0]))
            insStatistics._totalTriedInsertionWindowPrototypeCount += insStatistics._lastTriedInsertionWindowPrototypeCount
            insStatistics._lastTriedInsertionWindowPrototypeDensity = insStatistics._lastTriedInsertionWindowPrototypeCount / float(len(windowSamplesLabels))
            insStatistics._totalTriedInsertionWindowPrototypeDensity += insStatistics._lastTriedInsertionWindowPrototypeDensity
            insStatistics._triedInsertionCount += 1
            minProtoCandidateIdx, minDistMatrix, minAvgCost, initialAvgCost, allDeltas = InsertionStrategies.getCandidateSamplingCost(distMatrixTmp, windowSamples,
                                                                                    windowSamplesLabels, tmpPrototypes, tmpPrototypesLabels, numTries,
                                                                                    samplingFct, activFct, logisticFactor, getDistanceFct)
            if i == 0:
                windowCosts.append(initialAvgCost)
            if minProtoCandidateIdx is not None:
                insStatistics._lastTriedInsertionWindowDeltaCost = initialAvgCost - minAvgCost
                insStatistics._totalTriedInsertionWindowDeltaCost += insStatistics._lastTriedInsertionWindowDeltaCost

                #if minAvgCost <= (initialAvgCost):
                deltaNullRef = 0
                ownDelta = initialAvgCost - minAvgCost
                #print ownDelta, deltaNullRef
                if ownDelta > deltaNullRef:

                    windowCosts.append(minAvgCost)
                    windowDeltaCosts.append(initialAvgCost - minAvgCost)

                    insStatistics._lastInsertionWindowPrototypeCount = insStatistics._lastTriedInsertionWindowPrototypeCount
                    insStatistics._totalInsertionWindowPrototypeCount += insStatistics._lastInsertionWindowPrototypeCount
                    insStatistics._lastInsertionWindowPrototypeDensity = insStatistics._lastTriedInsertionWindowPrototypeDensity
                    insStatistics._totalInsertionWindowPrototypeDensity += insStatistics._lastInsertionWindowPrototypeDensity
                    insStatistics._lastInsertionWindowDeltaCost = insStatistics._lastTriedInsertionWindowDeltaCost
                    insStatistics._totalInsertionWindowDeltaCost += insStatistics._lastInsertionWindowDeltaCost
                    candidates = np.vstack([candidates, windowSamples[minProtoCandidateIdx]])
                    candidatesLabels = np.append(candidatesLabels, windowSamplesLabels[minProtoCandidateIdx])
                    tmpPrototypes = np.vstack([tmpPrototypes, windowSamples[minProtoCandidateIdx]])
                    tmpPrototypesLabels = np.append(tmpPrototypesLabels, windowSamplesLabels[minProtoCandidateIdx])
                    distMatrixTmp = minDistMatrix
                    addedProtos += 1
                else:
                    logging.debug('not found')

        return candidates, candidatesLabels

    @staticmethod
    def samplingCostGap(protoAdds, distMatrix, numTries, windowSamples, windowSamplesLabels, prototypes, prototypesLabels, insStatistics, samplingFct, activFct, logisticFactor, getDistanceFct):
        #figRef, subplotRef = GLVQPlot.plotAll(None, prototypes, prototypesLabels, samples=windowSamples, samplesLabels=windowSamplesLabels,
        #                                                colors=GLVQPlot.getDefColors(), title='before', plotBoundary=False, XRange=GLVQPlot.getDefXRange2(), YRange=GLVQPlot.getDefYRange2())
        #plt.show()

        distMatrixTmp = np.copy(distMatrix)
        candidates = np.empty(shape=(0, windowSamples.shape[1]))
        candidatesLabels = np.empty(shape=(0, 1))
        tmpPrototypesLabels = np.copy(prototypesLabels)
        tmpPrototypes = np.copy(prototypes)
        addedProtos = 0
        windowCosts = []
        windowDeltaCosts = []
        for i in range(protoAdds):
            insStatistics._lastTriedInsertionWindowPrototypeCount = len(np.unique(distMatrixTmp[:, 0]))
            insStatistics._totalTriedInsertionWindowPrototypeCount += insStatistics._lastTriedInsertionWindowPrototypeCount
            insStatistics._lastTriedInsertionWindowPrototypeDensity = insStatistics._lastTriedInsertionWindowPrototypeCount / float(len(windowSamplesLabels))
            insStatistics._totalTriedInsertionWindowPrototypeDensity += insStatistics._lastTriedInsertionWindowPrototypeDensity
            insStatistics._triedInsertionCount += 1
            minProtoCandidateIdx, minDistMatrix, minAvgCost, initialAvgCost, allDeltas = InsertionStrategies.getCandidateSamplingCost(distMatrixTmp, windowSamples,
                                                                                    windowSamplesLabels, tmpPrototypes, tmpPrototypesLabels, numTries,
                                                                                    samplingFct, activFct, logisticFactor, getDistanceFct)
            if i == 0:
                windowCosts.append(initialAvgCost)
            if minProtoCandidateIdx is not None:
                insStatistics._lastTriedInsertionWindowDeltaCost = initialAvgCost - minAvgCost
                insStatistics._totalTriedInsertionWindowDeltaCost += insStatistics._lastTriedInsertionWindowDeltaCost

                #if minAvgCost <= (initialAvgCost):
                deltaNullRef = 0
                ownDelta = initialAvgCost - minAvgCost
                #print ownDelta, deltaNullRef
                if ownDelta > deltaNullRef:

                    windowCosts.append(minAvgCost)
                    windowDeltaCosts.append(initialAvgCost - minAvgCost)

                    insStatistics._lastInsertionWindowPrototypeCount = insStatistics._lastTriedInsertionWindowPrototypeCount
                    insStatistics._totalInsertionWindowPrototypeCount += insStatistics._lastInsertionWindowPrototypeCount
                    insStatistics._lastInsertionWindowPrototypeDensity = insStatistics._lastTriedInsertionWindowPrototypeDensity
                    insStatistics._totalInsertionWindowPrototypeDensity += insStatistics._lastInsertionWindowPrototypeDensity
                    insStatistics._lastInsertionWindowDeltaCost = insStatistics._lastTriedInsertionWindowDeltaCost
                    insStatistics._totalInsertionWindowDeltaCost += insStatistics._lastInsertionWindowDeltaCost
                    candidates = np.vstack([candidates, windowSamples[minProtoCandidateIdx]])
                    candidatesLabels = np.append(candidatesLabels, windowSamplesLabels[minProtoCandidateIdx])
                    tmpPrototypes = np.vstack([tmpPrototypes, windowSamples[minProtoCandidateIdx]])
                    tmpPrototypesLabels = np.append(tmpPrototypesLabels, windowSamplesLabels[minProtoCandidateIdx])
                    distMatrixTmp = minDistMatrix
                    addedProtos += 1
                else:
                    logging.debug('not found')

        windowRefCostsAll = []
        windowRefDeltaCostsAll = []

        windowRefCostsAll2 = []
        windowRefDeltaCostsAll2 = []
        windowRefCosts = []
        windowRefDeltaCosts = []
        windowRefCosts2 = []
        windowRefDeltaCosts2 = []
        gap2 = []
        gapDifference2 = []

        windowPrototypes = prototypes[np.unique(np.append(distMatrix[:, 0], distMatrix[:, 2])).astype(int), :]
        windowPrototypesLabels = prototypesLabels[np.unique(np.append(distMatrix[:, 0], distMatrix[:, 2])).astype(int)]

        B = 3
        for i in range(B):
            windowRefCosts, windowRefDeltaCosts = InsertionStrategies._getCostFromNullReferencePermutated2(distMatrix, numTries, addedProtos,
                                                                                                        windowSamples, windowSamplesLabels, windowPrototypes, windowPrototypesLabels, samplingFct, activFct, logisticFactor, getDistanceFct)

            #windowRefCosts2, windowRefDeltaCosts2 = InsertionStrategies._getCostFromNullReferencePermutated(distMatrix, numTries, addedProtos,
            #                                                                                            windowSamples, windowSamplesLabels, windowPrototypes, windowPrototypesLabels, samplingFct, activFct, logisticFactor, getDistanceFct)

            windowRefCostsAll.append(windowRefCosts)
            windowRefDeltaCostsAll.append(windowRefDeltaCosts)
            windowRefCostsAll2.append(windowRefCosts2)
            windowRefDeltaCostsAll2.append(windowRefDeltaCosts2)

        gap = np.mean(windowRefCostsAll, axis=0) - np.array(windowCosts)
        gapStd = np.std(windowRefCostsAll, axis=0)
        #simulationErr = math.sqrt(1 + 1./B) * np.std(np.array(windowRefCostsAll)[:, 1:], axis=0)
        #simulationErr = math.sqrt(1./B) * np.std(np.array(windowRefCostsAll)[:, 1:], axis=0)
        simulationErr = 0

        gapDelta = gap[1:] - (gap[:-1] + simulationErr)
        gapDifference = gap[:-1] - (gap[1:] - simulationErr)

        #gap2 = np.mean(windowRefCostsAll2, axis=0) - np.array(windowCosts)
        #gap2Std = np.std(windowRefCostsAll2, axis=0)

        #simulationErr2 = math.sqrt(1 + 1./B) * np.std(np.array(windowRefCostsAll2)[:, 1:], axis=0)
        #simulationErr2 = 0
        #gapDifference2 = gap2[:-1] - (gap2[1:] - simulationErr2)
        #gapDelta2 = gap2[1:] - (gap2[:-1] + simulationErr2)

        #InsertionStrategies.plotCostCourse(windowCosts, windowRefCostsAll, gap, gapStd, gapDelta, gapDifference, windowRefCostsAll2, gap2, gap2Std, gapDelta2, gapDifference2, InsertionStrategies.ax[0], InsertionStrategies.ax[1], InsertionStrategies.ax[2], InsertionStrategies.ax[3])

        deltaSum = 0
        deltaMaxSum = 0
        stopIdx = 0
        for i, delta in zip(range(len(gapDelta)), gapDelta):
            deltaSum += delta
            if deltaSum > deltaMaxSum:
                deltaMaxSum = deltaSum
                stopIdx = i+1
        candidates = candidates[:stopIdx, :]
        candidatesLabels = candidatesLabels[:stopIdx]

        '''posIndices = np.where(gapDifference > 0)[0]
        if len(posIndices) > 0:
            if posIndices[0] == 0:
                stopIdx = 0
            else:
                stopIdx = posIndices[0] + 1
            candidates = candidates[:stopIdx, :]
            candidatesLabels = candidatesLabels[:stopIdx]'''
        return candidates, candidatesLabels

    '''
    #XXVL refactoring not finished!
    @staticmethod
    def _insertionStrategySamplingCostReg(distMatrix, windowSamples, windowSamplesLabels, numTries, protoAdds):
        distMatrixTmp = np.copy(distMatrix)
        for i in range(protoAdds):
            initialAvgCost = np.average(LVQCommon.getCostFunctionValuesByMatrix(distMatrixTmp))
            self.lastWindowProtoDensity = len(np.unique(distMatrixTmp[:, 0]))
            density = self.lastWindowProtoDensity / float(self.windowSize)
            regulParam = 0.001
            regulParam2 = 0.002
            if density < 1:
                self._deltaCost = regulParam + regulParam2 * (density / float(1 - density))
            else:
                self._deltaCost = 1
            self._totalDeltaCost += self._deltaCost
            self._windowProtoDensity += self.lastWindowProtoDensity
            self._triedInsertionCount += 1

            minAvgCost = initialAvgCost
            # print 'minCost', minAvgCost
            minProtoCandidateIdx = -1
            distMatrix = []
            #permutation = np.random.permutation(len(self.samplesLabels))
            for j in range(min(numTries, len(windowSamplesLabels))):
                #protoCandidateIdx = permutation[i]
                protoCandidateIdx = len(windowSamplesLabels) - 1 - j
                protoCandidate = windowSamples[protoCandidateIdx]
                protoCandidateLabel = windowSamplesLabels[protoCandidateIdx]
                if not self._doesProtoExist(protoCandidate):
                    newDistMatrix = self._addProtoToDistanceMatrix(distMatrixTmp, self._windowSamples, protoCandidate,
                                                                   protoCandidateLabel, len(self._prototypesLabels),
                                                                   self._prototypesLabels)
                    avgCost = np.average(LVQCommon.getCostFunctionValuesByMatrix(newDistMatrix))
                    #print 'avgCost', avgCost
                    if avgCost < minAvgCost:
                        minAvgCost = avgCost
                        minProtoCandidateIdx = protoCandidateIdx
                        distMatrix = np.copy(newDistMatrix)
            if minProtoCandidateIdx > -1 and minAvgCost <= (initialAvgCost - self._deltaCost):
                self._lastInsertionWindowDeltaCost = initialAvgCost - minAvgCost
                self._totalInsertionWindowDeltaCost += self._lastInsertionWindowDeltaCost

                self.addPrototype(self._windowSamples[minProtoCandidateIdx],
                                  self._windowSamplesLabels[minProtoCandidateIdx])
                distMatrixTmp = distMatrix
            else:
                logging.debug('not found')

    @staticmethod
    def _insertionStrategySamplingCostCV2(distMatrix, numTries):
        #print 'add'
        distMatrixTmp = np.copy(distMatrix)
        maxCandidates = 50
        votes = np.zeros(shape=(distMatrix.shape[0], 1))
        for i in range(self.protoAdds):
            self._triedInsertionCount += 1
            kf = cross_validation.KFold(n=distMatrix.shape[0], n_folds=3, shuffle=True)

            candidateEntryToInsert = None
            for train_index, test_index in kf:
                #print train_index, test_index
                trainDistMatrix = np.copy(distMatrixTmp[train_index, :])
                testDistMatrix = np.copy(distMatrixTmp[test_index, :])

                maxRelCostReduction = 1

                initialAvgCost = np.average(LVQCommon.getCostFunctionValuesByMatrix(trainDistMatrix))
                #print 'initialCost ', initialAvgCost
                #print initialAvgCost
                protoCandidates = []
                for j in range(min(numTries, len(self._windowSamples))):
                    protoCandidateIdx = len(self._windowSamples) - 1 - j
                    protoCandidate = self._windowSamples[protoCandidateIdx]
                    protoCandidateLabel = self._windowSamplesLabels[protoCandidateIdx]
                    if not self._doesProtoExist(protoCandidate):
                        newDistMatrix = self._addProtoToDistanceMatrix(trainDistMatrix,
                                                                       self._windowSamples[train_index, :],
                                                                       protoCandidate,
                                                                       protoCandidateLabel, len(self._prototypesLabels),
                                                                       self._prototypesLabels)
                        avgCost = np.average(LVQCommon.getCostFunctionValuesByMatrix(newDistMatrix))
                        #print 'avgCost', avgCost

                        if avgCost < initialAvgCost:
                            if len(protoCandidates) < maxCandidates:
                                protoCandidates.append(
                                    [avgCost, protoCandidate, protoCandidateLabel, protoCandidateIdx])
                                protoCandidates = sorted(protoCandidates, key=lambda proto: proto[0])
                            elif protoCandidates[-1][0] > avgCost:
                                protoCandidates.pop()
                                protoCandidates.append(
                                    [avgCost, protoCandidate, protoCandidateLabel, protoCandidateIdx])
                                protoCandidates = sorted(protoCandidates, key=lambda proto: proto[0])

                #print len(protoCandidates)
                if len(protoCandidates) > 0:
                    initialAvgCost = np.average(LVQCommon.getCostFunctionValuesByMatrix(testDistMatrix))
                    #print 'initialCostCV ', initialAvgCost
                    for candidateEntry in protoCandidates:
                        #print 'candidatecost ',candidateEntry[0]
                        newDistMatrix = self._addProtoToDistanceMatrix(testDistMatrix,
                                                                       self._windowSamples[test_index, :],
                                                                       candidateEntry[1], candidateEntry[2],
                                                                       len(self._prototypesLabels),
                                                                       self._prototypesLabels)
                        avgCost = np.average(LVQCommon.getCostFunctionValuesByMatrix(newDistMatrix))
                        relCostReduction = initialAvgCost / avgCost
                        #print 'candidateCost ', candidateEntry[0], avgCost, relCostReduction

                        if maxRelCostReduction < relCostReduction:
                            maxRelCostReduction = relCostReduction
                            candidateEntryToInsert = candidateEntry
                if not (candidateEntryToInsert is None):
                    votes[candidateEntryToInsert[3]] += 1

            if votes.max() > 0:
                protoIdx = votes.argmax()

                #print 'candidateEntryToInsert' , candidateEntryToInsert[0]
                #print 'insDelta ', self.lastInsertionDeltaCost
                initialAvgCost = np.average(LVQCommon.getCostFunctionValuesByMatrix(distMatrixTmp))
                distMatrixTmp = self._addProtoToDistanceMatrix(distMatrixTmp, self._windowSamples,
                                                               self._windowSamples[protoIdx, :],
                                                               self._windowSamplesLabels[protoIdx],
                                                               len(self._prototypesLabels), self._prototypesLabels)
                self._lastInsertionWindowDeltaCost = initialAvgCost - np.average(
                    LVQCommon.getCostFunctionValuesByMatrix(distMatrixTmp))
                self._totalInsertionWindowDeltaCost += self._lastInsertionWindowDeltaCost
                self.addPrototype(candidateEntryToInsert[1], candidateEntryToInsert[2])
            else:
                logging.debug('not found')

    @staticmethod
    def _insertionStrategyClosest(distMatrix):
        def findClosestErrorSampleForLabel(label, distMatrix, protoLabels, samples):
            errorIndices = \
                np.where((distMatrix[:, 3] <= distMatrix[:, 1]) &
                         (protoLabels[distMatrix[:, 0].astype(int)] == label))[0]
            if len(errorIndices) > 0:
                sampleIdx = np.argmin(distMatrix[errorIndices, 3])
                return samples[errorIndices[sampleIdx]], len(errorIndices), label
            else:
                return None

        protoList = []
        for i in np.unique(self._prototypesLabels):
            newProto = findClosestErrorSampleForLabel(i, distMatrix, self._prototypesLabels, self._windowSamples)
            if not (newProto is None):
                protoList.append(newProto)
        protoList = sorted(protoList, key=lambda proto: proto[1], reverse=True)
        protoCandidates = np.empty(shape=(0, self.dataDimensions))
        candidateLabels = np.empty(shape=(0, 1))
        for i in range(len(protoList)):
            protoCandidates = np.vstack([protoCandidates, protoList[i][0]])
            candidateLabels = np.append(candidateLabels, protoList[i][2])
        self._insertProtos(protoCandidates, candidateLabels)

    @staticmethod
    def getVoronoiCandidates(distMatrix):
        classClusters = []
        for i in np.unique(self._prototypesLabels):
            protoIdx, sampleIndices = self.getProtoypeIdxWithMostWrongSamplesForClass(distMatrix, i)
            if len(sampleIndices) > 0:
                classClusters.append((self._windowSamples[sampleIndices], i, len(sampleIndices)))
        classClusters = sorted(classClusters, key=lambda cluster: cluster[2], reverse=True)
        return classClusters

    @staticmethod
    def insertionStrategyVoronoi(distMatrix):
        candidates = self.getVoronoiCandidates(distMatrix)
        protoCandidates = np.empty(shape=(0, self.dataDimensions))
        candidateLabels = np.empty(shape=(0, 1))
        for i in range(len(candidates)):
            protoCandidates = np.vstack([protoCandidates, np.average(candidates[i][0], 0)])
            candidateLabels = np.append(candidateLabels, candidates[i][1])
        self._insertProtos(protoCandidates, candidateLabels)

    @staticmethod
    def _findBiggestClusterKMeans(samples, factor):
        km = KMeans(n_clusters=max(min(int(math.sqrt(len(samples) / 2.0) * factor), len(samples)), 1), n_init=1)
        km.fit(samples)
        clusterMapping = km.labels_
        counts = np.bincount(clusterMapping)
        return samples[np.where(clusterMapping == np.argmax(counts))]

    @staticmethod
    def _findBiggestClassCluster(samples, sampleLabels, clusterMethod, factor=1):
        result = []
        for i in np.unique(self._prototypesLabels):
            classIndices = np.where(sampleLabels == i)
            if len(classIndices[0]) >= 1:  # mindestens 4
                if clusterMethod == 'kMeans':
                    clusterSamples = _findBiggestClusterKMeans(samples[classIndices], factor)
                # if len(clusterSamples) > self.minClusterSize:
                result.append([clusterSamples, i, len(clusterSamples)])
        result = sorted(result, key=lambda cluster: cluster[2], reverse=True)
        return result

    @staticmethod
    def _getClusteringCandidates(distMatrix, clusterMethod, factor):
        costValues = distMatrix[:, 4]
        wrongIndices = np.where(costValues <= 0)
        wrongSamples = self._windowSamples[wrongIndices]
        wrongSamplesLabels = self._windowSamplesLabels[wrongIndices]
        return self._findBiggestClassCluster(wrongSamples, wrongSamplesLabels, clusterMethod, factor)

    @staticmethod
    def _insertionStrategyClustering(distMatrix, clusterMethod, factor=1):
        candidates = self._getClusteringCandidates(distMatrix, clusterMethod, factor)

        protoCandidates = np.empty(shape=(0, self.dataDimensions))
        candidateLabels = np.empty(shape=(0, 1))
        for i in range(len(candidates)):
            protoCandidates = np.vstack([protoCandidates, np.average(candidates[i][0], 0)])
            candidateLabels = np.append(candidateLabels, candidates[i][1])
        self._insertProtos(protoCandidates, candidateLabels) '''

    '''
    #Gap Statistics experiments
    @staticmethod
    def plotCostCourse(windowCosts, windowRefCosts, gap, gapStd, gapDelta, gapDifference, windowRefCosts2, gap2, gap2Std, gapDelta2, gapDifference2, ax, ax2, ax3, ax4):
        ax.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax.plot(np.arange(len(windowCosts)), windowCosts, c='r')

        windowRefCostsMean = np.mean(windowRefCosts, axis=0)
        windowRefCostsStd = np.std(windowRefCosts, axis=0)

        #ax.fill_between(np.arange(len(windowRefCostsMean)), np.array(windowRefCostsMean)-np.array(windowRefCostsStd), np.array(windowRefCostsMean) + np.array(windowRefCostsStd), color='gray')
        ax.plot(np.arange(len(windowRefCostsMean)), windowRefCostsMean, c='b')

        windowRefCostsMean2 = np.mean(windowRefCosts2, axis=0)
        windowRefCostsStd2 = np.std(windowRefCosts2, axis=0)

        #ax.fill_between(np.arange(len(windowRefCostsMean2)), np.array(windowRefCostsMean2)-np.array(windowRefCostsStd2), np.array(windowRefCostsMean2) + np.array(windowRefCostsStd2), color='yellow')
        ax.plot(np.arange(len(windowRefCostsMean2)), windowRefCostsMean2, c='g')

        #ax.plot(np.arange(len(windowRefCostsMean)), windowRefCostsMean - np.array(windowCosts), c='black')

        #ax.plot(np.arange(len(windowMeanDeltaCosts)), windowMeanDeltaCosts, c='g')
        #ax.plot(np.arange(len(windowRefDeltaCostsMean)), np.array(self.windowDeltaCosts) - windowRefDeltaCostsMean, c='black')

        ax2.fill_between(np.arange(len(gap)), np.array(gap) - np.array(gapStd), np.array(gap) + np.array(gapStd), color='gray')
        ax2.fill_between(np.arange(len(gap2)), np.array(gap2) - np.array(gap2Std), np.array(gap2) + np.array(gap2Std), color='yellow')
        ax2.plot(np.arange(len(gap)), gap, c='b')
        ax2.plot(np.arange(len(gap2)), gap2, c='g')
        ax2.set_ylim([-0.1, 0.9])

        ax3.plot(np.arange(len(gapDelta)), gapDelta, c='b')
        ax3.plot(np.arange(len(gapDelta2)), gapDelta2, c='g')
        ax3.set_ylim([-0.1, 0.15])

        ax4.plot(np.arange(len(gapDifference)), gapDifference, c='b')
        ax4.plot(np.arange(len(gapDifference2)), gapDifference2, c='g')

        plt.show()
        InsertionStrategies.fig.canvas.draw()

    @staticmethod
    def generateUniformDistribution(dimMinValues, dimMaxValues, numSamples):
        samples = np.empty(shape=(numSamples, len(dimMinValues)))

        for i, minValue, maxValue in zip(range(len(dimMinValues)), dimMinValues, dimMaxValues):
            samples[:, i] = np.random.uniform(minValue, maxValue, numSamples)
        return samples

    @staticmethod
    def getMajorityVoteLabels(protoIndicesLables, sampleLabels, orgProtoLabels):
        prototypeLabels = orgProtoLabels.copy()
        protoIndicesLables = protoIndicesLables.astype(int)
        protoIndices, indices = np.unique(protoIndicesLables[:, 0], return_inverse=True)
        correctionList = []
        for i in range(len(protoIndices)):
            closestSampleIndices = np.where(indices == i)[0]
            closestSampleLabels = protoIndicesLables[closestSampleIndices, 1]
            counts = np.bincount(closestSampleLabels)
            majorityVoteLabel = np.argmax(counts)
            prototypeLabels[protoIndices[i]] = majorityVoteLabel
            correctionList.append(np.append([majorityVoteLabel, protoIndices[i]], np.bincount(closestSampleLabels, minlength=np.max(sampleLabels)+1)))
        correctionList = np.array(correctionList)
        for label in np.unique(sampleLabels):
            labelPrototypeIndices = np.where(correctionList[:,0] == label)[0]
            if len(labelPrototypeIndices) == 0:
                counts = np.bincount(correctionList[:,0])
                replacedLabel = np.argmax(counts)
                replacedLabelIndices = np.where(correctionList[:,0]==replacedLabel)[0]
                replacedIdx = np.argmin(correctionList[replacedLabelIndices,replacedLabel+2] - correctionList[replacedLabelIndices, label+2])
                correctionList[replacedLabelIndices[replacedIdx], 0] = label
                prototypeLabels[correctionList[replacedLabelIndices[replacedIdx], 1]] = label
        return prototypeLabels

    @staticmethod
    def _getCostFromNullReferenceUniform2(distMatrix, numTries, addProtos, windowSamples, windowSamplesLabels, windowPrototypes, windowPrototypesLabels, samplingFct, activFct, logisticFactor, getDistanceFct):
        dimMinValues = np.min(windowSamples, axis=0)
        dimMaxValues = np.max(windowSamples, axis=0)
        uniSampleLables = windowSamplesLabels.copy()
        uniSamples = InsertionStrategies.generateUniformDistribution(dimMinValues, dimMaxValues, len(uniSampleLables))
        #uniPrototypes = InsertionStrategies.generateUniformDistribution(dimMinValues, dimMaxValues, len(windowPrototypesLabels))
        #uniPrototypesLabels = windowPrototypesLabels.copy()
        #distMatrixTmp = DistanceMatrix.getDistanceMatrix(uniSamples, uniSampleLables, uniPrototypes, tmpPrototypeLabels, activFct, logisticFactor, getDistanceFct)
        uniPrototypes = np.empty(shape=(0, uniSamples.shape[1]))
        uniPrototypesLabels =np.empty(shape=(0, 1), dtype=int)
        windowRefCosts = []
        windowRefDeltaCosts = []

        for label in np.unique(uniSampleLables):
            protoCandidates = uniSamples[uniSampleLables == label]
            #shuffledIndices = np.random.permutation(len(protoCandidates))
            #protoCandidates = protoCandidates[shuffledIndices]
            km = KMeans(n_clusters=1, n_init=1)
            km.fit(protoCandidates)
            protoCandidates = km.cluster_centers_
            uniPrototypes = np.vstack([uniPrototypes, protoCandidates[0, :]])
            uniPrototypesLabels = np.append(uniPrototypesLabels, label)

        distMatrixTmp = DistanceMatrix.getDistanceMatrix(uniSamples, uniSampleLables, uniPrototypes, uniPrototypesLabels, activFct, logisticFactor, getDistanceFct)

        protosToAdd = len(windowPrototypesLabels) - len(np.unique(uniSampleLables))
        for i in range(protosToAdd):
            minProtoCandidateIdx, minDistMatrix, minAvgCost, initialAvgCost, dummy = InsertionStrategies.getCandidateSamplingCost(distMatrixTmp, uniSamples, uniSampleLables, uniPrototypes, uniPrototypesLabels, numTries, samplingFct, activFct, logisticFactor, getDistanceFct)
            uniPrototypes = np.vstack([uniPrototypes, uniSamples[minProtoCandidateIdx, :]])
            uniPrototypesLabels = np.append(uniPrototypesLabels, uniSampleLables[minProtoCandidateIdx])
            distMatrixTmp = minDistMatrix

        windowRefCosts.append(np.average(LVQCommon.getCostFunctionValuesByMatrix(distMatrixTmp)))
        for i in range(addProtos):
            minProtoCandidateIdx, minDistMatrix, minAvgCost, initialAvgCost, dummy = InsertionStrategies.getCandidateSamplingCost(distMatrixTmp, uniSamples, uniSampleLables, uniPrototypes, uniPrototypesLabels, numTries, samplingFct, activFct, logisticFactor, getDistanceFct)
            uniPrototypes = np.vstack([uniPrototypes, uniSamples[minProtoCandidateIdx, :]])
            uniPrototypesLabels = np.append(uniPrototypesLabels, uniSampleLables[minProtoCandidateIdx])
            distMatrixTmp = minDistMatrix
            windowRefCosts.append(minAvgCost)
            windowRefDeltaCosts.append(initialAvgCost - minAvgCost)
        return windowRefCosts, windowRefDeltaCosts

    @staticmethod
    def _getCostFromNullReferenceUniform(distMatrix, numTries, addProtos, windowSamples, windowSamplesLabels, windowPrototypes, windowPrototypesLabels, samplingFct, activFct, logisticFactor, getDistanceFct):
        dimMinValues = np.min(windowSamples, axis=0)
        dimMaxValues = np.max(windowSamples, axis=0)
        uniSampleLables = np.random.randint(low=0, high=len(np.unique(windowSamplesLabels)), size=len(windowSamplesLabels))
        uniSamples = InsertionStrategies.generateUniformDistribution(dimMinValues, dimMaxValues, len(uniSampleLables))
        uniPrototypes = InsertionStrategies.generateUniformDistribution(dimMinValues, dimMaxValues, len(windowPrototypesLabels))
        uniPrototypesLabels = windowPrototypesLabels.copy()
        #distMatrixTmp = DistanceMatrix.getDistanceMatrix(uniSamples, uniSampleLables, uniPrototypes, tmpPrototypeLabels, activFct, logisticFactor, getDistanceFct)

        protoIndicesLables = np.empty(shape=(0, 2), dtype=int)

        for sample, i in zip(uniSamples, range(len(uniSamples))):
            tmpSample = sample.copy()
            tmpSample.shape = [len(sample), 1]
            sampleMat = tmpSample * np.ones(shape=[1, len(uniPrototypes)])
            distances = getDistanceFct(np.transpose(uniPrototypes), sampleMat)
            closestPrototypeIdx = np.argmin(distances)
            protoIndicesLables = np.vstack([protoIndicesLables, np.hstack([np.atleast_2d(closestPrototypeIdx).T, np.atleast_2d(uniSampleLables[i]).T])])

        uniPrototypesLabels = InsertionStrategies.getMajorityVoteLabels(protoIndicesLables, uniSampleLables, uniPrototypesLabels)
        distMatrixTmp = DistanceMatrix.getDistanceMatrix(uniSamples, uniSampleLables, uniPrototypes, uniPrototypesLabels, activFct, logisticFactor, getDistanceFct)

        windowRefCosts = []
        windowRefDeltaCosts = []

        windowRefCosts.append(np.average(LVQCommon.getCostFunctionValuesByMatrix(distMatrixTmp)))
        #print 'cost', np.average(LVQCommon.getCostFunctionValuesByMatrix(distMatrixTmp)), np.average(LVQCommon.getCostFunctionValuesByMatrix(distMatrixTmp2))

        #figRef, subplotRef = GLVQPlot.plotAll(None, uniPrototypes, tmpPrototypeLabels, samples=uniSamples, samplesLabels=windowSamplesLabels,
        #                                                colors=GLVQPlot.getDefColors(), title='ref', plotBoundary=False, XRange=GLVQPlot.getDefXRange2(), YRange=GLVQPlot.getDefYRange2())
        #plt.show()

        for i in range(addProtos):
            minProtoCandidateIdx, minDistMatrix, minAvgCost, initialAvgCost, dummy = InsertionStrategies.getCandidateSamplingCost(distMatrixTmp, uniSamples, uniSampleLables, uniPrototypes, uniPrototypesLabels, numTries, samplingFct, activFct, logisticFactor, getDistanceFct)
            uniPrototypes = np.vstack([uniPrototypes, uniSamples[minProtoCandidateIdx, :]])
            uniPrototypesLabels = np.append(uniPrototypesLabels, uniSampleLables[minProtoCandidateIdx])
            distMatrixTmp = minDistMatrix
            windowRefCosts.append(minAvgCost)
            windowRefDeltaCosts.append(initialAvgCost - minAvgCost)
        return windowRefCosts, windowRefDeltaCosts

    @staticmethod
    def _getCostFromNullReferencePermutated(distMatrix, numTries, addProtos, windowSamples, windowSamplesLabels, windowPrototypes, windowPrototypesLabels, samplingFct, activFct, logisticFactor, getDistanceFct):
        mjvPrototypes = windowPrototypes.copy()
        mjvPrototypesLabels = windowPrototypesLabels.copy()

        ownSampleLabels = windowSamplesLabels.copy()
        permIndices = np.random.permutation(len(ownSampleLabels))
        ownSampleLabels = ownSampleLabels[permIndices]


        protoIndicesLables = np.empty(shape=(0, 2), dtype=int)

        for sample, i in zip(windowSamples, range(len(windowSamples))):
            tmpSample = sample.copy()
            tmpSample.shape = [len(sample), 1]
            sampleMat = tmpSample * np.ones(shape=[1, len(mjvPrototypes)])
            distances = getDistanceFct(np.transpose(mjvPrototypes), sampleMat)
            closestPrototypeIdx = np.argmin(distances)
            protoIndicesLables = np.vstack([protoIndicesLables, np.hstack([np.atleast_2d(closestPrototypeIdx).T, np.atleast_2d(ownSampleLabels[i]).T])])
        mjvPrototypesLabels = InsertionStrategies.getMajorityVoteLabels(protoIndicesLables, ownSampleLabels, mjvPrototypesLabels)
        distMatrixTmp = DistanceMatrix.getDistanceMatrix(windowSamples, ownSampleLabels, mjvPrototypes, mjvPrototypesLabels, activFct, logisticFactor, getDistanceFct)

        windowRefCosts = []
        windowRefDeltaCosts = []

        windowRefCosts.append(np.average(LVQCommon.getCostFunctionValuesByMatrix(distMatrixTmp)))
        #self.figRef, self.subplotRef = GLVQPlot.plotAll(self, prototypes, prototypesLabels, samples=self._windowSamples, samplesLabels=tmpWindowLabels,
        #                                                colors=GLVQPlot.getDefColors(), title='ref', plotBoundary=False, XRange=GLVQPlot.getDefXRange2(), YRange=GLVQPlot.getDefYRange2())
        #plt.show()

        for i in range(addProtos):
            minProtoCandidateIdx, minDistMatrix, minAvgCost, initialAvgCost, dummy = InsertionStrategies.getCandidateSamplingCost(distMatrixTmp, windowSamples, ownSampleLabels, mjvPrototypes, mjvPrototypesLabels, numTries, samplingFct, activFct, logisticFactor, getDistanceFct)
            mjvPrototypes = np.vstack([mjvPrototypes, windowSamples[minProtoCandidateIdx, :]])
            mjvPrototypesLabels = np.append(mjvPrototypesLabels, ownSampleLabels[minProtoCandidateIdx])
            distMatrixTmp = minDistMatrix
            windowRefCosts.append(minAvgCost)
            windowRefDeltaCosts.append(initialAvgCost - minAvgCost)

        #self.figRef, self.subplotRef = GLVQPlot.plotAll(self, prototypes, prototypesLabels, samples=self._windowSamples, samplesLabels=tmpWindowLabels,
        #                                                colors=GLVQPlot.getDefColors(), title='ref', plotBoundary=False, XRange=GLVQPlot.getDefXRange2(), YRange=GLVQPlot.getDefYRange2())
        #plt.show()
        return windowRefCosts, windowRefDeltaCosts

    @staticmethod
    def _getCostFromNullReferencePermutated2(distMatrix, numTries, addProtos, windowSamples, windowSamplesLabels, windowPrototypes, windowPrototypesLabels, samplingFct, activFct, logisticFactor, getDistanceFct):
        mjvPrototypes = np.empty(shape=(0, windowSamples.shape[1]))
        mjvPrototypesLabels =np.empty(shape=(0, 1), dtype=int)
        windowRefCosts = []
        windowRefDeltaCosts = []
        ownSampleLabels = windowSamplesLabels.copy()
        permIndices = np.random.permutation(len(ownSampleLabels))
        ownSampleLabels = ownSampleLabels[permIndices]

        for label in np.unique(ownSampleLabels):
            protoCandidates = windowSamples[ownSampleLabels == label]
            #shuffledIndices = np.random.permutation(len(protoCandidates))
            #protoCandidates = protoCandidates[shuffledIndices]
            km = KMeans(n_clusters=1, n_init=1)
            km.fit(protoCandidates)
            protoCandidates = km.cluster_centers_
            mjvPrototypes = np.vstack([mjvPrototypes, protoCandidates[0, :]])
            mjvPrototypesLabels = np.append(mjvPrototypesLabels, label)

        distMatrixTmp = DistanceMatrix.getDistanceMatrix(windowSamples, ownSampleLabels, mjvPrototypes, mjvPrototypesLabels, activFct, logisticFactor, getDistanceFct)

        protosToAdd = len(windowPrototypesLabels) - len(np.unique(ownSampleLabels))
        for i in range(protosToAdd):
            minProtoCandidateIdx, minDistMatrix, minAvgCost, initialAvgCost, dummy = InsertionStrategies.getCandidateSamplingCost(distMatrixTmp, windowSamples, ownSampleLabels, mjvPrototypes, mjvPrototypesLabels, numTries, samplingFct, activFct, logisticFactor, getDistanceFct)
            mjvPrototypes = np.vstack([mjvPrototypes, windowSamples[minProtoCandidateIdx, :]])
            mjvPrototypesLabels = np.append(mjvPrototypesLabels, ownSampleLabels[minProtoCandidateIdx])
            distMatrixTmp = minDistMatrix

        windowRefCosts.append(np.average(LVQCommon.getCostFunctionValuesByMatrix(distMatrixTmp)))
        for i in range(addProtos):
            minProtoCandidateIdx, minDistMatrix, minAvgCost, initialAvgCost, dummy = InsertionStrategies.getCandidateSamplingCost(distMatrixTmp, windowSamples, ownSampleLabels, mjvPrototypes, mjvPrototypesLabels, numTries, samplingFct, activFct, logisticFactor, getDistanceFct)
            mjvPrototypes = np.vstack([mjvPrototypes, windowSamples[minProtoCandidateIdx, :]])
            mjvPrototypesLabels = np.append(mjvPrototypesLabels, ownSampleLabels[minProtoCandidateIdx])
            distMatrixTmp = minDistMatrix
            windowRefCosts.append(minAvgCost)
            windowRefDeltaCosts.append(initialAvgCost - minAvgCost)
        return windowRefCosts, windowRefDeltaCosts'''