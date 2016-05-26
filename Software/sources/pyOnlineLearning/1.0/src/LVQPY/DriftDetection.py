__author__ = 'viktor'
import numpy as np
import math
from sklearn.metrics import accuracy_score
from DeletionStrategies import DeletionStrategies
from InsertionStrategies import InsertionStrategies
from DistanceMatrix import DistanceMatrix
class DriftDetection(object):
    @staticmethod
    def getWindowSize(driftStrategy, binaryValues, samples, labels, distMatrix, prototypes, prototypesLabels, protoLabelCounts, insertionStrategy,
                            protoAdds, deletionStrategy, protoStatistics, insStatstics, trainStepCount, lastInsertionStep, sampling, activFct, logisticFactor, getDistanceFct, predictFct):
        if driftStrategy is None:
            return len(labels)
        elif driftStrategy == 'adwin':
            return DriftDetection.getADWINWindowSize(binaryValues)
        elif driftStrategy == 'maxACC':
            return DriftDetection.getMaxAccWindowSize(samples, labels, distMatrix, prototypes, prototypesLabels, protoLabelCounts, insertionStrategy,
                            protoAdds, deletionStrategy, protoStatistics, insStatstics, trainStepCount, lastInsertionStep, sampling, activFct, logisticFactor, getDistanceFct, predictFct)
        else:
            raise Exception('unknown driftStrategy')

    @staticmethod
    def getADWINWindowSize(windowValues, minWindowLenght=10, minSubWindowLength=5, delta=0.002, stepSize=10):
        width = len(windowValues)
        minValue = 1
        if width > minWindowLenght:
            v = np.var(windowValues)
            for n0 in np.arange(minSubWindowLength, width-minSubWindowLength, stepSize):
                n1 = width - n0
                u0 = np.sum(windowValues[n1:])
                u1 = np.sum(windowValues[0:n1])
                absValue = np.abs(u0/float(n0) - u1/float(n1))
                dd = math.log(2 * math.log(width) / delta)
                m = ( 1. / (n0 - minSubWindowLength + 1)) + (1. / (n1 - minSubWindowLength + 1))
                epsilon = math.sqrt(2 * m * v * dd) + (2./3.) * dd * m
                if epsilon - absValue < minValue:
                    minValue = epsilon - absValue
                #print v, n0, n1, u0/float(n0), u1/float(n1), epsilon, absValue


                if absValue > epsilon:
                    #print 'drift detection', n0, n1, u0/float(n0), u1/float(n1)
                    #print v, n0, n1, u0/float(n0), u1/float(n1), epsilon, absValue
                    return n0
        return width

    @staticmethod
    def getTestAcc(trainSamples, trainLabels, trainDistMatrix, testSamples, testSamplesLabels, prototypes, prototypesLabels, protoLabelCounts, insertionStrategy, protoAdds, deletionStrategy, protoStatistics, insStatstics, trainStepCount, sampling, activFct, logisticFactor, getDistanceFct, predictFct):
        tmpPrototypes = prototypes.copy()
        tmpPrototypesLabels = prototypesLabels.copy()
        tmpProtoLabelCounts = protoLabelCounts.copy()

        for delStrategy in deletionStrategy:
            delIndices = DeletionStrategies.getPrototypeDelIndices(delStrategy, trainDistMatrix, trainSamples, trainLabels, tmpPrototypes, tmpPrototypesLabels, activFct, logisticFactor, getDistanceFct, protoStatistics, trainStepCount)
            delIndices = np.sort(delIndices)[::-1]
            for idx in delIndices:
                if tmpProtoLabelCounts[tmpPrototypesLabels[idx]] > 1:
                    trainDistMatrix, dummy = DistanceMatrix.deleteProtoFromDistanceMatrix(trainDistMatrix,
                                                                                          trainSamples,
                                                                                          trainLabels,
                                                                                          tmpPrototypes,
                                                                                          tmpPrototypesLabels,
                                                                                          idx,
                                                                                          getDistanceFct)
                    tmpProtoLabelCounts[tmpPrototypesLabels[idx]] -= 1
                    tmpPrototypes = np.delete(tmpPrototypes, idx, 0)
                    tmpPrototypesLabels = np.delete(tmpPrototypesLabels, idx, 0)
        candidates, candidatesLabels = InsertionStrategies.getCandidates(insertionStrategy, trainDistMatrix, trainSamples, trainLabels, tmpPrototypes, tmpPrototypesLabels, insStatstics,  sampling, activFct, logisticFactor, getDistanceFct, protoAdds)
        tmpPrototypes = np.vstack([tmpPrototypes, candidates])
        tmpPrototypesLabels = np.append(tmpPrototypesLabels, candidatesLabels)
        predLabels = predictFct(testSamples, tmpPrototypes, tmpPrototypesLabels)
        return accuracy_score(testSamplesLabels, predLabels)

    @staticmethod
    def getMaxAccWindowSize(samples, labels, distMatrix, prototypes, prototypesLabels, protoLabelCounts, insertionStrategy,
                            protoAdds, deletionStrategy, protoStatistics, insStatstics, trainStepCount, lastInsertionStep, sampling, activFct, logisticFactor, getDistanceFct, predictFct, minSize = 50):
        #testSetIdx = len(self._windowSamplesLabels) - (self._trainStepCount - self._lastInsertionStep)
        testSetIdx = max(len(labels) - (trainStepCount - lastInsertionStep), len(labels) - 100)

        if testSetIdx < minSize or testSetIdx - minSize < 10:
            return len(labels)

        testSamples = samples[testSetIdx:, :]
        testSamplesLabels = labels[testSetIdx:]

        '''bestIdx = 0
        low = 0
        high = testSetIdx - minSize
        maxAcc = self.getTestAcc(high, testSetIdx, testSamples, testSamplesLabels)
        iteration = 0
        while iteration < 4:
            currIdx = (low + high)/ 2
            acc = self.getTestAcc(currIdx, testSetIdx, testSamples, testSamplesLabels)
            print acc, currIdx, maxAcc
            if acc > maxAcc:
                high = currIdx
                maxAcc = acc
                bestIdx = currIdx
            else:
                low = currIdx
            iteration += 1
        print 'final', maxAcc, bestIdx'''

        bestIdx = 0
        maxAcc = 0
        for currIdx in np.arange(0, testSetIdx-minSize, (testSetIdx-minSize)/5):
            acc = DriftDetection.getTestAcc(samples[currIdx:testSetIdx, :], labels[currIdx:testSetIdx], distMatrix[currIdx:testSetIdx, :], testSamples, testSamplesLabels, prototypes, prototypesLabels, protoLabelCounts, insertionStrategy,
                                            protoAdds, deletionStrategy, protoStatistics, insStatstics, trainStepCount, sampling, activFct, logisticFactor, getDistanceFct, predictFct)
            if acc > maxAcc:
                maxAcc = acc
                bestIdx = currIdx
        return len(labels) - bestIdx
