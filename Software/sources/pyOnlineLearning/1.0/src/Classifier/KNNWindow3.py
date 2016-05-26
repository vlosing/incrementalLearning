__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math

from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import collections
from LVQPY.LVQCommon import LVQCommon
import libNNPythonIntf
from sklearn.cluster import KMeans
class KNNWindow(BaseClassifier):
    def __init__(self, n_neighbors=5, windowSize=200, LTMSize = 0.4, driftStrategy=None, listener=[]):
        self.n_neighbors = n_neighbors
        self._windowSamples = None
        self._windowSamplesLabels = np.empty(shape=(0, 1), dtype=int)
        self._currWindowSize = 0
        self._windowBinaryValues = np.empty(shape=(0, 1), dtype=int)
        #self.distanceMatrix = np.empty(shape=(windowSize, windowSize), dtype=float)
        self._LTMSamples = None
        self._LTMLabels = np.empty(shape=(0, 1), dtype=int)
        self._LTMSize = LTMSize * windowSize

        self.maxWindowSize = windowSize - self._LTMSize
        self.driftStrategy = driftStrategy
        self.dataDimensions = None
        self._trainStepCount = 0
        self.windowSizes = []
        self.LTMSizes = []
        self.reducedWindowSizes = []
        self.listener = listener

        self.STMCorrectCount = 0
        self.BothCorrectCount = 0
        self.allDeletedCount = 0

    def getClassifier(self):
        return None

    def getInfos(self):
        return ''

    @staticmethod
    def getDistances(sample, samples):
        return libNNPythonIntf.get1ToNDistances(sample, samples)
        '''tmpSample = sample.copy()
        tmpSample.shape = [len(sample), 1]

        sampleMat = tmpSample * np.ones(shape=[1, len(samples)])
        return LVQCommon.getSquaredDistance(np.transpose(samples), sampleMat)'''

        '''def addToDistanceMatrix(self, distances):
        numSamples = len(self._windowSamplesLabels)
        #self.distanceMatrix[1:min(numSamples+1,self.maxWindowSize), 1:min(numSamples+1,self.maxWindowSize)] = self.distanceMatrix[0:min(numSamples,self.maxWindowSize-1), 0:min(numSamples,self.maxWindowSize-1)]
        self.distanceMatrix[0, :min(numSamples+1,self.maxWindowSize)] = np.append(0, distances[:self.maxWindowSize-1])
        self.distanceMatrix[:min(numSamples+1,self.maxWindowSize), 0] = np.append(0, distances[:self.maxWindowSize-1])
        #self.distanceMatrix[:min(numSamples+1,self.maxWindowSize), 0] = np.append(0, distances[:self.maxWindowSize-1])
        #self.distanceMatrix[0, :min(numSamples+1,self.maxWindowSize)] = np.append(0, distances[:self.maxWindowSize-1])

    def delFromDistanceMatrix(self, idx):
            self.distanceMatrix = np.delete(self.distanceMatrix, idx, 0)
            self.distanceMatrix = np.delete(self.distanceMatrix, idx, 1)'''

    def getMargin(self, samples, labels):
        distances = libNNPythonIntf.getNToNDistances(samples, samples)
        indices = np.zeros(shape=(len(labels), len(labels)))
        for i in range(len(labels)):
            indices[i, :] = labels != labels[i]
        if indices.any():
            distances = distances[indices.astype(np.bool)]
            #meanMargin = np.mean(distances[indices.astype(np.bool)])
            #minMargin = np.min(distances[indices.astype(np.bool)])
            #return max(minMargin, meanMargin * 0.0005)
            distancesSorted = np.sort(distances)[:len(distances)/500]
            print len(distances), np.mean(distances), np.mean(distancesSorted)
            return np.mean(distancesSorted)
            #return np.percentile(distances, 1)
            #return minMargin
        else:
            return None
        #return np.mean(distancesSTM)

    def getMargin2(self, samples, labels):
        if len(np.unique(labels)) > 1:
            distances = libNNPythonIntf.getNToNDistances(samples, samples)
            marginDistances = []
            for i in range(len(labels)):
                indices = (labels != labels[i]).astype(np.bool)
                marginDistances = np.append(marginDistances, np.min(distances[i, indices]))
            marginDistances = np.sort(marginDistances)
            print len(marginDistances), np.mean(marginDistances[:max(len(labels)/10, 50)]), np.mean(marginDistances)
            return np.mean(marginDistances[:max(len(labels)/10, 50)])
        else:
            return None


    def clusterDown(self, samples, labels):
        print 'cluster Down'
        uniqueLabels = np.unique(labels)
        newSamples = np.empty(shape=(0, samples.shape[1]))
        newLabels = np.empty(shape=(0, 1))
        for i in uniqueLabels:
            tmpSamples = samples[labels == i]
            clustering = KMeans(n_clusters=tmpSamples.shape[0]/2)
            clustering.fit(tmpSamples)
            newSamples = np.vstack([newSamples, clustering.cluster_centers_])
            newLabels = np.append(newLabels, i*np.ones(shape=(tmpSamples.shape[0]/2)))
        return newSamples, newLabels

    '''def addToLTM(self, oldSTMSamples, oldSTMLabels, stmMargin):
        addedCount = 0
        for i in np.arange(len(oldSTMLabels)):
            if len(self._LTMSamples > 10):
                distances = libNNPythonIntf.get1ToNDistances(oldSTMSamples[i, :], self._windowSamples[:self._currWindowSize, :])
                if np.min(distances) > stmMargin:
                    self._LTMSamples = np.vstack([oldSTMSamples[i, :], self._LTMSamples])
                    self._LTMLabels = np.append(oldSTMLabels[i], self._LTMLabels)
                    addedCount += 1
                    if len(self._LTMLabels) > self._LTMSize:
                        self._LTMSamples, self._LTMLabels = self.clusterDown(self._LTMSamples, self._LTMLabels)
                        #self._LTMSamples = np.delete(self._LTMSamples, self._LTMSize, 0)
                        #self._LTMLabels = np.delete(self._LTMLabels, self._LTMSize, 0)
            else:
                self._LTMSamples = np.vstack([oldSTMSamples[i, :], self._LTMSamples])
                self._LTMLabels = np.append(oldSTMLabels[i], self._LTMLabels)
        print 'added ', addedCount, stmMargin'''

    def getHypothesisMargin(self, label, labels, distances):
        indices = labels==label
        indices2 = labels!=label
        if any(indices):
            if any(indices2):
                dSame = np.min(distances[indices])
                dOther = np.min(distances[indices2])
                return dOther - dSame
            else:
                return 1
        else:
            return -1

    def addToLTM(self, oldSTMSamples, oldSTMLabels, stmMargin, ltmMargin):
        addedCount = 0
        deletedCount = 0
        for i in np.arange(len(oldSTMLabels)):
            if len(self._LTMSamples > 10):
                distances = libNNPythonIntf.get1ToNDistances(oldSTMSamples[i, :], self._windowSamples[:self._currWindowSize, :])
                #hypMargin = self.getHypothesisMargin(oldSTMLabels[i], self._windowSamplesLabels[:self._currWindowSize], distances)
                #if np.min(distances) > stmMargin and hypMargin > 0:
                if np.min(distances) > min(stmMargin, ltmMargin):
                    distances = libNNPythonIntf.get1ToNDistances(oldSTMSamples[i, :], self._LTMSamples)
                    #removeIndices = np.where((distances < min(stmMargin, ltmMargin)) & (self._LTMLabels != oldSTMLabels[i]))[0]
                    #deletedCount += len(removeIndices)
                    #self._LTMSamples = np.delete(self._LTMSamples, removeIndices, 0)
                    #self._LTMLabels = np.delete(self._LTMLabels, removeIndices, 0)
                    self._LTMSamples = np.vstack([oldSTMSamples[i, :], self._LTMSamples])
                    self._LTMLabels = np.append(oldSTMLabels[i], self._LTMLabels)
                    addedCount += 1
                    if len(self._LTMLabels) > self._LTMSize:
                        #self._LTMSamples, self._LTMLabels = self.clusterDown(self._LTMSamples, self._LTMLabels)
                        self._LTMSamples = np.delete(self._LTMSamples, self._LTMSize, 0)
                        self._LTMLabels = np.delete(self._LTMLabels, self._LTMSize, 0)
            else:
                self._LTMSamples = np.vstack([oldSTMSamples[i, :], self._LTMSamples])
                self._LTMLabels = np.append(oldSTMLabels[i], self._LTMLabels)
        print 'added ', addedCount,'deleted', deletedCount, stmMargin

    def consolidateLTM(self, stmMargin, ltmMargin):
        if len(self._LTMLabels) > 0:
            deletedCount = 0
            for i in np.arange(len(self._LTMLabels)-1, -1, -1):
                distances = libNNPythonIntf.get1ToNDistances(self._LTMSamples[i,:], self._windowSamples[:self._currWindowSize, :])
                margin = min(stmMargin, ltmMargin)
                if np.min(distances) < margin:
                    self._LTMSamples = np.delete(self._LTMSamples, i, 0)
                    self._LTMLabels = np.delete(self._LTMLabels, i, 0)
                    deletedCount += 1
            print self._trainStepCount, 'len after consolidate ', len(self._LTMLabels), deletedCount, margin


    def trainInc(self, sample, sampleLabel, predictedLabel):
        self._trainStepCount += 1
        self._windowBinaryValues = np.append(predictedLabel != sampleLabel, self._windowBinaryValues)

        #self.addToDistanceMatrix(distances)
        self._windowSamples = np.vstack([sample, self._windowSamples])
        self._windowSamplesLabels = np.append(sampleLabel, self._windowSamplesLabels)


        if len(self._windowSamples) > self.maxWindowSize:
            self._windowSamples = np.delete(self._windowSamples, self.maxWindowSize, 0)
            self._windowSamplesLabels = np.delete(self._windowSamplesLabels, self.maxWindowSize, 0)
            self._windowBinaryValues = np.delete(self._windowBinaryValues, self.maxWindowSize, 0)
            #self.delFromDistanceMatrix(self.maxWindowSize)

        self._currWindowSize = min(self._currWindowSize + 1, self.maxWindowSize)


        newWindowSize = DriftDetection.getWindowSize(self.driftStrategy, self._windowBinaryValues[:self._currWindowSize], self._windowSamples, self._windowSamplesLabels, self._currWindowSize, self.n_neighbors)
        #if newWindowSize != len(self._windowSamples):
        #    print self._trainStepCount, 'new wSize', newWindowSize, len(self._windowSamples)
        if newWindowSize < self._currWindowSize:

            '''oldSTMSamples = self._windowSamples[newWindowSize:self._currWindowSize, :]
            oldSTMLabels = self._windowSamplesLabels[newWindowSize:self._currWindowSize]'''
            self._currWindowSize = newWindowSize
            '''stmMargin = self.getMargin2(self._windowSamples[:self._currWindowSize, :], self._windowSamplesLabels[:self._currWindowSize])
            ltmMargin = self.getMargin2(self._LTMSamples, self._LTMLabels)
            #print stmMargin, ltmMargin
            self.consolidateLTM(stmMargin, ltmMargin)
            self.addToLTM(oldSTMSamples, oldSTMLabels, stmMargin, ltmMargin)'''

        self.windowSizes = np.append(self.windowSizes, self._currWindowSize)
        self.LTMSizes = np.append(self.LTMSizes, len(self._LTMLabels))


        for listener in self.listener:
            listener.onNewTrainStep(self, False, self._trainStepCount)

    def predictSample(self, sample):
        if len(self._windowSamples) > 0:
            distances = KNNWindow.getDistances(sample, np.vstack([self._windowSamples[:self._currWindowSize, :], self._LTMSamples]))
        else:
            distances = np.array(np.atleast_2d([]))
        if self._currWindowSize < self.n_neighbors:
            predictedLabel = -1
        else:
            predictedLabel = self.getMajorityLabelByDistances(distances, np.append(self.windowSamplesLabels[:self._currWindowSize], self._LTMLabels), self.n_neighbors)
        return predictedLabel

    def predictSampleAdaptive(self, sample, label):
        if len(self._windowSamples) > 0:
            distancesSTM = KNNWindow.getDistances(sample, self._windowSamples[:self._currWindowSize, :])
            distancesLTM = KNNWindow.getDistances(sample, self._LTMSamples)
        if self._currWindowSize < self.n_neighbors:
            predictedLabel = -1
        else:
            predictedLabelSTM = self.getMajorityLabelByDistances(distancesSTM, self.windowSamplesLabels[:self._currWindowSize], self.n_neighbors)
            if predictedLabelSTM == label:
                self.STMCorrectCount +=1
            predictedLabelBoth = self.getMajorityLabelByDistances(np.append(distancesSTM, distancesLTM), np.append(self.windowSamplesLabels[:self._currWindowSize], self._LTMLabels), self.n_neighbors)
            if predictedLabelBoth == label:
                self.BothCorrectCount +=1
            if self.BothCorrectCount >= self.STMCorrectCount:
                predictedLabel = predictedLabelBoth
            else:
                predictedLabel = predictedLabelSTM
        return predictedLabel



    def partial_fit(self, samples, labels, classes):
        if self.dataDimensions is None:
            self.dataDimensions = samples.shape[1];
            self._windowSamples = np.empty(shape=(0, self.dataDimensions))
            self._LTMSamples = np.empty(shape=(0, self.dataDimensions))

        predictedLabels = []
        for i in range(len(samples)):
            predictedLabels.append(self.trainInc(samples[i, :], labels[i]))
        return predictedLabels

    def alternateFitPredict(self, samples, labels, classes):
        if self.dataDimensions is None:
            self.dataDimensions = samples.shape[1];
            self._windowSamples = np.empty(shape=(0, self.dataDimensions))
            self._LTMSamples = np.empty(shape=(0, self.dataDimensions))
        predictedTrainLabels = []
        for i in range(len(samples)):
            predLabel = self.predictSample(samples[i,:])
            #predLabel = self.predictSampleAdaptive(samples[i,:], labels[i])
            self.trainInc(samples[i,:], labels[i], predLabel)
            predictedTrainLabels.append(predLabel)
        return predictedTrainLabels

    @staticmethod
    def getMajorityLabelByDistances(distances, labels, numNeighbours):
        '''if distances.ndim ==1:
            nnIndices = np.argsort(distances)[:numNeighbours]
            return collections.Counter(labels[nnIndices]).most_common()[0][0]
        else:
            nnIndices = np.argsort(distances)[:, :numNeighbours]
            predLabels = []
            for labelsNN in labels[nnIndices]:
                predLabels.append(collections.Counter(labelsNN).most_common()[0][0])'''
        nnIndices = libNNPythonIntf.nArgMin(numNeighbours, distances)
        predLabels = libNNPythonIntf.mostCommon(labels[nnIndices].astype(np.int32))
        return predLabels

    def getComplexity(self):
        return 0

    def getComplexityNumParameterMetric(self):
        return 0

    @property
    def windowSamples(self):
        return self._windowSamples[:self._currWindowSize, :]

    @property
    def windowSamplesLabels(self):
        return self._windowSamplesLabels[:self._currWindowSize]


    @property
    def LTMSamples(self):
        return self._LTMSamples

    @property
    def LTMLabels(self):
        return self._LTMLabels

class DriftDetection(object):
    notShrinkedCount = 0
    shrinkedCount = 0
    currentSizeAccs = []
    smallestSizeAccs = []
    largestSizeAccs = []
    @staticmethod
    def getWindowSize(driftStrategy, errorSequence, samples, labels, currWindowSize, nNeighbours):
        if driftStrategy is None:
            return len(labels)
        elif driftStrategy == 'adwin':
            return DriftDetection.getADWINWindowSize(errorSequence)
        elif driftStrategy == 'maxACC7':
            return DriftDetection.getMaxAccWindowSize7(currWindowSize, samples, labels, nNeighbours)
        elif driftStrategy == 'both':
            return min(DriftDetection.getMaxAccWindowSize7(currWindowSize, samples, labels), DriftDetection.getADWINWindowSize(errorSequence))
        else:
            raise Exception('unknown driftStrategy')
        '''elif driftStrategy == 'maxACC2':
            return DriftDetection.getMaxAccWindowSize2(len(labels), samples, labels, classifier)
        elif driftStrategy == 'maxACC3':
            return DriftDetection.getMaxAccWindowSize3(samples, labels, classifier)
        elif driftStrategy == 'maxACC6':
            return DriftDetection.getMaxAccWindowSize6(len(labels), samples, labels, classifier)'''


    @staticmethod
    def getADWINWindowSize(windowValues, minWindowLenght=10, minSubWindowLength=5, delta=0.002, stepSize=10):
        width = len(windowValues)
        minValue = 1
        if width > minWindowLenght:
            v = np.var(windowValues)
            for n0 in np.arange(minSubWindowLength, width-minSubWindowLength, stepSize):
                n1 = width - n0
                u0 = np.sum(windowValues[:n0])
                u1 = np.sum(windowValues[n0:])
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
                    DriftDetection.shrinkedCount += 1
                    return n0
        DriftDetection.notShrinkedCount += 1
        return width

    @staticmethod
    def accScore(predLabels, labels):
        return np.sum(predLabels == labels)/float(len(predLabels))

    @staticmethod
    def getTestAcc(trainSamples, trainLabels, testSamples, testSamplesLabels, nNeighbours):
        '''distances = np.empty(shape=(0, len(trainLabels)))
        for sample in testSamples:
            distances = np.vstack([distances, KNNWindow.getDistances(sample, trainSamples)])
        predLabels = KNNWindow.getMajorityLabelByDistances(distances, trainLabels, nNeighbours)'''
        distances = libNNPythonIntf.getNToNDistances(testSamples, trainSamples)
        predLabels = KNNWindow.getMajorityLabelByDistances(distances, trainLabels, nNeighbours)
        return DriftDetection.accScore(testSamplesLabels, predLabels)

    @staticmethod
    def getMaxAccWindowSize7(currLen, samples, labels, nNeighbours, minSize = 100):
        numSamples = currLen
        if numSamples < minSize:
            DriftDetection.currentSizeAccs.append(-.5)
            DriftDetection.smallestSizeAccs.append(-.5)
            DriftDetection.largestSizeAccs.append(-.5)
            return currLen

        shrinkedSamplesNum = numSamples
        enlargedSamplesNum = numSamples
        numSamplesRange = [numSamples]
        while shrinkedSamplesNum >= minSize:
            shrinkedSamplesNum /= 2.
            numSamplesRange = np.append(shrinkedSamplesNum, numSamplesRange)
            enlargedSamplesNum = enlargedSamplesNum  + shrinkedSamplesNum
        #enlargedSamplesNum = min(enlargedSamplesNum, len(labels))
        #if enlargedSamplesNum not in numSamplesRange:
        #    numSamplesRange = np.append(numSamplesRange, enlargedSamplesNum)


        numSamplesRange = numSamplesRange.astype(int)
        accuracies = []
        stds = []

        for numSamplesIt in numSamplesRange:
            accs = []
            rs = cross_validation.ShuffleSplit(numSamplesIt, n_iter=4, test_size=.3)
            for train_index, test_index in rs:
                #accs.append(DriftDetection.getTestAccByDistanceMatrix(distanceMatrix, train_index, test_index, samples, labels))
                accs.append(DriftDetection.getTestAcc(samples[train_index, :], labels[train_index], samples[test_index, :], labels[test_index], nNeighbours))
            #print numSamplesIt, np.mean(accs), np.std(accs)
            accuracies = np.append(accuracies, np.mean(accs))
            stds = np.append(stds, np.std(accs))
        accuraciesBefore = accuracies
        accuracies = np.round(accuracies - stds, decimals=4)
        DriftDetection.currentSizeAccs.append(accuracies[np.argmax(numSamplesRange==numSamples)])
        DriftDetection.smallestSizeAccs.append(accuracies[0])
        DriftDetection.largestSizeAccs.append(accuracies[-1])

        bestNumTrainIdx = np.argmax(accuracies)
        if numSamplesRange[bestNumTrainIdx] == numSamples:
            DriftDetection.notShrinkedCount += 1
        else:
            DriftDetection.shrinkedCount += 1
        accDelta = accuracies[np.argmax(numSamplesRange==numSamples)] - accuracies[bestNumTrainIdx]
        #windowSize = numSamplesRange[bestNumTrainIdx]
        windowSize = int(numSamples + (numSamples-numSamplesRange[bestNumTrainIdx]) * accDelta)
        #print currLen, numSamplesRange, np.round(accuraciesBefore, decimals=2), np.round(accuracies, decimals=2), np.round(stds, decimals=2), accDelta, windowSize
        return windowSize






    '''@staticmethod
    def getTestAccByDistanceMatrix(distanceMatrix, trainIndex, testIndex, samples, labels):
        predLabels = []
        #print trainIndex, testIndex
        #print distanceMatrix
        predLabels = KNNWindow.getMajorityLabelByDistances(distanceMatrix[testIndex, :][:, trainIndex], labels[trainIndex], 5)
        #print predLabels, labels[testIndex]
        return accuracy_score(labels[testIndex], predLabels)'''

    '''@staticmethod
    def getMaxAccWindowSize2(currLen, samples, labels, classifier, minSizeTest = 30, minSizeTrain = 30):
        numTrainSamples = currLen - minSizeTest
        if numTrainSamples < minSizeTrain:
            return currLen

        testSamples = samples[:minSizeTest, :]
        testSamplesLabels = labels[:minSizeTest:]

        if len(labels) == currLen:
            numTrainRange = [numTrainSamples, numTrainSamples/2]
        else:
            numTrainRange = [numTrainSamples, numTrainSamples/2, min(numTrainSamples*1.5, len(labels))]

        accuracies = []
        for numTrain in numTrainRange:
            accuracies.append(DriftDetection.getTestAcc(classifier, samples[minSizeTest:minSizeTest + numTrain, :], labels[minSizeTest:minSizeTest + numTrain], testSamples, testSamplesLabels))

        bestNumTrainIdx = np.argmax(accuracies)
        accDelta = abs(accuracies[bestNumTrainIdx] - accuracies[0])
        if bestNumTrainIdx != 0:
            #print accuracies, numTrainRange
            if len(labels) != currLen and bestNumTrainIdx == 2:
                windowSize = min(int(numTrainSamples * (1 + accDelta/2.) + minSizeTest), len(labels))
            else:
                windowSize = int(numTrainSamples * (1 - accDelta/2.)) + minSizeTest
        else:
            windowSize = currLen
        #print len(labels), currLen, tmp, accs, windowSize
        return windowSize


    @staticmethod
    def getMaxAccWindowSize3(samples, labels, classifier, minSizeTest = 50, minSizeTrain = 50):
        numTrainSamples = len(samples) - minSizeTest
        if numTrainSamples < minSizeTrain:
            return len(labels)
        testSamples = samples[:minSizeTest, :]
        testSamplesLabels = labels[:minSizeTest:]

        accuracies = []
        numTrainRange = np.arange(numTrainSamples, 4, -numTrainSamples/4)

        for numTrain in numTrainRange:
            #print len(samples), numTrain
            accuracies.append(DriftDetection.getTestAcc(classifier, samples[minSizeTest:minSizeTest + numTrain, :], labels[minSizeTest:minSizeTest + numTrain], testSamples, testSamplesLabels))
        bestNumTrainIdx = np.argmax(accuracies)
        if numTrainRange[bestNumTrainIdx] == numTrainSamples:
            DriftDetection.notShrinkedCount += 1
        else:
            DriftDetection.shrinkedCount += 1
        accDelta = accuracies[0] - accuracies[bestNumTrainIdx]
        windowSize = int(numTrainSamples + (numTrainSamples-numTrainRange[bestNumTrainIdx])/float(numTrainSamples) * accDelta * numTrainSamples) + minSizeTest
        #print accuracies, accDelta, numTrainRange, windowSize
        return windowSize


    @staticmethod
    def getMaxAccWindowSize6(currLen, samples, labels, classifier, minSize = 300):
        numSamples = currLen
        if numSamples < minSize:
            return currLen

        shrinkedSamplesNum = numSamples
        enlargedSamplesNum = numSamples
        numSamplesRange = [numSamples]
        while shrinkedSamplesNum >= minSize:
            shrinkedSamplesNum /= 2.
            numSamplesRange = np.append(shrinkedSamplesNum, numSamplesRange)
            enlargedSamplesNum = min(enlargedSamplesNum + shrinkedSamplesNum, len(labels))
            if enlargedSamplesNum not in numSamplesRange:
                numSamplesRange = np.append(numSamplesRange, enlargedSamplesNum)
        #numSamplesRange = np.arange(numSamples, 30, -numSamples/2)
        #if currLen < len(labels):
        #    numSamplesRange = np.append(numSamplesRange, min(numSamples * 1.5, len(labels)))

        numSamplesRange = numSamplesRange.astype(int)
        accuracies = []

        for numSamplesIt in numSamplesRange:
            kf = cross_validation.KFold(n=numSamplesIt, n_folds=5, shuffle=True)
            accs = []
            for train_index, test_index in kf:
                accs.append(DriftDetection.getTestAcc(classifier, samples[train_index,:], labels[train_index], samples[test_index,:], labels[test_index]))
            #print numSamplesIt, np.mean(accs), np.std(accs)
            accuracies.append(np.mean(accs))
        accuracies = np.round(accuracies, decimals=4)
        bestNumTrainIdx = np.argmax(accuracies)
        if numSamplesRange[bestNumTrainIdx] == numSamples:
            DriftDetection.notShrinkedCount += 1
        else:
            DriftDetection.shrinkedCount += 1

        accDelta = accuracies[np.argmax(numSamplesRange==numSamples)] - accuracies[bestNumTrainIdx]
        #windowSize = numSamplesRange[bestNumTrainIdx]
        windowSize = int(numSamples + (numSamples-numSamplesRange[bestNumTrainIdx]) * accDelta)
        #windowSize = int(numSamples + (numSamples-numSamplesRange[bestNumTrainIdx]) * accDelta * numSamplesRange[bestNumTrainIdx]/(2*numSamples))


        #print currLen, numSamplesRange, accuracies, accDelta, windowSize
        return windowSize'''

