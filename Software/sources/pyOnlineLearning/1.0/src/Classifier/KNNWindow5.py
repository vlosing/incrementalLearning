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
        self._windowBinaryValues = np.empty(shape=(0, 1), dtype=int)
        #self.distanceMatrix = np.empty(shape=(windowSize, windowSize), dtype=float)
        self._LTMSamples = None
        self._LTMLabels = np.empty(shape=(0, 1), dtype=int)
        self.maxLTMSize = LTMSize * windowSize

        self.maxSTMSize = windowSize - self.maxLTMSize
        self.driftStrategy = driftStrategy
        self.dataDimensions = None
        self._trainStepCount = 0
        self.windowSizes = []
        self.LTMSizes = []
        self.reducedWindowSizes = []
        self.listener = listener

        self.STMCorrectCount = 0
        self.LTMCorrectCount = 0
        self.BothCorrectCount = 0
        self.allDeletedCount = 0
        self.LTMPredictions = []
        self.STMPredictions = []
        self.BothPredictions = []

    def getClassifier(self):
        return None

    def getInfos(self):
        return ''


    @staticmethod
    def getDistances(sample, samples):
        return libNNPythonIntf.get1ToNDistances(sample, samples)

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
        for label in uniqueLabels:
            tmpSamples = samples[labels == label]
            newLength = max(tmpSamples.shape[0]/2, 1)
            clustering = KMeans(n_clusters=newLength)
            clustering.fit(tmpSamples)
            newSamples = np.vstack([newSamples, clustering.cluster_centers_])
            newLabels = np.append(newLabels, label*np.ones(shape=newLength))
        return newSamples, newLabels

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

    def doSizeCheck(self, totalSize, maxSTMSize, maxLTMSize, currSTMSize, currLTMSize):
        if currSTMSize + currLTMSize > totalSize:
            if currLTMSize > maxLTMSize:
                self._LTMSamples, self._LTMLabels = self.clusterDown(self._LTMSamples, self._LTMLabels)
                #self._LTMSamples = np.delete(self._LTMSamples, self.maxLTMSize, 0)
                #self._LTMLabels = np.delete(self._LTMLabels, self.maxLTMSize, 0)
            elif currSTMSize > maxSTMSize:
                self._windowSamples = np.delete(self._windowSamples, self.maxSTMSize, 0)
                self._windowSamplesLabels = np.delete(self._windowSamplesLabels, self.maxSTMSize, 0)
                self._windowBinaryValues = np.delete(self._windowBinaryValues, self.maxSTMSize, 0)

    def validateSamples(self, samples, labels, samples2, labels2, onlyFirst = False):
        delCount = 0
        maxDeletions=self.n_neighbors
        #maxDeletions=5

        if len(labels) > self.n_neighbors and len(labels2) > 0:
            if onlyFirst:
                loopRange = [0]
            else:
                loopRange = range(len(labels))
            for i in loopRange:
                samplesShortened = np.delete(samples, i, 0)
                labelsShortened = np.delete(labels, i, 0)
                deleted = True
                localDelCount = 0
                while deleted and localDelCount < maxDeletions:
                    deleted = False
                    trainSamples = np.vstack([samplesShortened, samples2])
                    trainLabels = np.append(labelsShortened, labels2)
                    distances = libNNPythonIntf.get1ToNDistances(samples[i,:], trainSamples)
                    nnIndicesShortened = libNNPythonIntf.nArgMin(self.n_neighbors, distances[:len(labelsShortened)])[0]
                    correctCount = np.sum(labelsShortened[nnIndicesShortened] == labels[i])

                    nnIndices = libNNPythonIntf.nArgMin(self.n_neighbors, distances)[0]
                    correctCount2 = np.sum(trainLabels[nnIndices] == labels[i])

                    if correctCount > correctCount2:
                        nnIndices -= len(labelsShortened)
                        indices = np.where(nnIndices >= 0)[0]
                        if len(indices) > 0:
                            #print nnIndices, indices
                            wrongIndices = np.where(labels2[nnIndices[indices]] != labels[i])[0]
                            if len(wrongIndices) > 0:
                                #print wrongIndices, nnIndices[indices[wrongIndices]]
                                samples2 = np.delete(samples2, nnIndices[indices[wrongIndices]], 0)
                                labels2 = np.delete(labels2, nnIndices[indices[wrongIndices]], 0)
                                localDelCount += len(wrongIndices)
                                delCount += len(wrongIndices)
                                deleted = True
        #print initialLength, len(labels2), delCount
        return samples2, labels2


    def trainInc(self, sample, sampleLabel, predictedLabel):
        self._trainStepCount += 1
        self._windowBinaryValues = np.append(predictedLabel != sampleLabel, self._windowBinaryValues)

        self._windowSamples = np.vstack([sample, self._windowSamples])
        self._windowSamplesLabels = np.append(sampleLabel, self._windowSamplesLabels)
        self.doSizeCheck(self.maxSTMSize + self.maxLTMSize, self.maxSTMSize, self.maxLTMSize, len(self._windowSamplesLabels), len(self._LTMLabels))

        self._LTMSamples, self._LTMLabels = self.validateSamples(self._windowSamples, self.windowSamplesLabels, self._LTMSamples, self._LTMLabels, onlyFirst=True)

        newWindowSize = DriftDetection.getWindowSize(self.driftStrategy, self._windowBinaryValues, self._windowSamples, self._windowSamplesLabels, self.n_neighbors)
        if newWindowSize < len(self._windowSamplesLabels):
            oldSTMSamples = self._windowSamples[newWindowSize:]
            oldSTMLabels = self._windowSamplesLabels[newWindowSize:]

            delIndices = np.arange(newWindowSize, len(self._windowSamples))
            self._windowSamples = np.delete(self._windowSamples, delIndices, 0)
            self._windowSamplesLabels = np.delete(self._windowSamplesLabels, delIndices, 0)
            self._windowBinaryValues = np.delete(self._windowBinaryValues, delIndices, 0)
            if self.driftStrategy != 'adwin':
                oldSTMSamples, oldSTMLabels = self.validateSamples(self._windowSamples, self.windowSamplesLabels, oldSTMSamples, oldSTMLabels)
                #self._LTMSamples, self._LTMLabels = self.validateSamples(oldSTMSamples, oldSTMLabels, self._LTMSamples, self._LTMLabels)
                self._LTMSamples = np.vstack([oldSTMSamples, self._LTMSamples])
                self._LTMLabels = np.append(oldSTMLabels, self._LTMLabels)
                self.doSizeCheck(self.maxSTMSize + self.maxLTMSize, self.maxSTMSize, self.maxLTMSize, len(self._windowSamplesLabels), len(self._LTMLabels))
        self.windowSizes = np.append(self.windowSizes, len(self._windowSamplesLabels))
        self.LTMSizes = np.append(self.LTMSizes, len(self._LTMLabels))


        for listener in self.listener:
            listener.onNewTrainStep(self, False, self._trainStepCount)

    def predictSample(self, sample):
        if len(self._windowSamples) > 0:
            distances = KNNWindow.getDistances(sample, np.vstack([self._windowSamples, self._LTMSamples]))
        else:
            distances = np.array(np.atleast_2d([]))
        if len(self._windowSamplesLabels) < self.n_neighbors:
            predictedLabel = -1
        else:
            predictedLabel = self.getMajorityLabelByDistances(distances, np.append(self.windowSamplesLabels, self._LTMLabels), self.n_neighbors)
        return predictedLabel

    def predictSampleAdaptive(self, sample, label):
        if len(self._windowSamples) > 0:
            distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
            distancesLTM = KNNWindow.getDistances(sample, self._LTMSamples)
        if len(self._windowSamplesLabels) < self.n_neighbors:
            predictedLabel = -1
        else:

            predictedLabelSTM = self.getMajorityLabelByDistances(distancesSTM, self.windowSamplesLabels, self.n_neighbors)
            predictedLabelBoth = self.getMajorityLabelByDistances(np.append(distancesSTM, distancesLTM), np.append(self.windowSamplesLabels, self._LTMLabels), self.n_neighbors)
            if self.BothCorrectCount >= self.STMCorrectCount:
                predictedLabel = predictedLabelBoth
            else:
                predictedLabel = predictedLabelSTM
            if predictedLabelSTM == label:
                self.STMCorrectCount += 1
            if predictedLabelBoth == label:
                self.BothCorrectCount += 1
        return predictedLabel


    def getPredictionLabel(self, labelsSTM, confSTM, labelsLTM, confLTM):
        allLabels = np.unique(np.append(labelsSTM, labelsLTM))
        allConfidences = np.zeros(shape=allLabels.shape)
        for j in range(len(allLabels)):
            for i in range(len(labelsSTM)):
                if labelsSTM[i] == allLabels[j]:
                    allConfidences[j] += confSTM[i]
            for i in range(len(labelsLTM)):
                if labelsLTM[i] == allLabels[j]:
                    allConfidences[j] += confLTM[i]
        #print labelsSTM, confSTM, labelsLTM, confLTM, allLabels, allConfidences, allLabels[np.argmax(allConfidences)]
        return allLabels[np.argmax(allConfidences)]


    def predictSampleAdaptive3(self, sample, label):
        if len(self._windowSamplesLabels) < self.n_neighbors:
            predictedLabel = -1
        else:
            distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
            distancesLTM = KNNWindow.getDistances(sample, self._LTMSamples)
            labelsSTM, confSTM = self.getLabelConfidences(distancesSTM, self.windowSamplesLabels, self.n_neighbors)
            predictedLabelSTM = labelsSTM[np.argmax(confSTM)]
            if len(self._LTMLabels) >= self.n_neighbors:
                labelsLTM, confLTM = self.getLabelConfidences(distancesLTM, self._LTMLabels, self.n_neighbors)
                predictedLabelLTM = labelsLTM[np.argmax(confLTM)]
                correctLTM = np.sum(self.LTMPredictions)
                correctSTM = np.sum(self.STMPredictions)
                priorLTM = correctLTM/float(correctSTM + correctLTM)
                priorSTM = correctSTM/float(correctSTM + correctLTM)
                #print 'c', confLTM, confSTM, priorLTM, priorSTM
                confLTM *= priorLTM
                confSTM *= priorSTM

                predictedLabel = self.getPredictionLabel(labelsSTM, confSTM, labelsLTM, confLTM)
                self.LTMPredictions = np.append(predictedLabelLTM == label, self.LTMPredictions)
                if predictedLabelLTM == label:
                    self.LTMCorrectCount += 1
            else:
                predictedLabel = predictedLabelSTM
            self.STMPredictions = np.append(predictedLabelSTM == label, self.STMPredictions)
            if predictedLabelSTM == label:
                self.STMCorrectCount += 1

        return predictedLabel

    def getWeightedLabel(self, distancesSTM, labelsSTM, priorSTM, distancesLTM, labelsLTM, priorLTM):
        distances = np.append(distancesSTM, distancesLTM)
        labels = np.append(labelsSTM, labelsLTM)


        nnIndices = libNNPythonIntf.nArgMin(self.n_neighbors, distances)[0]
        weights = np.ones(shape=len(nnIndices)) * priorSTM
        weights[np.where(nnIndices >= len(distancesSTM))] = priorLTM

        predictedLabels = np.unique(labels[nnIndices])
        confidences = np.zeros(shape=len(predictedLabels))
        for i in range(len(nnIndices)):
            labelIdx = np.where(predictedLabels == labels[nnIndices[i]])[0]
            confidences[labelIdx] += weights[i]

        #print len(distancesSTM), priorSTM, priorLTM, nnIndices, weights, labels[nnIndices], predictedLabels, confidences
        return predictedLabels[np.argmax(confidences)]

    def predictSampleAdaptive4(self, sample, label):
        if len(self._windowSamplesLabels) < self.n_neighbors:
            predictedLabel = -1
        else:
            distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)

            predictedLabelSTM = self.getMajorityLabelByDistances(distancesSTM, self.windowSamplesLabels, self.n_neighbors)[0]

            if len(self._LTMLabels) >= self.n_neighbors:
                distancesLTM = KNNWindow.getDistances(sample, self._LTMSamples)
                predictedLabelLTM = self.getMajorityLabelByDistances(distancesLTM, self._LTMLabels, self.n_neighbors)[0]
                correctLTM = np.sum(self.LTMPredictions[:len(self._windowSamplesLabels)])
                correctSTM = np.sum(self.STMPredictions[:len(self._windowSamplesLabels)])
                priorLTM = (correctLTM/float(correctSTM + correctLTM))
                priorSTM = (correctSTM/float(correctSTM + correctLTM))
                predictedLabel = self.getWeightedLabel(distancesSTM, self.windowSamplesLabels, priorSTM, distancesLTM, self._LTMLabels, priorLTM)

                self.LTMPredictions = np.append(predictedLabelLTM == label, self.LTMPredictions)
                if predictedLabelLTM == label:
                    self.LTMCorrectCount += 1

                predictedLabelBoth = self.getMajorityLabelByDistances(np.append(distancesSTM, distancesLTM), np.append(self.windowSamplesLabels, self._LTMLabels), self.n_neighbors)[0]
                if predictedLabelBoth == label:
                    self.BothCorrectCount += 1
            else:
                predictedLabel = predictedLabelSTM
            self.STMPredictions = np.append(predictedLabelSTM == label, self.STMPredictions)
            if predictedLabelSTM == label:
                self.STMCorrectCount += 1
        return predictedLabel

    def predictSampleAdaptive2(self, sample, label):
        if len(self._windowSamplesLabels) < self.n_neighbors:
            predictedLabel = -1
        else:
            distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
            distancesLTM = KNNWindow.getDistances(sample, self._LTMSamples)
            predictedLabelSTM = self.getMajorityLabelByDistances(distancesSTM, self.windowSamplesLabels, self.n_neighbors)

            if len(self._LTMLabels) >= self.n_neighbors:
                predictedLabelLTM = self.getMajorityLabelByDistances(distancesLTM, self._LTMLabels, self.n_neighbors)

                correctLTM = np.sum(self.LTMPredictions[:len(self._windowSamplesLabels)])
                correctSTM = np.sum(self.STMPredictions[:len(self._windowSamplesLabels)])
                priorLTM = correctLTM/float(correctSTM + correctLTM)
                priorSTM = correctSTM/float(correctSTM + correctLTM)
                if priorLTM > priorSTM:
                    predictedLabel = predictedLabelLTM
                else:
                    predictedLabel = predictedLabelSTM
                self.LTMPredictions = np.append(predictedLabelLTM == label, self.LTMPredictions)
                if predictedLabelLTM == label:
                    self.LTMCorrectCount += 1

            else:
                predictedLabel = predictedLabelSTM

            predictedLabelBoth = self.getMajorityLabelByDistances(np.append(distancesSTM, distancesLTM), np.append(self.windowSamplesLabels, self._LTMLabels), self.n_neighbors)
            if predictedLabelBoth == label:
                self.BothCorrectCount += 1
            self.STMPredictions = np.append(predictedLabelSTM == label, self.STMPredictions)
            if predictedLabelSTM == label:
                self.STMCorrectCount += 1
        return predictedLabel

    def predictSampleAdaptive5(self, sample, label):
        if len(self._windowSamplesLabels) < self.n_neighbors:
            predictedLabel = -1
        else:
            distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
            distancesLTM = KNNWindow.getDistances(sample, self._LTMSamples)
            predictedLabelSTM = self.getMajorityLabelByDistances(distancesSTM, self.windowSamplesLabels, self.n_neighbors)
            predictedLabelBoth = self.getMajorityLabelByDistances(np.append(distancesSTM, distancesLTM), np.append(self.windowSamplesLabels, self._LTMLabels), self.n_neighbors)
            correctSTM = np.sum(self.STMPredictions[:len(self._windowSamplesLabels)])
            correctBoth = np.sum(self.BothPredictions[:len(self._windowSamplesLabels)])
            if len(self._LTMLabels) >= self.n_neighbors:
                predictedLabelLTM = self.getMajorityLabelByDistances(distancesLTM, self._LTMLabels, self.n_neighbors)

                correctLTM = np.sum(self.LTMPredictions[:len(self._windowSamplesLabels)])
                priorLTM = correctLTM/float(correctSTM + correctLTM + correctBoth)
                priorSTM = correctSTM/float(correctSTM + correctLTM + correctBoth)
                priorBoth = correctBoth/float(correctSTM + correctLTM + correctBoth)

                predLabels = [predictedLabelSTM, predictedLabelLTM, predictedLabelBoth]
                predictedLabel = predLabels[np.argmax([priorSTM, priorLTM, priorBoth])]

                self.LTMPredictions = np.append(predictedLabelLTM == label, self.LTMPredictions)
                if predictedLabelLTM == label:
                    self.LTMCorrectCount += 1
            else:
                self.LTMPredictions = np.append(0, self.LTMPredictions)
                priorSTM = correctSTM/float(correctSTM + correctBoth)
                priorBoth = correctBoth/float(correctSTM + correctBoth)
                if priorBoth > priorSTM:
                    predictedLabel = predictedLabelBoth
                else:
                    predictedLabel = predictedLabelSTM

            self.STMPredictions = np.append(predictedLabelSTM == label, self.STMPredictions)
            if predictedLabelSTM == label:
                self.STMCorrectCount += 1
            self.BothPredictions = np.append(predictedLabelBoth == label, self.BothPredictions)
            if predictedLabelBoth == label:
                self.BothCorrectCount += 1
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
            #predLabel = self.predictSampleAdaptive2(samples[i,:], labels[i])
            #predLabel = self.predictSampleAdaptive3(samples[i,:], labels[i])
            #predLabel = self.predictSampleAdaptive4(samples[i,:], labels[i])
            #predLabel = self.predictSampleAdaptive5(samples[i,:], labels[i])


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

    @staticmethod
    def getLabelConfidences(distances, labels, numNeighbours):
        nnIndices = libNNPythonIntf.nArgMin(numNeighbours, distances)
        labels, counts = np.unique(labels[nnIndices], return_counts=True)
        confidences = counts /float(np.sum(counts))
        return labels, confidences

    def getComplexity(self):
        return 0

    def getComplexityNumParameterMetric(self):
        return 0

    @property
    def windowSamples(self):
        return self._windowSamples

    @property
    def windowSamplesLabels(self):
        return self._windowSamplesLabels


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
    def getWindowSize(driftStrategy, errorSequence, samples, labels, nNeighbours):
        if driftStrategy is None:
            return len(labels)
        elif driftStrategy == 'adwin':
            return DriftDetection.getADWINWindowSize(errorSequence)
        elif driftStrategy == 'maxACC7':
            return DriftDetection.getMaxAccWindowSize7(samples, labels, nNeighbours)
        elif driftStrategy == 'both':
            return min(DriftDetection.getMaxAccWindowSize7(samples, labels, nNeighbours), DriftDetection.getADWINWindowSize(errorSequence))
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
    def getMaxAccWindowSize7(samples, labels, nNeighbours, minSize = 100):
        numSamples = len(labels)
        if numSamples < minSize:
            DriftDetection.currentSizeAccs.append(-.5)
            DriftDetection.smallestSizeAccs.append(-.5)
            DriftDetection.largestSizeAccs.append(-.5)
            return numSamples

        shrinkedSamplesNum = numSamples
        numSamplesRange = [numSamples]
        while shrinkedSamplesNum >= minSize:
            shrinkedSamplesNum /= 2.
            numSamplesRange = np.append(shrinkedSamplesNum, numSamplesRange)
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
