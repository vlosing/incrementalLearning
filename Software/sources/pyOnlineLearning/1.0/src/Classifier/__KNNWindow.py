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
    def __init__(self, n_neighbors=5, windowSize=200, LTMSize = 0.4, driftStrategy=None, weights='uniform', listener=[]):
        self.n_neighbors = n_neighbors
        self._windowSamples = None
        self._windowSamplesLabels = np.empty(shape=(0, 1), dtype=int)
        self._windowBinaryValues = np.empty(shape=(0, 1), dtype=int)
        #self.distanceMatrix = np.empty(shape=(windowSize, windowSize), dtype=float)
        self._LTMSamples = None
        self._LTMLabels = np.empty(shape=(0, 1), dtype=int)
        self.maxLTMSize = LTMSize * windowSize
        self.weights = weights

        if weights == 'distance':
            self.getLabelsFct = KNNWindow.getLinearWeightedLabels
        elif weights == 'uniform':
            self.getLabelsFct = KNNWindow.getMajLabels
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
        self.possCorrect = 0
        self.correctCount = 0


        self.LTMPredictions = []
        self.STMPredictions = []
        self.BothPredictions = []

        self.classifierChoice = []
        self.classifierCorrect = []
    def getClassifier(self):
        return None

    def getInfos(self):
        return ''


    @staticmethod
    def getDistances(sample, samples):
        return libNNPythonIntf.get1ToNDistances(sample, samples)

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

    def doSizeCheck(self, maxTotalSize, currSTMSize):
        if currSTMSize > maxTotalSize:
            self._windowSamples = np.delete(self._windowSamples, currSTMSize-1, 0)
            self._windowSamplesLabels = np.delete(self._windowSamplesLabels, currSTMSize-1, 0)
            self._windowBinaryValues = np.delete(self._windowBinaryValues, currSTMSize-1, 0)

    def doSizeCheck2(self, maxTotalSize, maxLTMSize, currSTMSize, currLTMSize):
        if currSTMSize + currLTMSize > maxTotalSize:
            if currLTMSize > maxLTMSize:
                self._LTMSamples, self._LTMLabels = self.clusterDown(self._LTMSamples, self._LTMLabels)
            else:
                while currSTMSize + len(self._LTMLabels) > maxTotalSize:
                    sample, label = self.validateSamples(self._windowSamples[:currSTMSize-1, :], self._windowSamplesLabels[:currSTMSize-1], np.array([self._windowSamples[currSTMSize-1, :]]), np.array([self._windowSamplesLabels[currSTMSize-1]]))
                    self._LTMSamples = np.vstack([sample, self._LTMSamples])
                    self._LTMLabels = np.append(label, self._LTMLabels)
                    if len(self._LTMLabels) > maxLTMSize:
                        self._LTMSamples, self._LTMLabels = self.clusterDown(self._LTMSamples, self._LTMLabels)
                    self._windowSamples = np.delete(self._windowSamples, currSTMSize-1, 0)
                    self._windowSamplesLabels = np.delete(self._windowSamplesLabels, currSTMSize-1, 0)
                    self._windowBinaryValues = np.delete(self._windowBinaryValues, currSTMSize-1, 0)
                    currSTMSize = len(self._windowSamplesLabels)

    def doSizeCheck3(self, totalSize, maxLTMSize, currSTMSize, currLTMSize):
        if currSTMSize + currLTMSize > totalSize:
            if currLTMSize > maxLTMSize:
                self._LTMSamples = np.delete(self._LTMSamples, currLTMSize-1, 0)
                self._LTMLabels = np.delete(self._LTMLabels, currLTMSize-1, 0)
            else:
                while currSTMSize + len(self._LTMLabels) > totalSize:
                    sample, label = self.validateSamples(self._windowSamples[:currSTMSize-1, :], self._windowSamplesLabels[:currSTMSize-1], np.array([self._windowSamples[currSTMSize-1, :]]), np.array([self._windowSamplesLabels[currSTMSize-1]]))
                    self._LTMSamples = np.vstack([sample, self._LTMSamples])
                    self._LTMLabels = np.append(label, self._LTMLabels)
                    if len(self._LTMLabels) > maxLTMSize:
                        self._LTMSamples = np.delete(self._LTMSamples, currLTMSize-1, 0)
                        self._LTMLabels = np.delete(self._LTMLabels, currLTMSize-1, 0)
                    self._windowSamples = np.delete(self._windowSamples, currSTMSize-1, 0)
                    self._windowSamplesLabels = np.delete(self._windowSamplesLabels, currSTMSize-1, 0)
                    self._windowBinaryValues = np.delete(self._windowBinaryValues, currSTMSize-1, 0)

    def validateSamples(self, samples, labels, samples2, labels2, onlyFirst = False):
        delCount = 0
        maxDeletions=self.n_neighbors
        if len(labels) > self.n_neighbors and samples2.shape[0] > 0:
            if onlyFirst:
                loopRange = [0]
            else:
                loopRange = range(len(labels))
            for i in loopRange:
                samplesShortened = np.delete(samples, i, 0)
                labelsShortened = np.delete(labels, i, 0)
                deleted = True
                localDelCount = 0
                while deleted and len(labels2) > 0 and localDelCount < maxDeletions:
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
        return samples2, labels2

    def trainInc(self, sample, sampleLabel, predictedLabel):
        self._trainStepCount += 1

        #print self._trainStepCount, len(self._windowSamplesLabels), len(self._LTMLabels)
        self._windowBinaryValues = np.append(predictedLabel != sampleLabel, self._windowBinaryValues)

        self._windowSamples = np.vstack([sample, self._windowSamples])
        self._windowSamplesLabels = np.append(sampleLabel, self._windowSamplesLabels)
        if self.driftStrategy in ['maxACC7', 'maxACC8']:
            self.doSizeCheck2(self.maxSTMSize + self.maxLTMSize, self.maxLTMSize, len(self._windowSamplesLabels), len(self._LTMLabels))
            #self.doSizeCheck3(self.maxSTMSize + self.maxLTMSize, self.maxLTMSize, len(self._windowSamplesLabels), len(self._LTMLabels))
        else:
            self.doSizeCheck(self.maxSTMSize + self.maxLTMSize, len(self._windowSamplesLabels))
        self._LTMSamples, self._LTMLabels = self.validateSamples(self._windowSamples, self.windowSamplesLabels, self._LTMSamples, self._LTMLabels, onlyFirst=True)

        newWindowSize = DriftDetection.getWindowSize(self.driftStrategy, self._windowBinaryValues, self._windowSamples, self._windowSamplesLabels, self.n_neighbors, self.getLabelsFct)
        if newWindowSize < len(self._windowSamplesLabels):
            oldSTMSamples = self._windowSamples[newWindowSize:]
            oldSTMLabels = self._windowSamplesLabels[newWindowSize:]

            delIndices = np.arange(newWindowSize, len(self._windowSamples))
            self._windowSamples = np.delete(self._windowSamples, delIndices, 0)
            self._windowSamplesLabels = np.delete(self._windowSamplesLabels, delIndices, 0)
            self._windowBinaryValues = np.delete(self._windowBinaryValues, delIndices, 0)
            if self.driftStrategy != 'adwin':
                oldSTMSamples, oldSTMLabels = self.validateSamples(self._windowSamples, self._windowSamplesLabels, oldSTMSamples, oldSTMLabels)
                #self._LTMSamples, self._LTMLabels = self.validateSamples(oldSTMSamples, oldSTMLabels, self._LTMSamples, self._LTMLabels)
                self._LTMSamples = np.vstack([oldSTMSamples, self._LTMSamples])
                self._LTMLabels = np.append(oldSTMLabels, self._LTMLabels)
                if self.driftStrategy in ['maxACC7', 'maxACC8']:
                    self.doSizeCheck2(self.maxSTMSize + self.maxLTMSize, self.maxLTMSize, len(self._windowSamplesLabels), len(self._LTMLabels))
                    #self.doSizeCheck3(self.maxSTMSize + self.maxLTMSize, self.maxLTMSize, len(self._windowSamplesLabels), len(self._LTMLabels))
                else:
                    self.doSizeCheck(self.maxSTMSize + self.maxLTMSize, len(self._windowSamplesLabels))

        self.windowSizes = np.append(self.windowSizes, len(self._windowSamplesLabels))
        self.LTMSizes = np.append(self.LTMSizes, len(self._LTMLabels))
        for listener in self.listener:
            listener.onNewTrainStep(self, False, self._trainStepCount)

    def predictSampleAdaptive(self, sample, label):
        predictedLabelLTM = 0
        predictedLabelSTM = 0
        predictedLabelBoth = 0
        classifierChoice = 0
        if len(self._windowSamplesLabels) == 0:
            predictedLabel = predictedLabelSTM
        else:
            if len(self._windowSamplesLabels) < self.n_neighbors:
                distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
                predictedLabelSTM = self.getLabelsFct(distancesSTM, self._windowSamplesLabels, len(self._windowSamplesLabels))[0]
                predictedLabel = predictedLabelSTM
            else:
                distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
                distancesLTM = KNNWindow.getDistances(sample, self._LTMSamples)
                predictedLabelSTM = self.getLabelsFct(distancesSTM, self._windowSamplesLabels, self.n_neighbors)[0]
                predictedLabelBoth = self.getLabelsFct(np.append(distancesSTM, distancesLTM), np.append(self._windowSamplesLabels, self._LTMLabels), self.n_neighbors)[0]

                if len(self._LTMLabels) >= self.n_neighbors:
                    predictedLabelLTM = self.getLabelsFct(distancesLTM, self._LTMLabels, self.n_neighbors)[0]
                    correctLTM = np.sum(self.LTMPredictions[:len(self._windowSamplesLabels)])
                    correctSTM = np.sum(self.STMPredictions[:len(self._windowSamplesLabels)])
                    correctBoth = np.sum(self.BothPredictions[:len(self._windowSamplesLabels)])
                    labels = [predictedLabelSTM, predictedLabelBoth, predictedLabelLTM]
                    classifierChoice = np.argmax([correctSTM, correctBoth, correctLTM])
                    self.classifierChoice.append(classifierChoice)
                    predictedLabel = labels[classifierChoice]
                else:
                    predictedLabel = predictedLabelSTM
        self.classifierChoice.append(classifierChoice)
        self.BothPredictions = np.append(predictedLabelBoth == label, self.BothPredictions)
        self.BothCorrectCount += predictedLabelBoth == label
        self.STMPredictions = np.append(predictedLabelSTM == label, self.STMPredictions)
        self.STMCorrectCount += predictedLabelSTM == label
        self.LTMPredictions = np.append(predictedLabelLTM == label, self.LTMPredictions)
        self.LTMCorrectCount += predictedLabelLTM == label
        self.possCorrect += label in [predictedLabelSTM, predictedLabelBoth, predictedLabelLTM]
        self.correctCount += predictedLabel == label
        return predictedLabel

    '''def predictSampleAdaptiveLocal(self, sample, label):
        predictedLabelLTM = 0
        predictedLabelSTM = 0
        predictedLabelBoth = 0
        classifierChoice = 0
        if len(self._windowSamplesLabels) == 0:
            predictedLabel = predictedLabelSTM
        else:
            if len(self._windowSamplesLabels) < self.n_neighbors:
                distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
                predictedLabelSTM = self.getLabelsFct(distancesSTM, self._windowSamplesLabels, len(self._windowSamplesLabels))[0]
                predictedLabel = predictedLabelSTM
            else:
                distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
                distancesLTM = KNNWindow.getDistances(sample, self._LTMSamples)
                predictedLabelSTM = self.getLabelsFct(distancesSTM, self._windowSamplesLabels, self.n_neighbors)[0]
                predictedLabelBoth = self.getLabelsFct(np.append(distancesSTM, distancesLTM), np.append(self._windowSamplesLabels, self._LTMLabels), self.n_neighbors)[0]
                localClassifierChoice = self.getLabelsFct(distancesSTM, self.classifierCorrect[:len(self._windowSamplesLabels)], self.n_neighbors)[0]

                if len(self._LTMLabels) >= self.n_neighbors:
                    predictedLabelLTM = self.getLabelsFct(distancesLTM, self._LTMLabels, self.n_neighbors)[0]
                    if localClassifierChoice == 3:
                        classifierChoice = np.random.randint(3)
                    else:
                        classifierChoice = localClassifierChoice
                    labels = [predictedLabelSTM, predictedLabelBoth, predictedLabelLTM]
                    predictedLabel = labels[classifierChoice]
                else:
                    predictedLabel = predictedLabelSTM
                    if localClassifierChoice == 3:
                        classifierChoice = np.random.randint(2)
                    else:
                        classifierChoice = localClassifierChoice

        self.classifierChoice.append(classifierChoice)
        self.BothPredictions = np.append(predictedLabelBoth == label, self.BothPredictions)
        if predictedLabelBoth == label:
            self.BothCorrectCount += 1
        self.STMPredictions = np.append(predictedLabelSTM == label, self.STMPredictions)
        if predictedLabelSTM == label:
            self.STMCorrectCount += 1
        self.LTMPredictions = np.append(predictedLabelLTM == label, self.LTMPredictions)
        if predictedLabelLTM == label:
            self.LTMCorrectCount += 1
        labels = [predictedLabelSTM, predictedLabelBoth, predictedLabelLTM]
        if label in labels:
            self.possCorrect += 1
        if predictedLabel == label:
            self.correctCount += 1

        correctClassifierIndices = np.arange(3)[labels == label]
        if len(correctClassifierIndices) == 0:
            self.classifierCorrect = np.append(3, self.classifierCorrect)
        else:
            self.classifierCorrect = np.append(correctClassifierIndices[np.random.randint(len(correctClassifierIndices))], self.classifierCorrect)
        #print labels, correctClassifierIndices, self.classifierCorrect[0]
        return predictedLabel

    def predictSampleAdaptiveLocal2(self, sample, label):
        predictedLabelLTM = 0
        predictedLabelSTM = 0
        predictedLabelBoth = 0
        classifierChoice = 0
        correctLTM = np.sum(self.LTMPredictions[:len(self._windowSamplesLabels)])
        correctSTM = np.sum(self.STMPredictions[:len(self._windowSamplesLabels)])
        correctBoth = np.sum(self.BothPredictions[:len(self._windowSamplesLabels)])
        classifierPriorChoice = np.argmax([correctSTM, correctBoth, correctLTM])
        if len(self._windowSamplesLabels) == 0:
            predictedLabel = predictedLabelSTM
        else:
            if len(self._windowSamplesLabels) < self.n_neighbors:
                distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
                predictedLabelSTM = self.getLabelsFct(distancesSTM, self._windowSamplesLabels, len(self._windowSamplesLabels))[0]
                predictedLabel = predictedLabelSTM
            else:
                distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
                distancesLTM = KNNWindow.getDistances(sample, self._LTMSamples)
                predictedLabelSTM = self.getLabelsFct(distancesSTM, self._windowSamplesLabels, self.n_neighbors)[0]
                predictedLabelBoth = self.getLabelsFct(np.append(distancesSTM, distancesLTM), np.append(self._windowSamplesLabels, self._LTMLabels), self.n_neighbors)[0]
                localClassifierChoice = self.getLabelsFct(distancesSTM, self.classifierCorrect[:len(self._windowSamplesLabels)], self.n_neighbors)[0]

                if len(self._LTMLabels) >= self.n_neighbors:
                    predictedLabelLTM = self.getLabelsFct(distancesLTM, self._LTMLabels, self.n_neighbors)[0]
                    if localClassifierChoice == 3:
                        classifierChoice = classifierPriorChoice
                    else:
                        classifierChoice = localClassifierChoice
                    labels = [predictedLabelSTM, predictedLabelBoth, predictedLabelLTM]
                    predictedLabel = labels[classifierChoice]
                else:
                    predictedLabel = predictedLabelSTM
                    if localClassifierChoice == 3:
                        classifierChoice = np.argmax([correctSTM, correctBoth])
                    else:
                        classifierChoice = localClassifierChoice

        self.classifierChoice.append(classifierChoice)
        self.BothPredictions = np.append(predictedLabelBoth == label, self.BothPredictions)
        if predictedLabelBoth == label:
            self.BothCorrectCount += 1
        self.STMPredictions = np.append(predictedLabelSTM == label, self.STMPredictions)
        if predictedLabelSTM == label:
            self.STMCorrectCount += 1
        self.LTMPredictions = np.append(predictedLabelLTM == label, self.LTMPredictions)
        if predictedLabelLTM == label:
            self.LTMCorrectCount += 1
        labels = [predictedLabelSTM, predictedLabelBoth, predictedLabelLTM]
        if label in labels:
            self.possCorrect += 1
        if predictedLabel == label:
            self.correctCount += 1

        correctClassifierIndices = np.arange(3)[labels == label]
        if len(correctClassifierIndices) == 0:
            self.classifierCorrect = np.append(3, self.classifierCorrect)
        else:
            self.classifierCorrect = np.append(correctClassifierIndices[np.random.randint(len(correctClassifierIndices))], self.classifierCorrect)
        #print labels, correctClassifierIndices, self.classifierCorrect[0]
        return predictedLabel

    def predictSampleAdaptiveLocal3(self, sample, label):
        predictedLabelLTM = 0
        predictedLabelSTM = 0
        predictedLabelBoth = 0
        classifierChoice = 0
        correctLTM = np.sum(self.LTMPredictions[:len(self._windowSamplesLabels)])
        correctSTM = np.sum(self.STMPredictions[:len(self._windowSamplesLabels)])
        correctBoth = np.sum(self.BothPredictions[:len(self._windowSamplesLabels)])
        classifierPriorChoice = np.argsort([correctSTM, correctBoth, correctLTM])[::-1]
        if len(self._windowSamplesLabels) == 0:
            predictedLabel = predictedLabelSTM
        else:
            if len(self._windowSamplesLabels) < self.n_neighbors:
                distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
                predictedLabelSTM = self.getLabelsFct(distancesSTM, self._windowSamplesLabels, len(self._windowSamplesLabels))[0]
                predictedLabel = predictedLabelSTM
            else:
                distancesSTM = KNNWindow.getDistances(sample, self._windowSamples)
                distancesLTM = KNNWindow.getDistances(sample, self._LTMSamples)
                predictedLabelSTM = self.getLabelsFct(distancesSTM, self._windowSamplesLabels, self.n_neighbors)[0]
                predictedLabelBoth = self.getLabelsFct(np.append(distancesSTM, distancesLTM), np.append(self._windowSamplesLabels, self._LTMLabels), self.n_neighbors)[0]
                localClassifierChoice = self.getLabelsFct(distancesSTM, self.classifierCorrect[:len(self._windowSamplesLabels)], self.n_neighbors)[0]

                if len(self._LTMLabels) >= self.n_neighbors:
                    predictedLabelLTM = self.getLabelsFct(distancesLTM, self._LTMLabels, self.n_neighbors)[0]
                    if localClassifierChoice == 3:
                        classifierChoice = classifierPriorChoice[0]
                    else:
                        classifierChoice = localClassifierChoice
                    labels = [predictedLabelSTM, predictedLabelBoth, predictedLabelLTM]
                    predictedLabel = labels[classifierChoice]
                else:
                    predictedLabel = predictedLabelSTM
                    if localClassifierChoice == 3:
                        classifierChoice = np.argmax([correctSTM, correctBoth])
                    else:
                        classifierChoice = localClassifierChoice

        self.classifierChoice.append(classifierChoice)
        self.BothPredictions = np.append(predictedLabelBoth == label, self.BothPredictions)
        if predictedLabelBoth == label:
            self.BothCorrectCount += 1
        self.STMPredictions = np.append(predictedLabelSTM == label, self.STMPredictions)
        if predictedLabelSTM == label:
            self.STMCorrectCount += 1
        self.LTMPredictions = np.append(predictedLabelLTM == label, self.LTMPredictions)
        if predictedLabelLTM == label:
            self.LTMCorrectCount += 1
        labels = [predictedLabelSTM, predictedLabelBoth, predictedLabelLTM]
        if label in labels:
            self.possCorrect += 1
        if predictedLabel == label:
            self.correctCount += 1

        correctClassifierIndices = np.arange(3)[labels == label]
        if len(correctClassifierIndices) == 0:
            self.classifierCorrect = np.append(3, self.classifierCorrect)
        else:
            if classifierPriorChoice[0] in correctClassifierIndices:
                self.classifierCorrect = np.append(classifierPriorChoice[0], self.classifierCorrect)
            elif classifierPriorChoice[1] in correctClassifierIndices:
                self.classifierCorrect = np.append(classifierPriorChoice[1], self.classifierCorrect)
            else:
                self.classifierCorrect = np.append(classifierPriorChoice[2], self.classifierCorrect)
        #print labels, correctClassifierIndices, self.classifierCorrect[0]
        return predictedLabel'''

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
            predLabel = self.predictSampleAdaptive(samples[i,:], labels[i])
            #predLabel = self.predictSampleAdaptiveLocal(samples[i,:], labels[i])
            #predLabel = self.predictSampleAdaptiveLocal2(samples[i,:], labels[i])
            #predLabel = self.predictSampleAdaptiveLocal3(samples[i,:], labels[i])

            self.trainInc(samples[i,:], labels[i], predLabel)
            predictedTrainLabels.append(predLabel)
        return predictedTrainLabels

    @staticmethod
    def getMajLabels(distances, labels, numNeighbours):
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
    def getLinearWeightedLabels(distances, labels, numNeighbours):
        nnIndices = libNNPythonIntf.nArgMin(numNeighbours, distances)

        if distances.ndim > 1 and distances.shape[0]>1:
            rowIndices = np.repeat(np.arange(distances.shape[0]), numNeighbours)
            sqrtDistances = np.sqrt(distances[rowIndices, nnIndices.reshape(len(rowIndices))])
            sqrtDistances = sqrtDistances.reshape(distances.shape[0], numNeighbours)
        else:
            sqrtDistances = np.sqrt(distances[nnIndices])
        predLabels = libNNPythonIntf.getLinearWeightedLabels(labels[nnIndices].astype(np.int32), sqrtDistances)
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
    def getWindowSize(driftStrategy, errorSequence, samples, labels, nNeighbours, getLabelsFct):
        if driftStrategy is None:
            return len(labels)
        elif driftStrategy == 'adwin':
            return DriftDetection.getADWINWindowSize(errorSequence)
        elif driftStrategy == 'maxACC7':
            return DriftDetection.getMaxAccWindowSize7(samples, labels, nNeighbours, getLabelsFct)
        elif driftStrategy == 'maxACC8':
            return DriftDetection.getMaxAccWindowSize8(samples, labels, nNeighbours, getLabelsFct)
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
    def getTestAcc(trainSamples, trainLabels, testSamples, testSamplesLabels, nNeighbours, getLabelsFct):
        '''distances = np.empty(shape=(0, len(trainLabels)))
        for sample in testSamples:
            distances = np.vstack([distances, KNNWindow.getDistances(sample, trainSamples)])
        predLabels = KNNWindow.getMajorityLabelByDistances(distances, trainLabels, nNeighbours)'''
        distances = libNNPythonIntf.getNToNDistances(testSamples, trainSamples)
        #predLabels = KNNWindow.getMajorityLabelByDistances(distances, trainLabels, nNeighbours)
        predLabels = getLabelsFct(distances, trainLabels, nNeighbours)
        return DriftDetection.accScore(testSamplesLabels, predLabels)

    '''@staticmethod
    def getInterleavedTrainTestAcc(samples, labels, nNeighbours, getLabelsFct):
        print labels
        predLabels = []
        for i in range(nNeighbours, len(labels)):
            distances = libNNPythonIntf.get1ToNDistances(samples[i, :], samples[:i, :])
            predLabels.append(getLabelsFct(distances, labels[:i], nNeighbours)[0])
        return DriftDetection.accScore(predLabels, labels[nNeighbours:])'''

    @staticmethod
    def getInterleavedTrainTestAcc(samples, labels, nNeighbours, getLabelsFct):
        #print labels
        predLabels = []
        #for i in range(nNeighbours, len(labels)):
        for i in range(len(labels)-nNeighbours-1, -1, -1):
            distances = libNNPythonIntf.get1ToNDistances(samples[i, :], samples[i+1:, :])
            predLabels.append(getLabelsFct(distances, labels[i+1:], nNeighbours)[0])
        return DriftDetection.accScore(predLabels[::-1], labels[:len(labels)-nNeighbours])

    @staticmethod
    def getInterleavedTrainTestAcc2(samples, labels, nNeighbours, getLabelsFct):
        predLabels = []
        distances = libNNPythonIntf.getNToNDistances(samples, samples)
        #print distances.shape, distances, distances[1,2]

        for i in range(nNeighbours, len(labels)):
            distances2 = distances[i, :i]
            predLabels.append(getLabelsFct(distances2, labels[:i], nNeighbours)[0])
        return DriftDetection.accScore(predLabels, labels[nNeighbours:])


    @staticmethod
    def getMaxAccWindowSize7(samples, labels, nNeighbours, getLabelsFct, minSize = 100):
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
                accs.append(DriftDetection.getTestAcc(samples[train_index, :], labels[train_index], samples[test_index, :], labels[test_index], nNeighbours, getLabelsFct))
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

    @staticmethod
    def getMaxAccWindowSize8(samples, labels, nNeighbours, getLabelsFct, minSize=100):
        numSamples = len(labels)
        if numSamples < minSize:
            return numSamples
        shrinkedSamplesNum = numSamples
        numSamplesRange = [numSamples]
        while shrinkedSamplesNum >= minSize:
            shrinkedSamplesNum /= 2.
            numSamplesRange = np.append(shrinkedSamplesNum, numSamplesRange)
        numSamplesRange = numSamplesRange.astype(int)
        accuracies = []
        for numSamplesIt in numSamplesRange:
            accs = []
            accs.append(DriftDetection.getInterleavedTrainTestAcc(samples[:numSamplesIt, :], labels[:numSamplesIt], nNeighbours, getLabelsFct))
            accuracies = np.append(accuracies, np.mean(accs))
        accuracies = np.round(accuracies, decimals=4)

        bestNumTrainIdx = np.argmax(accuracies)
        windowSize = numSamplesRange[bestNumTrainIdx]
        #accDelta = accuracies[np.argmax(numSamplesRange==numSamples)] - accuracies[bestNumTrainIdx]
        #windowSize = int(numSamples + (numSamples-numSamplesRange[bestNumTrainIdx]) * accDelta)
        #print len(labels), numSamplesRange, np.round(accuracies, decimals=2), windowSize
        return windowSize
