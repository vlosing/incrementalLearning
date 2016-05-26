__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
class KNNWindow(BaseClassifier):
    def __init__(self, n_neighbors=5, windowSize=200, driftStrategy=None, listener=[]):
        self.n_neighbors = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        self._windowSamples = None
        self._windowSamplesLabels = np.empty(shape=(0, 1), dtype=int)
        self._currWindowSize = 0
        self._windowBinaryValues = np.empty(shape=(0, 1), dtype=int)

        self._LTMSamples = None
        self._LTMLabels = np.empty(shape=(0, 1), dtype=int)
        self._LTMSize = 1000

        self.maxWindowSize = windowSize
        self.driftStrategy = driftStrategy
        self.dataDimensions = None
        self._trainStepCount = 0
        self.windowSizes = []
        self.LTMSizes = []
        self.reducedWindowSizes = []
        self.listener = listener

    def fit(self, samples, labels, epochs):
        if self.dataDimensions is None:
            self.dataDimensions = samples.shape[1];
            self._windowSamples = np.empty(shape=(0, self.dataDimensions))
            self._LTMSamples = np.empty(shape=(0, self.dataDimensions))

        self.classifier.fit(samples, labels)

    def getClassifier(self):
        return self.classifier

    def getInfos(self):
        return ''

    def consolidateLTM(self):
        if len(self._LTMLabels) + len(self._windowSamplesLabels[:self._currWindowSize]) > 100:
            for i in np.arange(len(self._LTMLabels)-1, -1, -1):
                classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                trainSamples = np.vstack([self._LTMSamples[0:i:, :], self._windowSamples[:self._currWindowSize, :]])
                trainLabels = np.append(self._LTMLabels[0:i:], self._windowSamplesLabels[:self._currWindowSize])
                classifier.fit(trainSamples, trainLabels)
                predLabel = classifier.predict(self._LTMSamples[i,:].reshape(1, -1))
                if predLabel <> self._LTMLabels[i]:
                    self._LTMSamples = np.delete(self._LTMSamples, i, 0)
                    self._LTMLabels = np.delete(self._LTMLabels, i, 0)

    def trainInc(self, sample, sampleLabel):
        self._trainStepCount += 1

        if self._currWindowSize < self.n_neighbors:
            predictedLabel = -1
        else:
            predictedLabel = self.predict(sample.reshape(1, -1))
        self._windowBinaryValues = np.append(predictedLabel != sampleLabel, self._windowBinaryValues)

        self._windowSamples = np.vstack([sample, self._windowSamples])
        self._windowSamplesLabels = np.append(sampleLabel, self._windowSamplesLabels)



        if len(self._windowSamples) > self.maxWindowSize:
            self._windowSamples = np.delete(self._windowSamples, self.maxWindowSize, 0)
            self._windowSamplesLabels = np.delete(self._windowSamplesLabels, self.maxWindowSize, 0)
            self._windowBinaryValues = np.delete(self._windowBinaryValues, self.maxWindowSize, 0)

        self._currWindowSize = min(self._currWindowSize + 1, self.maxWindowSize)


        #if predictedLabel != sampleLabel:

        newWindowSize = DriftDetection.getWindowSize(self.driftStrategy, self._windowBinaryValues[:self._currWindowSize], self._windowSamples, self._windowSamplesLabels, self.classifier, self._currWindowSize)
        #if newWindowSize != len(self._windowSamples):
        #    print self._trainStepCount, 'new wSize', newWindowSize, len(self._windowSamples)

        if newWindowSize < self._currWindowSize:
            self.consolidateLTM()
        if newWindowSize != self._currWindowSize:
            self._currWindowSize = newWindowSize


        alpha = 0.3
        beta = 0.05
        k=-beta+alpha
        #prob = alpha -k * len(self._LTMLabels)/float(self._LTMSize)
        prob = 0
        if np.random.sample() < prob:
            delIdx = np.random.randint(0, self._currWindowSize)

            self._LTMSamples = np.vstack([self._windowSamples[delIdx,:], self._LTMSamples])
            self._LTMLabels = np.append(self._windowSamplesLabels[delIdx], self._LTMLabels)
            #print self._trainStepCount, 'added', len(self._LTMLabels)

            self._windowSamples = np.delete(self._windowSamples, delIdx, 0)
            self._windowSamplesLabels = np.delete(self._windowSamplesLabels, delIdx, 0)
            self._currWindowSize -= 1

            if len(self._LTMLabels) > self._LTMSize:
                delIdx = np.random.randint(0, len(self._LTMLabels))
                self._LTMSamples = np.delete(self._LTMSamples, delIdx, 0)
                self._LTMLabels = np.delete(self._LTMLabels, delIdx, 0)


        self.windowSizes = np.append(self.windowSizes, self._currWindowSize)
        self.LTMSizes = np.append(self.LTMSizes, len(self._LTMLabels))
        self.classifier.fit(np.vstack([self._windowSamples[:self._currWindowSize, :], self._LTMSamples]), np.append(self._windowSamplesLabels[:self._currWindowSize], self._LTMLabels))

        #print self._trainStepCount, len(self._windowSamplesLabels)
        for listener in self.listener:
            listener.onNewTrainStep(self, False, self._trainStepCount)
        return predictedLabel

    def alternateFitPredict(self, samples, labels, classes):
        if self.dataDimensions is None:
            self.dataDimensions = samples.shape[1];
            self._windowSamples = np.empty(shape=(0, self.dataDimensions))
            self._LTMSamples = np.empty(shape=(0, self.dataDimensions))
        predictedTrainLabels = []
        for i in range(len(samples)):
            predictedTrainLabels.append(self.trainInc(samples[i, :], labels[i]))
        return predictedTrainLabels

    def predict(self, samples):
        return self.classifier.predict(samples)

    def predict_proba(self, samples):
        predictions = self.classifier.predict_proba(samples)
        return predictions

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


class DriftDetection(object):
    notShrinkedCount = 0
    shrinkedCount = 0
    currentSizeAccs = []
    smallestSizeAccs = []
    largestSizeAccs = []
    @staticmethod
    def getWindowSize(driftStrategy, errorSequence, samples, labels, classifier, currWindowSize):
        if driftStrategy is None:
            return len(labels)
        elif driftStrategy == 'adwin':
            return DriftDetection.getADWINWindowSize(errorSequence)
        elif driftStrategy == 'maxACC2':
            return DriftDetection.getMaxAccWindowSize2(len(labels), samples, labels, classifier)
        elif driftStrategy == 'maxACC3':
            return DriftDetection.getMaxAccWindowSize3(samples, labels, classifier)
        elif driftStrategy == 'maxACC6':
            return DriftDetection.getMaxAccWindowSize6(len(labels), samples, labels, classifier)
        elif driftStrategy == 'maxACC7':
            return DriftDetection.getMaxAccWindowSize7(currWindowSize, samples, labels, classifier)
        elif driftStrategy == 'both':
            return min(DriftDetection.getMaxAccWindowSize7(currWindowSize, samples, labels, classifier), DriftDetection.getADWINWindowSize(errorSequence))
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
    def getTestAcc(classifier, trainSamples, trainLabels, testSamples, testSamplesLabels):
        #classifier = KNeighborsClassifier(n_neighbors=classifier.n_neighbors)
        classifier.fit(trainSamples, trainLabels)
        predLabels = classifier.predict(testSamples)
        return accuracy_score(testSamplesLabels, predLabels)


    @staticmethod
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
        return windowSize

    @staticmethod
    def getMaxAccWindowSize7(currLen, samples, labels, classifier, minSize = 100):
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
        enlargedSamplesNum = min(enlargedSamplesNum, len(labels))
        if enlargedSamplesNum not in numSamplesRange:
            numSamplesRange = np.append(numSamplesRange, enlargedSamplesNum)


        numSamplesRange = numSamplesRange.astype(int)
        accuracies = []
        stds = []

        for numSamplesIt in numSamplesRange:
            accs = []
            for iter in range(4):
                XTrain, XTest, yTrain, yTest = train_test_split(samples[:numSamplesIt, :], labels[:numSamplesIt], test_size=0.3)
                accs.append(DriftDetection.getTestAcc(classifier, XTrain, yTrain, XTest, yTest))
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



