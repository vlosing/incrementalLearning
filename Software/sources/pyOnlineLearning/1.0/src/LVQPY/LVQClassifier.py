import logging

import numpy as np
import math
from LVQCommon import LVQCommon
from ClassifierCommon.BaseClassifier import BaseClassifier
from sklearn.cluster import KMeans
from InsertionStrategies import InsertionStrategies
from InsertionStatistics import InsertionStatistics
from DeletionStrategies import DeletionStrategies
from DistanceMatrix import DistanceMatrix
from DriftDetection import DriftDetection
class LVQClassifier(BaseClassifier):
    def __init__(self, activFct='linear', logisticFactor=10.0, learnRateInitial=1.0,
                 learnRateAnnealingSteps=0, learnRatePerProto=False, windowSize=200,
                 insertionTimingThresh=1, protoAdds=1,
                 retrainFreq=0, insertionStrategy='SamplingCost', insertionTiming='errorCount', sampling='random',
                 deletionStrategy=[None],
                 driftStrategy=None,
                 listener=np.array([]), ):
        """The constructor."""
        super(LVQClassifier, self).__init__()
        self.dataDimensions = None
        self.learnRateInitial = learnRateInitial
        self.learnRateAnnealingSteps = learnRateAnnealingSteps
        self.learnRatePerProto = learnRatePerProto
        self.activFct = activFct
        self.logisticFactor = logisticFactor
        self.windowSize = windowSize
        self.protoAdds = protoAdds
        self.sampling = sampling
        self.driftStrategy = driftStrategy

        self.listener = listener

        self._prototypes = None
        self._prototypesLabels = np.empty(shape=(0, 1), dtype=int)
        self._protoStatistics = np.empty(shape=(0, 7))

        self._windowSamples = None
        self._windowSamplesLabels = np.empty(shape=(0, 1), dtype=int)
        self._windowBinaryValues = np.empty(shape=(0, 1), dtype=int)
        self._windowDistanceMatrix = np.empty(shape=(0, 5))

        self._metricWeights = np.array([])
        self._omegaMetricWeights = np.array([])

        self._trainStepCount = 0
        self._lastInsertionStep = 0
        self._errorCount = 0


        self.retrainFreq = retrainFreq
        self.insertionStrategy = insertionStrategy
        self.insertionTiming = insertionTiming
        self.insertionTimingThresh = insertionTimingThresh

        self.deletionStrategy = deletionStrategy
        self.protoLabelCounts = {}

        self._deletedProtoCount = 0
        self._totalDeletionWindowDeltaCost = 0
        self._lastDeletionWindowDeltaCost = 0

        self._deltaCost = 0
        self._totalDeltaCost = 0
        self.reducedWindowSizes = []

        self.windowSizes = []

        logging.debug('Configuration: \n' + self.getConfigurationStr())

        self.minClusterSize = max(self.windowSize * 0.005, 3)

        self._insStatistics = InsertionStatistics()

    def _adaptPrototype(self, protoIdx, delta, factor, learnRate, activFctDeriv):
        raise NotImplementedError()

    def _adaptMetric(self, deltaWinner, deltaLooser, factorWinner, factorLooser, activFctDeriv):
        raise NotImplementedError()

    def _getDistance(self, A, B):
        raise NotImplementedError()

    def _initMetricWeights(self):
        raise NotImplementedError

    def _normalizeOmegaMetricWeights(self):
        """Normalizes omegaMetricWeights by the sum of the diagonal entries of the matrix
           _omegaMetricWeights.T * _omegaMetricWeights)."""
        normalizationValue = math.sqrt(np.sum(np.einsum('ij,ji->i',
                                                        self._omegaMetricWeights.T, self._omegaMetricWeights)))
        self._omegaMetricWeights /= normalizationValue

    def _internTrainStep(self, sample, sampleLabel, adaptMetric=True):
        values = LVQCommon._getWinnerLooserPrototypeIndices(sample, sampleLabel, self._prototypes, self._prototypesLabels, self._getDistance)
        winnerIdx = values[0][0]
        looserIdx = values[1][0]

        winnerDist = values[0][1]
        looserDist = values[1][1]

        if winnerIdx == -1:
            logging.debug('no prototype for label ' + str(sampleLabel) + ' added one!')
            self.addPrototype(sample, sampleLabel)
            winnerIdx = len(self._prototypes) - 1
            winnerDist = 0

        if looserIdx > -1 and winnerIdx > -1 and (((winnerDist + looserDist) * (winnerDist + looserDist)) > 0):
            deltaWinner = sample - self._prototypes[winnerIdx]

            factorWinner = looserDist / ((winnerDist + looserDist) * (winnerDist + looserDist))
            deltaLooser = sample - self._prototypes[looserIdx]
            factorLooser = winnerDist / ((winnerDist + looserDist) * (winnerDist + looserDist))

            activFctDeriv = 1  # for linear activationFct
            if self.activFct == 'logistic':
                mu = LVQCommon.getMuValue(winnerDist, looserDist)
                tmp = LVQCommon.logisticFunction(mu, logisticFactor=self.logisticFactor)
                activFctDeriv = tmp * (1 - tmp)

            learnRate = self._getLearnRate(winnerIdx)
            self._adaptPrototype(winnerIdx, deltaWinner, factorWinner, learnRate, activFctDeriv)
            learnRate = self._getLearnRate(looserIdx)
            self._adaptPrototype(looserIdx, deltaLooser, factorLooser, -learnRate, activFctDeriv)
            if adaptMetric:
                self._adaptMetric(deltaWinner, deltaLooser, factorWinner, factorLooser, activFctDeriv)

            if looserIdx > -1 and winnerIdx > -1:
                if winnerDist < looserDist:
                    protoIdx = winnerIdx
                    self._protoStatistics[winnerIdx, 1] += 1
                    self._protoStatistics[protoIdx, 3] = self._trainStepCount
                else:
                    protoIdx = looserIdx
                    self._protoStatistics[protoIdx, 2] += 1
                self._protoStatistics[protoIdx, 0] += 1


        return winnerIdx, winnerDist, looserIdx, looserDist

    @property
    def metricWeights(self):
        return self._metricWeights

    @property
    def windowSamples(self):
        return self._windowSamples

    @property
    def windowSamplesLabels(self):
        return self._windowSamplesLabels

    @property
    def prototypes(self):
        return self._prototypes

    @property
    def prototypesLabels(self):
        return self._prototypesLabels

    @property
    def insertedProtoCount(self):
        return self._insStatistics._insertionCount

    @property
    def deletedProtoCount(self):
        return self._deletedProtoCount

    @property
    def triedInsertions(self):
        return self._insStatistics._triedInsertionCount

    @property
    def avgInsertionWindowDeltaCost(self):
        return self._insStatistics._totalInsertionWindowDeltaCost / float(max(self._insStatistics._insertionCount, 1))

    @property
    def lastInsertionWindowDeltaCost(self):
        return self._insStatistics._lastInsertionWindowDeltaCost

    @property
    def avgTriedInsertionWindowDeltaCost(self):
        return self._insStatistics._totalTriedInsertionWindowDeltaCost / float(max(self._insStatistics._triedInsertionCount, 1))

    @property
    def lastTriedInsertionWindowDeltaCost(self):
        return self._insStatistics._lastTriedInsertionWindowDeltaCost

    @property
    def avgInsertionWindowPrototypeCount(self):
        return self._insStatistics._totalInsertionWindowPrototypeCount / float(max(self._insStatistics._insertionCount, 1))

    @property
    def lastInsertionWindowPrototypeCount(self):
        return self._insStatistics._lastInsertionWindowPrototypeCount

    @property
    def avgTriedInsertionWindowPrototypeCount(self):
        return self._insStatistics._totalTriedInsertionWindowPrototypeCount / float(max(self._insStatistics._triedInsertionCount, 1))

    @property
    def lastTriedInsertionWindowPrototypeCount(self):
        return self._insStatistics._lastTriedInsertionWindowPrototypeCount

    @property
    def avgInsertionWindowPrototypeDensity(self):
        return self._insStatistics._totalInsertionWindowPrototypeDensity / float(max(self._insStatistics._insertionCount, 1))

    @property
    def lastInsertionWindowPrototypeDensity(self):
        return self._insStatistics._lastInsertionWindowPrototypeDensity

    @property
    def avgTriedInsertionWindowPrototypeDensity(self):
        return self._insStatistics._totalTriedInsertionWindowPrototypeDensity / float(max(self._insStatistics._triedInsertionCount, 1))

    @property
    def lastTriedInsertionWindowPrototypeDensity(self):
        return self._insStatistics._lastTriedInsertionWindowPrototypeDensity

    @property
    def protoDeletionDeltaCost(self):
        return self._totalDeletionWindowDeltaCost / float(self._insStatistics._deltriedInsertionCount)

    @property
    def lastDeletionDeltaCost(self):
        return self._lastDeletionWindowDeltaCost

    @property
    def totalDeltaCost(self):
        return self._totalDeltaCost

    @property
    def trainStepCount(self):
        return self._trainStepCount

    @property
    def deltaCost(self):
        return self._deltaCost

    def _getLearnRate(self, protoIdx):
        if self.learnRatePerProto:
            stepCount = self._protoStatistics[protoIdx, 4]
        else:
            stepCount = self._trainStepCount
        return LVQCommon.getLinearAnnealedLearnRate(self.learnRateInitial, stepCount, self.learnRateAnnealingSteps)

    def _checkDistanceMatrix(self, distMatrix, samples, sampleLabels):
        groundTruthMatrix = DistanceMatrix.getDistanceMatrix(samples, sampleLabels, self._prototypes, self._prototypesLabels, self.activFct, self.logisticFactor, self._getDistance)
        equal = np.all(np.isclose(distMatrix, groundTruthMatrix))
        # equal = np.all(classMatrix == groundTruthMatrix)
        logging.info('matrix equal == ' + str(equal))
        if not equal:
            logging.info(groundTruthMatrix.shape)
            logging.info(groundTruthMatrix)
            logging.info(distMatrix.shape)
            logging.info(distMatrix)
            logging.info(groundTruthMatrix == distMatrix)


    def _refreshMetricWeights(self):
        self._metricWeights = np.dot(self._omegaMetricWeights.T, self._omegaMetricWeights)

    def addPrototype(self, prototype, label):
        self._prototypes = np.vstack([self._prototypes, prototype])
        self._prototypesLabels = np.append(self._prototypesLabels, label)

        self._protoStatistics = np.vstack([self._protoStatistics, [1, 1, 0, self._trainStepCount, 0, 0, 0]])
        self._windowDistanceMatrix, self._protoStatistics = DistanceMatrix.addProtoToDistanceMatrix(self._windowDistanceMatrix,
                                                                                                    self._windowSamples, prototype,
                                                                                                    label, len(self._prototypes) - 1,
                                                                                                    self._prototypesLabels,
                                                                                                    self.activFct,
                                                                                                    self.logisticFactor,
                                                                                                    self._getDistance,
                                                                                                    self._protoStatistics,
                                                                                                    checkOverlapping=True)
        self._insStatistics._insertionCount += 1

        logging.debug(str(len(self._prototypes)) + ' prototypes after ' + str(self._trainStepCount) + ' samples')

        if label in self.protoLabelCounts.keys():
            self.protoLabelCounts[label] += 1
        else:
            self.protoLabelCounts[label] = 1

        for listener in self.listener:
            listener.onNewPrototypes(self, prototype, label)



    def _predictSample(self, sample, prototypes, prototypesLabels, returnConfidence=False):
        [minIdx, minDist] = LVQCommon._getPrototypeIdxWithMinDistance(prototypes, sample, self._getDistance)
        if minIdx == -1:
            logging.error('no prototypes!')
            return -1
        else:
            if returnConfidence:
                dummy, minDistOtherLabel = LVQCommon._getPrototypeIdxWithMinDistance(prototypes, sample, self._getDistance,
                                                                                excludedLabel=prototypesLabels[minIdx],
                                                                                protoLabels=prototypesLabels)
                confidence = (minDistOtherLabel - minDist) / (minDist + minDistOtherLabel)
                return prototypesLabels[minIdx], confidence
            else:
                return prototypesLabels[minIdx]

    def _predictIntern(self, samples, prototypes, prototypesLabels):
        if samples.ndim == 2:
            result = []
            for sample in samples:
                result.append(self._predictSample(sample, prototypes, prototypesLabels, returnConfidence=False))
            return result
        else:
            return self._predictSample(samples, prototypes, prototypesLabels, returnConfidence=False)

    def predict(self, samples):
        return self._predictIntern(samples, self._prototypes, self._prototypesLabels)


    def predict_proba(self, samples):
        if len(np.unique(self.prototypesLabels)) > 2:
            raise 'LVQ predict proba is only implemented for 2 classes problems'
        result = np.empty(shape=(0, 2))
        if samples.ndim == 2:
            for sample in samples:
                label, confidence = self._predictSample(sample, self._prototypes, self._prototypesLabels, returnConfidence=True)
                confidence = 0.5 + confidence/2.
                if label == 0:
                    result = np.vstack([result, [confidence, 1 - confidence]])
                else:
                    result = np.vstack([result, [1 - confidence, confidence]])
            return result

        else:
            label, confidence = self._predictSample(samples, self._prototypes, self._prototypesLabels, returnConfidence=True)
            if label == 0:
                result = np.vstack([result, [confidence, 1 - confidence]])
            else:
                result = np.vstack([result, [1 - confidence, confidence]])
            return result

    def predict_confidence2Class(self, samples):
        samples = np.array(samples)
        predictions = self.predict_proba(samples)
        return predictions[:, 1]

    def getConfidenceForAllClasses(self, sample):
        protos = self._prototypes
        [minIdx, minDist] = self._getPrototypeIdxWithMinDistance(protos, sample)
        if minIdx == -1:
            logging.debug('no prototypes!')
            return None
        else:
            result = []
            for i in np.unique(self._prototypesLabels):
                if i == self._prototypesLabels[minIdx]:
                    dummy, minDistOtherLabel = self._getPrototypeIdxWithMinDistance(protos, sample,
                                                                                    excludedLabel=
                                                                                    self._prototypesLabels[minIdx],
                                                                                    protoLabels=self._prototypesLabels)
                    confidence = (minDistOtherLabel - minDist) / (minDist + minDistOtherLabel)
                else:
                    dummy, minDistOtherLabel = self._getPrototypeIdxWithMinDistance(protos, sample, forcedLabel=i,
                                                                                    protoLabels=self._prototypesLabels)
                    confidence = (minDist - minDistOtherLabel) / (minDist + minDistOtherLabel)
                result.append([i, confidence])
            return sorted(result, key=lambda item: item[1], reverse=True)

    def trainSamples(self, samples, sampleLabels):
        for i in range(len(sampleLabels)):
            self.train(samples[i, :], sampleLabels[i])


    def partial_fit(self, samples, labels, classes):
        self.fit(samples, labels, 1)

    def alternateFitPredict(self, samples, labels, classes):
        return self.fit(samples, labels, 1)

    def fit(self, samples, labels, epochs):
        if self.dataDimensions is None:
            self.dataDimensions = samples.shape[1];
            self._prototypes = np.empty(shape=(0, self.dataDimensions))
            self._windowSamples = np.empty(shape=(0, self.dataDimensions))
            self._initMetricWeights()

        predictedLabels = []
        for e in range(epochs):
            for i in range(len(samples)):
                predictedLabels.append(self.trainInc(samples[i, :], labels[i]))
        return predictedLabels

    def getInternDistanceMatrix(self):
        return DistanceMatrix.getDistanceMatrix(self._windowSamples, self._windowSamplesLabels, self._prototypes, self._prototypesLabels, self.activFct, self.logisticFactor, self._getDistance)

    def getShortTermMemoryClassRate(self):
        classMatrix = self.getInternDistanceMatrix()
        return LVQCommon.getAccuracyByDistanceMatrix(classMatrix), \
               LVQCommon.getAvgCostValueByDistanceMatrix(classMatrix)

    def getShortTermMemoryClassRateOnInternalDistanceMatrix(self):
        return LVQCommon.getAccuracyByDistanceMatrix(self._windowDistanceMatrix), \
               LVQCommon.getAvgCostValueByDistanceMatrix(self._windowDistanceMatrix)


    def getCostValues(self, samples, labels):
        return DistanceMatrix.getDistanceMatrix(samples, labels, self._prototypes, self._prototypesLabels, self.activFct, self.logisticFactor, self._getDistance)[:, 4]

    def trainInc(self, sample, sampleLabel):
        predictedLabel = self.train(sample, sampleLabel)
        isCorrect = predictedLabel == sampleLabel
        for listener in self.listener:
            listener.onNewTrainStep(self, isCorrect, self._trainStepCount)

        if len(self.prototypes) > 1:
            if self.insertionTiming == 'trainStepCount':
                if self.insertionTimingThresh > 0 and self._trainStepCount % self.insertionTimingThresh == 0:
                    self._prototypeInsertion(self.insertionStrategy, self._windowDistanceMatrix)
            elif self.insertionTiming == 'errorCount':
                if not isCorrect:
                    self._errorCount += 1
                    if self.insertionTimingThresh > 0 and self._errorCount > self.insertionTimingThresh:

                        newWindowSize = DriftDetection.getWindowSize(self.driftStrategy, self._windowBinaryValues, self._windowSamples, self._windowSamplesLabels, self._windowDistanceMatrix, self._prototypes,
                                                     self._prototypesLabels, self.protoLabelCounts, self.insertionStrategy, self.protoAdds, self.deletionStrategy, self._protoStatistics, self._insStatistics,
                                                     self._trainStepCount, self._lastInsertionStep, self.sampling, self.activFct, self.logisticFactor, self._getDistance, self._predictIntern)

                        if newWindowSize < len(self._windowSamples):
                            self.reducedWindowSizes.append(newWindowSize)
                            difference = len(self._windowSamples) - newWindowSize
                            print self._trainStepCount, 'new wSize', newWindowSize, len(self._windowSamples)
                            self._windowSamples = np.delete(self._windowSamples, [range(difference)], 0)
                            self._windowSamplesLabels = np.delete(self._windowSamplesLabels, [range(difference)], 0)
                            self._windowDistanceMatrix = np.delete(self._windowDistanceMatrix, [range(difference)], 0)
                            self._windowBinaryValues = np.delete(self._windowBinaryValues, [range(difference)], 0)
                            for listener in self.listener:
                                listener.onNewWindowSize(self)
                        #else:
                        #   print self._trainStepCount, 'no drift', newWindowSize
                        for delStrategy in self.deletionStrategy:
                            delIndices = DeletionStrategies.getPrototypeDelIndices(delStrategy, self._windowDistanceMatrix, self.windowSamples, self.windowSamplesLabels, self._prototypes, self._prototypesLabels, self.activFct, self.logisticFactor, self._getDistance, self._protoStatistics, self._trainStepCount)
                            self._removeProtoIndices(delIndices)
                        candidates, candidatesLabels = InsertionStrategies.getCandidates(self.insertionStrategy, self._windowDistanceMatrix, self.windowSamples, self.windowSamplesLabels, self._prototypes,
                                                                                         self._prototypesLabels, self._insStatistics,  self.sampling, self.activFct, self.logisticFactor, self._getDistance, self.protoAdds)
                        self._insertProtos(candidates, candidatesLabels)

                        self._lastInsertionStep = self._trainStepCount
                        #self.retrain(self._windowSamples, self._windowSamplesLabels, len(self._windowSamplesLabels)/4)
                        self._errorCount = 0
            else:
                raise NameError('unknown insertionTiming ' + self.insertionTiming)
        self.windowSizes.append(len(self._windowSamplesLabels))
        if self.retrainFreq > 0:
            randIndices = np.random.randint(len(self._windowSamplesLabels), size=self.retrainFreq)
            for i in range(self.retrainFreq):
                self._internTrainStep(self._windowSamples[randIndices[i]],
                                      self._windowSamplesLabels[randIndices[i]])

        return predictedLabel

    def retrain(self, samples, samplesLabels, amount):
        indices = InsertionStrategies._getRouletteSamplingIndices(amount, len(samplesLabels))
        for idx in indices:
                self._internTrainStep(samples[idx], samplesLabels[idx])

    def train(self, sample, sampleLabel):
        winnerIdx, winnerDist, looserIdx, looserDist = self._internTrainStep(sample, sampleLabel)
        if winnerDist < looserDist:
            predictedLabel = self.prototypesLabels[winnerIdx]
        else:
            predictedLabel = self.prototypesLabels[looserIdx]
        self._trainStepCount += 1

        self._windowSamples = np.vstack([self._windowSamples, sample])

        self._windowSamplesLabels = np.append(self._windowSamplesLabels, sampleLabel)
        self._windowDistanceMatrix = np.vstack([self._windowDistanceMatrix,
                                                np.array([winnerIdx, winnerDist, looserIdx, looserDist,
                                                          LVQCommon.getCostFunctionValue(winnerDist, looserDist,
                                                                                         activFct=self.activFct,
                                                                                         logisticFactor=self.logisticFactor)])])
        self._windowBinaryValues = np.append(self._windowBinaryValues, predictedLabel != sampleLabel)
        if len(self._windowSamples) > self.windowSize:
            self._windowSamples = np.delete(self._windowSamples, 0, 0)
            self._windowSamplesLabels = np.delete(self._windowSamplesLabels, 0, 0)
            self._windowDistanceMatrix = np.delete(self._windowDistanceMatrix, 0, 0)
            self._windowBinaryValues = np.delete(self._windowBinaryValues, 0, 0)
        '''if self.trainStepCount % 250 == 0:
            print self.trainStepCount, np.sum(self._windowBinaryValues)
        if self.trainStepCount % 50 == 0:
            newWindowSize = self.getADWINWindowSize(self._windowBinaryValues)
            if newWindowSize < len(self._windowSamples):
                difference = len(self._windowSamples) - newWindowSize
                print newWindowSize, len(self._windowSamples), difference
                self._windowSamples = np.delete(self._windowSamples, [range(difference)], 0)
                self._windowSamplesLabels = np.delete(self._windowSamplesLabels, [range(difference)], 0)
                self._windowDistanceMatrix = np.delete(self._windowDistanceMatrix, [range(difference)], 0)
                self._windowBinaryValues = np.delete(self._windowBinaryValues, [range(difference)], 0)
                print len(self._windowSamples)'''
        return predictedLabel

    def getComplexity(self):
        return len(self._prototypes)


    def initProtosFromSamples(self, TrainSamples, TrainLabels, numProtoPerLabel=1, shuffle=False, cluster=False):
        if self.dataDimensions is None:
            self.dataDimensions = TrainSamples.shape[1];
            self._prototypes = np.empty(shape=(0, self.dataDimensions))
            self._windowSamples = np.empty(shape=(0, self.dataDimensions))
            self._initMetricWeights()

        for i in np.unique(TrainLabels):
            protoCandidates = TrainSamples[TrainLabels == i]
            if cluster:
                km = KMeans(n_clusters=numProtoPerLabel, n_init=1)
                km.fit(protoCandidates)
                protoCandidates = km.cluster_centers_
            else:
                if shuffle:
                    shuffledIndices = np.random.permutation(len(protoCandidates))
                    protoCandidates = protoCandidates[shuffledIndices]

            for j in range(numProtoPerLabel):
                if len(protoCandidates) > j:
                    self.addPrototype(protoCandidates[j, :], i)
                else:
                    break

    def getConfigurationStr(self):
        result = 'learnRateInitial ' + str(self.learnRateInitial) + '\n'
        result += 'learnRateAnnealingSteps ' + str(self.learnRateAnnealingSteps) + '\n'
        result += 'learnRatePerProto ' + str(self.learnRatePerProto) + '\n'

        result += 'activFct ' + str(self.activFct) + '\n'
        result += 'logisticFactor ' + str(self.logisticFactor) + '\n'
        result += 'windowSize ' + str(self.windowSize) + '\n'
        result += 'insertionTimingThresh ' + str(self.insertionTimingThresh) + '\n'

        result += 'retrainFreq ' + str(self.retrainFreq) + '\n'
        result += 'insertionStrategy ' + str(self.insertionStrategy) + '\n'
        result += 'insertionTiming ' + str(self.insertionTiming) + '\n'
        result += 'sampling ' + str(self.sampling) + '\n'
        result += 'deletionStrat ' + str(self.deletionStrategy) + '\n'
        return result

    def getDetailedClassRateByMatrix(self, distMatrix):
        result = ''

        for i in np.unique(self._prototypesLabels):
            numTotal = len(np.where(self._prototypesLabels[distMatrix[:, 0].astype(int)] == i)[0])
            numCorrect = len(
                np.where((self._prototypesLabels[distMatrix[:, 0].astype(int)] == i) &
                         (distMatrix[:, 1] < distMatrix[:, 3]))[0])
            numIncorrect = numTotal - numCorrect
            result += 'class ' + str(i) + ' acc %0.3f \n' % (numCorrect/ float(numTotal))
            for j in np.unique(self._prototypesLabels):
                numConfused = len(np.where((self._prototypesLabels[distMatrix[:, 0].astype(int)] == i) &
                                           (distMatrix[:, 3] <= distMatrix[:, 1]) &
                                           (self._prototypesLabels[distMatrix[:, 2].astype(int)] == j))[0])
                if numConfused > 0:
                    result += '     confused with class ' + str(j) + ' %0.3f ' % (numConfused / float(numIncorrect)) + '(' + str(numConfused) + ')' + '\n'
        return result

    def getNumOfProtosPerClass(self):
        protoDistribution = []
        for label in np.unique(self._prototypesLabels):
            protoDistribution.append([label, len(np.where(self._prototypesLabels == label)[0])])
        return protoDistribution

    def getProtoDistribution(self):
        result = np.empty(shape=(0, 1))
        for i in np.unique(self._prototypesLabels):
            result = np.append(result, len(np.where(self._prototypesLabels == i)[0]))
        return result

    def getConfusionStats(self, samples, labels):
        distMatrix = DistanceMatrix.getDistanceMatrix(samples, labels, self._prototypes, self._prototypesLabels, self.activFct, self.logisticFactor, self._getDistance)
        statList = []
        for i in np.unique(self._prototypesLabels):
            statList.append([i, len(np.where(self._prototypesLabels == i)[0])])
        statList = sorted(statList, key=lambda item: item[1], reverse=True)
        for stat in statList:
            label = stat[0]
            numTotal = len(np.where(self._prototypesLabels[distMatrix[:, 0].astype(int)] == label)[0])
            numCorrect = len(
                np.where((self._prototypesLabels[distMatrix[:, 0].astype(int)] == label) & (distMatrix[:, 4] > 0))[0])
            stat.append(numCorrect / float(numTotal))
            confusedList = []
            for j in np.unique(self._prototypesLabels):
                numLabelConfused = len(np.where((self._prototypesLabels[distMatrix[:, 0].astype(int)] == label) &
                                                (distMatrix[:, 3] <= distMatrix[:, 1]) &
                                                (self._prototypesLabels[distMatrix[:, 2].astype(int)] == j))[0])
                numLabelConfused += len(np.where((self._prototypesLabels[distMatrix[:, 0].astype(int)] == j) &
                                                 (distMatrix[:, 3] <= distMatrix[:, 1]) &
                                                 (self._prototypesLabels[distMatrix[:, 2].astype(int)] == label))[0])
                if numLabelConfused > 0:
                    confusedList.append([j, numLabelConfused])
            confusedList = sorted(confusedList, key=lambda item: item[1], reverse=True)
            stat.append(confusedList)
        return statList

    def getInfos(self):
        result = ''
        result += 'Protos ' + str(self.getComplexity()) + '\n'
        result += 'TriedInsertions ' + str(self._insStatistics._triedInsertionCount) + '\n'
        '''result += 'Insertions ' + str(self._insertionCount) + '\n'
        result += 'avgInsDelta ' + "{0:.3f}".format(self.avgInsertionWindowDeltaCost) + '\n'
        result += 'avgInsProtoDens ' + "{0:.3f}".format(self.avgInsertionWindowPrototypeDensity) + '\n'
        result += 'avgInsProtos ' + "{0:.3f}".format(self.avgInsertionWindowPrototypeCount) + '\n'
        result += 'Deleted Nodes ' + str(self._deletedProtoCount) + '\n'
        result += 'DeleteDelta ' + "{0:.3f}".format(self._totalDeletionWindowDeltaCost / max(self._deletedProtoCount, 1)) + '\n'
        result += 'SingleSample Protos ' + str(len(np.where(self._protoStatistics[:, 0] == 1)[0]))'''

        return result

    def _deleteProtoIdx(self, protoIdx):
        self.protoLabelCounts[self._prototypesLabels[protoIdx]] -= 1
        self._windowDistanceMatrix, self._protoStatistics = DistanceMatrix.deleteProtoFromDistanceMatrix(self._windowDistanceMatrix,
                                                                                  self._windowSamples,
                                                                                  self._windowSamplesLabels,
                                                                                  self._prototypes,
                                                                                  self._prototypesLabels,
                                                                                  protoIdx,
                                                                                  self._getDistance,
                                                                                  protoStatistics=self._protoStatistics)
        self._prototypes = np.delete(self._prototypes, protoIdx, 0)
        self._prototypesLabels = np.delete(self._prototypesLabels, protoIdx, 0)
        self._deletedProtoCount += 1

    def getMissClassifiedIndices(self, samples, sampleLabels):
        classMatrix = DistanceMatrix.getDistanceMatrix(samples, sampleLabels, self._prototypes, self._prototypesLabels, self.activFct, self.logisticFactor, self._getDistance)
        indices = np.where(classMatrix[:, 3] <= classMatrix[:, 1])[0]

        confusedClasses = self._prototypesLabels[classMatrix[indices, 2].astype(int)]
        return indices, confusedClasses

    def _insertProtos(self, protoCandidates, candidateLabels):
        for candidate, label in zip(protoCandidates, candidateLabels):
            if not LVQCommon.doesProtoExist(candidate, self._prototypes):
                self.addPrototype(candidate, label)

    def removeProtosAfterTraining(self, trainSamples, trainLabels):
        distMatrix = DistanceMatrix.getDistanceMatrix(trainSamples, trainLabels, self._prototypes, self._prototypesLabels, self.activFct, self.logisticFactor, self._getDistance)
        deleted = 1
        while deleted > 0:
            deleted = self._removeWorstPrototype(trainSamples, trainLabels, distMatrix)
            distMatrix = DistanceMatrix.getDistanceMatrix(trainSamples, trainLabels, self._prototypes, self._prototypesLabels, self.activFct, self.logisticFactor, self._getDistance)

    def _removeProtoIndices(self, indices):
        deletedProtos = np.empty(shape=(0, self.dataDimensions))
        deletedProtoLabels = np.empty(shape=(0, 1))
        indices = np.sort(indices)[::-1]
        for i in indices:
            if self.protoLabelCounts[self._prototypesLabels[i]] > 1:
                self._lastDeletionWindowDeltaCost = self._protoStatistics[i, 5]
                deletedProtos = np.vstack([deletedProtos, self.prototypes[i]])
                deletedProtoLabels = np.append(deletedProtoLabels, self._prototypesLabels[i])
                self._deleteProtoIdx(i)
        if len(deletedProtos) > 0:
            for listener in self.listener:
                listener.onDelPrototypes(self, deletedProtos, deletedProtoLabels)

    def getProtoypeIdxWithMostWrongSamplesForClass(self, distMatrix, classLabel):
        maxIndices = np.array([])
        protoIdx = -1
        for i in range(len(self.prototypes)):
            indices = np.where((distMatrix[:, 3] <= distMatrix[:, 1]) & (distMatrix[:, 2] == i) & (
                self._prototypesLabels[distMatrix[:, 0].astype(int)] == classLabel))[0]
            if len(indices) > len(maxIndices):
                maxIndices = indices
                protoIdx = i
        return protoIdx, maxIndices

        #if len(candidatesLabels) > 0:
        #    print 'adding ', len(candidatesLabels)
        self._insertProtos(candidates, candidatesLabels)
        #self.figRef, self.subplotRef = GLVQPlot.plotAll(self, self._prototypes, self._prototypesLabels, samples=self._windowSamples, samplesLabels=self._windowSamplesLabels,
        #                                                colors=GLVQPlot.getDefColors(), title='after', plotBoundary=False, XRange=GLVQPlot.getDefXRange2(), YRange=GLVQPlot.getDefYRange2())
        #plt.show()