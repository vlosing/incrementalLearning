import numpy as np

from LVQClassifier import LVQClassifier
from LVQCommon import LVQCommon

class GMLVQ(LVQClassifier):
    def __init__(self, activFct='linear', logisticFactor=10, learnRateInitial=1.0,
                 learnRateAnnealingSteps=0, protoAdds=1, learnRatePerProto=False, windowSize=200,
                 retrainFreq=1,
                 insertionStrategy='SamplingCost', insertionTiming='errorCount', insertionTimingThresh=1, sampling='random',
                 deletionStrategy=None, driftStrategy=None, listener=np.array([]), metricLearnRate=0):
        self.metricLearnRate = metricLearnRate
        super(GMLVQ, self).__init__(activFct=activFct, logisticFactor=logisticFactor,
                                    learnRateInitial=learnRateInitial,
                                    learnRateAnnealingSteps=learnRateAnnealingSteps, protoAdds=protoAdds,
                                    learnRatePerProto=learnRatePerProto, windowSize=windowSize,
                                    retrainFreq=retrainFreq,
                                    insertionStrategy=insertionStrategy, insertionTiming=insertionTiming, insertionTimingThresh=insertionTimingThresh, sampling=sampling,
                                    deletionStrategy=deletionStrategy, driftStrategy=driftStrategy, listener=listener)


    def _getDistance(self, A, B):
        return LVQCommon.getMetricDistance(A, B, self._omegaMetricWeights)

    def _adaptPrototype(self, protoIdx, delta, factor, learnRate, activFctDeriv):
        self._prototypes[protoIdx] = self._prototypes[protoIdx] + learnRate * activFctDeriv * factor * np.dot(
            self._metricWeights, delta)

    def _adaptMetric(self, deltaWinner, deltaLooser, factorWinner, factorLooser, activFctDeriv):
        omegaMetricWeightsOld = self._omegaMetricWeights.copy()
        deltaWinner = np.atleast_2d(deltaWinner)
        deltaLooser = np.atleast_2d(deltaLooser)
        res1 = np.dot(omegaMetricWeightsOld, deltaWinner.T)
        res2 = np.dot(omegaMetricWeightsOld, deltaLooser.T)
        res11 = np.dot(res1, deltaWinner)
        res22 = np.dot(res2, deltaLooser)
        f1 = factorWinner * (res11 + res11.T)
        f2 = factorLooser * (res22 + res22.T)
        self._omegaMetricWeights -= self.metricLearnRate * activFctDeriv * (f1 - f2)
        self._normalizeOmegaMetricWeights()
        self._refreshMetricWeights()

    def _initMetricWeights(self):
        self._metricWeights = np.empty(shape=(self.dataDimensions, self.dataDimensions))
        self._omegaMetricWeights = np.eye(self.dataDimensions)
        self._normalizeOmegaMetricWeights()
        self._refreshMetricWeights()

    def getConfigurationStr(self):
        baseConfiguration = super(GMLVQ, self).getConfigurationStr()
        return 'netType ' + 'GMLVQ' + '\n' + \
               'metriclearnRate ' + str(self.metricLearnRate) + '\n' + baseConfiguration

    def getComplexityNumParameterMetric(self):
        return len(self._prototypes) * (self.dataDimensions + 1) + self.dataDimensions**2

