import numpy as np

from LVQClassifier import LVQClassifier
from LVQCommon import  LVQCommon

class GLVQ(LVQClassifier):
    def __init__(self, activFct='linear', logisticFactor=10, learnRateInitial=1.0,
                 learnRateAnnealingSteps=0, protoAdds=1, learnRatePerProto=False, windowSize=200,
                 retrainFreq=1,
                 insertionStrategy='SamplingCost', insertionTiming='errorCount', insertionTimingThresh=1, sampling='random',
                 deletionStrategy=None, driftStrategy=None, listener=np.array([])):
        super(GLVQ, self).__init__(activFct=activFct, logisticFactor=logisticFactor, learnRateInitial=learnRateInitial,
                                   learnRateAnnealingSteps=learnRateAnnealingSteps, protoAdds=protoAdds,
                                   learnRatePerProto=learnRatePerProto, windowSize=windowSize,
                                   retrainFreq=retrainFreq,
                                   insertionStrategy=insertionStrategy, insertionTiming=insertionTiming, insertionTimingThresh=insertionTimingThresh, sampling=sampling,
                                   deletionStrategy=deletionStrategy, driftStrategy=driftStrategy,
                                   listener=listener)

    def _getDistance(self, A, B):
        return LVQCommon.getSquaredDistance(A, B)

    def _adaptPrototype(self, protoIdx, delta, factor, learnRate, activFctDeriv):
        self._prototypes[protoIdx] = self._prototypes[protoIdx] + learnRate * activFctDeriv * factor * delta

    def _adaptMetric(self, deltaWinner, deltaLooser, factorWinner, factorLooser, activFctDeriv):
        pass

    def _initMetricWeights(self):
        pass

    def getConfigurationStr(self):
        str = super(GLVQ, self).getConfigurationStr()
        return 'netType ' + 'GLVQ' + '\n' + str

    def getComplexityNumParameterMetric(self):
        return len(self._prototypes) * (self.dataDimensions + 1)