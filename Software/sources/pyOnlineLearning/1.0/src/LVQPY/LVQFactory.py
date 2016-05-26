__author__ = 'vlosing'
import numpy as np
from GLVQ import GLVQ
from GMLVQ import GMLVQ
from LIRAMLVQ import LIRAMLVQ

def getLVQClassifier(netType='GLVQ', LIRAMLVQDimensions=1, activFct='linear', logisticFactor=10,
                     learnRateInitial=1.0,
                     learnRateAnnealingSteps=0, protoAdds=1, learnRatePerProto=False, windowSize=200,
                     metricLearnRate=0.003,
                     retrainFreq=0,
                     insertionStrategy='SamplingCost', insertionTiming = 'errorCount', insertionTimingThresh=1, sampling='random',
                     deletionStrategy=None, driftStrategy=None, listener=np.array([])):
    if netType == 'GLVQ':
        return GLVQ(activFct=activFct, logisticFactor=logisticFactor, learnRateInitial=learnRateInitial,
                    learnRateAnnealingSteps=learnRateAnnealingSteps, protoAdds=protoAdds,
                    learnRatePerProto=learnRatePerProto, windowSize=windowSize,
                    retrainFreq=retrainFreq,
                    insertionStrategy=insertionStrategy, insertionTiming=insertionTiming, insertionTimingThresh=insertionTimingThresh, sampling=sampling,
                    deletionStrategy=deletionStrategy, driftStrategy=driftStrategy, listener=listener)
    elif netType == 'GMLVQ':
        return GMLVQ(activFct=activFct, logisticFactor=logisticFactor, learnRateInitial=learnRateInitial,
                     learnRateAnnealingSteps=learnRateAnnealingSteps, protoAdds=protoAdds,
                     learnRatePerProto=learnRatePerProto, windowSize=windowSize,
                     retrainFreq=retrainFreq,
                     insertionStrategy=insertionStrategy, insertionTiming=insertionTiming, insertionTimingThresh=insertionTimingThresh,sampling=sampling,
                     deletionStrategy=deletionStrategy, driftStrategy=driftStrategy, listener=listener, metricLearnRate=metricLearnRate)
    elif netType == 'LIRAMLVQ':
        return LIRAMLVQ(activFct=activFct, logisticFactor=logisticFactor, learnRateInitial=learnRateInitial,
                        learnRateAnnealingSteps=learnRateAnnealingSteps, protoAdds=protoAdds,
                        learnRatePerProto=learnRatePerProto, windowSize=windowSize,
                        insertionStrategy=insertionStrategy, insertionTiming=insertionTiming, insertionTimingThresh=insertionTimingThresh,sampling=sampling,
                        deletionStrategy=deletionStrategy, driftStrategy=driftStrategy, listener=listener, metricLearnRate=metricLearnRate,
                        LIRAMLVQDimensions=LIRAMLVQDimensions)

def getLVQClassifierByCfg(cfg, listener=[], logisticFactor=10):
    return getLVQClassifier(netType=cfg['netType'],
                LIRAMLVQDimensions=cfg['LIRAMLVQDimensions'],
                activFct=cfg['activFct'],
                learnRateInitial=cfg['learnRateInitial'],
                learnRateAnnealingSteps=cfg['learnRateAnnealingSteps'],
                protoAdds=cfg['protoAdds'], learnRatePerProto=cfg['learnRatePerProto'],
                metricLearnRate=cfg['metricLearnRate'], windowSize=cfg['windowSize'],
                retrainFreq=cfg['retrainFreq'],
                insertionStrategy=cfg['insertionStrategy'], insertionTiming=cfg['insertionTiming'], insertionTimingThresh=cfg['insertionTimingThresh'], sampling=cfg['sampling'],
                deletionStrategy=cfg['deletionStrategy'], driftStrategy=cfg['driftStrategy'],
                logisticFactor=logisticFactor,
                listener=listener)




