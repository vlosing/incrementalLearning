from HyperParameter.hyperParamsScaled import getHyperParamsStationaryScaled, getHyperParamsNonStationaryScaled
from HyperParameter.hyperParamsUnscaled import getHyperParamsStationaryUnscaled, getHyperParamsNonStationaryUnscaled
from DataGeneration.DataSetFactory import isStationary
import logging
def getDefaultHyperParams(classifierName):
    params = {}
    params['GNB'] = {}
    params['LPPNSE'] = {}
    params['WAVG'] = {}
    params['DACC'] = {}
    params['LVGB'] = {'splitConfidence': 0.0000001, 'tieThresh': 0.05, 'gracePeriod': 200, 'numClassifier': 10}
    params['HoeffAdwin'] = {'splitConfidence': 0.0000001, 'tieThresh': 0.05, 'gracePeriod': 200}
    params['KNNPaw'] = {'windowSize': 5000}
    params['KNNWindow'] = {'windowSize': 5000, 'nNeighbours': 5, 'weights': 'distance', 'driftStrategy': None}
    params['ILVQ'] = {'classifierType': 'LVQ', 'netType': 'GMLVQ', 'activFct': 'logistic', 'retrainFreq': 0,
         'learnRatePerProto': True, 'learnRateInitial': 1, 'learnRateAnnealingSteps': 5000,
         'metricLearnRate': 0.01,
         'windowSize': 500, 'name': 'NoName', 'LIRAMLVQDimensions': '1',
         'insertionStrategy': 'samplingAccReg', 'insertionTiming': 'errorCount', 'insertionTimingThresh': 30,
         'sampling': 'random', 'protoAdds': 2, 'deletionStrategy': ['obsolete', 'accBelowChance'], 'driftStrategy': 'maxACC'}
    params['ISVM'] = {'C': 2**8, 'sigma': 1, 'kernel': 'RBF', 'windowSize': 500}
    params['ORF'] = {'numTrees': 10, 'numRandomTests': 250, 'maxDepth': 50, 'counterThreshold': 10}
    params['IELM'] = {'numHiddenNeurons': 100}
    params['LPP'] = {'classifierPerChunk': 1}
    params['SGD'] = {'eta0': 1, 'learningRate': 'constant'}
    return params[classifierName]

def getHyperParams(datasetName, classifierName, scaled):
    params = getDefaultHyperParams(classifierName)
    return params
    if classifierName in ['LPPNSE', 'GNB']:
        return params
    stationary = isStationary(datasetName)

    if scaled:
        if stationary:
            datasetSpecificParams = getHyperParamsStationaryScaled()
        else:
            datasetSpecificParams = getHyperParamsNonStationaryScaled()
    else:
        if stationary:
            datasetSpecificParams = getHyperParamsStationaryUnscaled()
        else:
            datasetSpecificParams = getHyperParamsNonStationaryUnscaled()

    if not datasetSpecificParams.has_key(datasetName) or not datasetSpecificParams[datasetName].has_key(classifierName):
        logging.info('No dataset specific hyperparams.')
        specificParams = {}
    else:
        specificParams = datasetSpecificParams[datasetName][classifierName]
    for key in specificParams.keys():
        params[key] = specificParams[key]
    return params
