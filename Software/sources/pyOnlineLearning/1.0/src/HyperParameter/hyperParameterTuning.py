__author__ = 'viktor'
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
import numpy as np
import math
from DataGeneration.TrainTestSplitsManager import TrainTestSplitsManager
from ClassifierComparison.auxiliaryFunctions import trainClassifier
import os

import json
import sys
def objective(x):
    metaParams = x['meta']
    params = metaParams['defaultParameter']
    for key in x['params'].keys():
        params[key] = x['params'][key]
    accuracies = []
    complexities = []
    global splitManager
    chunkSize = 500
    if metaParams['streamSetting']:
        for i in range(len(splitManager.TrainSamplesLst)):
            allPredictedTestLabels, allPredictedTrainLabels, complexitiesTmp, complexityNumParameterMetric = trainClassifier(metaParams['classifierName'], params, splitManager.TrainSamplesLst[i], splitManager.TrainLabelsLst[i], splitManager.TestSamplesLst[i], splitManager.TestLabelsLst[i], chunkSize, streamSetting=metaParams['streamSetting'])
            idx = chunkSize
            accuracies = []
            for classifiedLabels in allPredictedTrainLabels:
                accuracies.append(accuracy_score(splitManager.TrainLabelsLst[i][idx:idx+len(classifiedLabels)], classifiedLabels))
                idx += len(classifiedLabels)
            #print accuracies, np.mean(accuracies)
            complexities.append(complexitiesTmp[-1])


    else:
        if metaParams['classifierName'] == 'LVGB':
            chunkSize = sys.maxint
        for i in range(len(splitManager.TrainSamplesLst)):
            allPredictedTestLabels, allPredictedTrainLabels, complexitiesTmp, complexityNumParameterMetric = trainClassifier(metaParams['classifierName'], params, splitManager.TrainSamplesLst[i], splitManager.TrainLabelsLst[i], splitManager.TestSamplesLst[i], splitManager.TestLabelsLst[i], chunkSize, streamSetting=metaParams['streamSetting'])
            accuracies.append(accuracy_score(splitManager.TestLabelsLst[i], allPredictedTestLabels[-1]))
            complexities.append(complexitiesTmp[-1])
    complexity = np.mean(complexities)
    loss = 1-np.mean(accuracies)
    global trialCounter
    trialCounter += 1
    print trialCounter, x['params'], 'loss', loss, 'complexity', complexity
    return {'loss': loss, 'complexity': complexity, 'evaluatedParams': x['params'], 'allParams': params,  'status': STATUS_OK}


def getSearchSpace(classifierName, defaultParameter, streamSetting):
    if classifierName == 'ILVQ':
        result = {'meta': {'classifierName': 'ILVQ'},
                'params': {
                    #'netType': hp.choice('netType_', ['GLVQ', 'GMLVQ']),
                    #'activFct': hp.choice('activFct_', ['linear', 'logistic']),
                    #'metricLearnRate': hp.loguniform('metricLearnRate', math.log(1e-5), math.log(0.1)),
                    #'learnRateInitial': hp.loguniform('learnRateInitial_', math.log(1e-5), math.log(5000))
                    #'learnRateInitial': hp.loguniform('learnRateInitial_', math.log(5000), math.log(5000000))
                    'insertionTimingThresh': hp.qloguniform('insertionTimingThresh_', math.log(1), math.log(400), 1),
                }}
    elif classifierName == 'ISVM':
        result = {'meta': {'classifierName': 'ISVM'},
                'params': {
                    #'C': hp.quniform('C_', 1, 256, 1),
                    'sigma': hp.loguniform('sigma_', math.log(1e-5), math.log(5000))}}
    elif classifierName == 'ORF':
        result = {'meta': {'classifierName': 'ORF'},
                'params': {
                    #'numTrees': hp.quniform('numTrees_', 10, 50, 1)
                    #'numRandomTests': hp.quniform('numRandomTests_', 100, 300, 1),
                    'counterThreshold': hp.quniform('counterThreshold_', 3, 100, 1)
                }}
    elif classifierName == 'IELM':
        result = {'meta': {'classifierName': 'IELM'},
                'params': {
                    'numHiddenNeurons': hp.quniform('numHiddenNeurons_', 50, 500, 1)}}
    elif classifierName == 'LPP':
        result = {'meta': {'classifierName': 'LPP'},
                'params': {
                    'classifierPerChunk': hp.quniform('classifierPerChunk_', 1, 8, 1)}}
    elif classifierName == 'SGD':
        result = {'meta': {'classifierName': 'SGD'},
                 'params': {
                     'eta0': hp.loguniform('eta0_', math.log(1e-5), math.log(3))}}

    elif classifierName == 'LVGB':
        result = {'meta': {'classifierName': 'LVGB'},
                 'params': {
                     #'gracePeriod': hp.quniform('gracePeriod_', 10, 300, 1)
                     'splitConfidence': hp.loguniform('splitConfidence_', math.log(1e-7), math.log(0.45))
                     #'tieThresh': hp.loguniform('tieThresh_', math.log(0.05), math.log(0.45))
                 }}

    result['meta']['defaultParameter'] = defaultParameter
    result['meta']['streamSetting'] = streamSetting
    return result

def determineHyperParams(classifierName, X, y, dstDirectory, prefix, defaultParameter, streamSetting, scaled, max_evals=250):
    if classifierName in ['GNB', 'LPPNSE']:
        return {}
    elif classifierName in ['LPP', 'LVGB']:
        max_evals = 50
    elif classifierName in ['ORF']:
        max_evals = 100
    trials = Trials()
    global splitManager
    if streamSetting:
        splitManager = TrainTestSplitsManager(X, y, splitType='simple', dataOrder='original', trainSetSize=1, shuffle=False)
    else:
        splitManager = TrainTestSplitsManager(X, y, splitType='kFold', numberOfFolds=3, shuffle=True, stratified=True)

    splitManager.generateSplits()
    global trialCounter
    trialCounter = 0
    fmin(objective,
        space=hp.choice('classifier_type', [getSearchSpace(classifierName, defaultParameter, streamSetting)]),
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials)

    sortedIndices = np.argsort(trials.losses())
    results = []
    for i in sortedIndices:
        results.append([trials.losses()[i], trials.results[i]['complexity'], trials.results[i]['evaluatedParams']])
    output = open(os.path.join(dstDirectory, prefix + '_' + classifierName + '_scaled' + str(scaled) + '.json'), 'wb')
    json.dump(results, output)
    print 'best', trials.losses()[sortedIndices[0]], trials.results[sortedIndices[0]]['complexity'], trials.results[sortedIndices[0]]['evaluatedParams']
    return trials.results[sortedIndices[0]]['allParams']
