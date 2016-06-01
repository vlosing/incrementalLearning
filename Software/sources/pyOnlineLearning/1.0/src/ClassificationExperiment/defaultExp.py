import copy
import logging

from Base import Paths
from doExperiment import doExperiment
import matplotlib.pyplot as plt
import cProfile
import libNNPythonIntf
import numpy as np
if __name__ == '__main__':
    '''import numpy as np
    #a = np.array([[1.,2.], [3.,4.]])
    a = np.array([1.,2.])
    b = np.array([[2.,2.], [3.,4.], [3.,3.], [1.,1.]])
    labels = np.array([[1, 2, 1, 2]])
    c = np.array([[8, 9, 5, 7, 6, 1, 7], [2, 3, 2, 5, 7, 5, 2]], dtype=np.int32)
    #print c

    distances = libNNPythonIntf.getNToNDistances(a, b)
    predLabels = libNNPythonIntf.getLinearWeightedLabels(labels.astype(np.int32), distances[0])

    #indices = libNNPythonIntf.argSort(c)
    #indices = libNNPythonIntf.nArgMin(2, c)
    #indices = libNNPythonIntf.mostCommon(c)

    print labels, distances, predLabels'''

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    expCfg = {'iterations': 1, 'trainOfflineSVM': False, 'statisticsLevel': 0, 'statisticsRecordIntervall': 50,
              'saveDatasetFigure': False, 'saveFinalNet': False, 'saveInitialNet': False, 'saveNetIntervall': 0,
              'saveProtoHistogram': False, 'visualizeTraining': True, 'dstDirectory': Paths.StatisticsClassificationDir(),
              'exportToOFRFormat': False, 'epochs': 1, 'expName': ''}


    netIndivDef = {}

    defaultCfg = {'classifierName': 'ILVQ', 'indiv': netIndivDef}
    #randomState = 0
    #np.random.seed(randomState)

    '''ILVQCfg = copy.deepcopy(defaultCfg)
    ILVQCfg['indiv']['name'] = 'standard'
    ILVQCfg['indiv']['netType'] = 'GLVQ'
    ILVQCfg['indiv']['windowSize'] = 5000
    #standardModelCfg['indiv']['learnRateInitial'] = 0.000001
    ILVQCfg['indiv']['learnRateInitial'] = 0
    ILVQCfg['indiv']['insertionTimingThresh'] = 5

    ILVQCfg['indiv']['deletionStrategy'] = [None]
    #standardModelCfg['indiv']['deletionStrategy'] = ['obsolete']
    ILVQCfg['indiv']['deletionStrategy'] = ['obsolete', 'accBelowChance']
    #standardModelCfg['indiv']['deletionStrategy'] = 'accBelowChance'
    #standardModelCfg['indiv']['deletionStrategy'] = ['windowAccMax']
    #standardModelCfg['indiv']['deletionStrategy'] = ['obsolete', 'windowAccReg']

    ILVQCfg['indiv']['insertionStrategy'] = None
    #standardModelCfg['indiv']['insertionStrategy'] = 'samplingCost'
    #standardModelCfg['indiv']['insertionStrategy'] = 'samplingAccMax'
    ILVQCfg['indiv']['insertionStrategy'] = 'samplingAccReg'

    ILVQCfg['indiv']['driftStrategy'] = None
    #ILVQCfg['indiv']['driftStrategy'] = 'adwin'
    #standardModelCfg['indiv']['driftStrategy'] = 'maxACC'

    ILVQCfg['indiv']['sampling'] = 'random'
    ILVQCfg['indiv']['protoAdds'] = 1

    #standardModelCfg['indiv']['protoAdds'] = 5
    #standardModelCfg['indiv']['insertionTimingThresh'] = 100

    #cfgs = [noUpdatesCfg, standardModelCfg, retrtainedCfg]'''
    KNNWindowCfg = copy.deepcopy(defaultCfg)
    KNNWindowCfg['classifierName'] = 'KNNWindow'
    KNNWindowCfg['indiv']['name'] = 'KNNWindow'
    KNNWindowCfg['indiv']['windowSize'] = 2500
    #KNNWindowCfg['indiv']['windowSize'] = 1815
    #KNNWindowCfg['indiv']['weights'] = 'uniform'
    KNNWindowCfg['indiv']['weights'] = 'distance'
    KNNWindowCfg['indiv']['driftStrategy'] = None
    #KNNWindowCfg['indiv']['driftStrategy'] = 'adwin'
    #KNNWindowCfg['indiv']['driftStrategy'] = 'maxACC7'
    #KNNWindowCfg['indiv']['driftStrategy'] = 'maxACC8'
    #KNNWindowCfg['indiv']['driftStrategy'] = 'both'

    #WAVGCfg = copy.deepcopy(defaultCfg)
    #WAVGCfg['classifierName'] = 'WAVG'
    #WAVGCfg['indiv']['name'] = 'WAVG'


    cfgs = [KNNWindowCfg]
    plt.ion()

    #doExperiment({'dsName': 'squaresIncrXXL', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'squaresIncr', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'rbfAbruptXXL', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    doExperiment({'dsName': 'rbfAbruptSmall', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'chessVirtual', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'chessVirtualXXL', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)

    #doExperiment({'dsName': 'sea', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'rbfSlow', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'hypSlow', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'allDrift', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'allDriftXXL', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)

    #doExperiment({'dsName': 'weather', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'elec', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'covType', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'outdoorStream', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'rialto', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)

    #doExperiment({'dsName': 'chessIIDXXL', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)
    #doExperiment({'dsName': 'chessFields', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)

    #cProfile.run("doExperiment({'dsName': 'chessVirtualXXL', 'splitType': 'simple', 'folds': 3, 'trainOrder': 'original', 'stratified': False, 'shuffle': False, 'chunkSize': 1, 'trainSetSize': 1}, expCfg, cfgs)")