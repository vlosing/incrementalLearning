import copy
import logging

from Base import Paths
from doExperiment import doExperiment
import cProfile



if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    expCfg = {'iterations': 1, 'trainOfflineSVM': False, 'statisticsLevel': 0, 'statisticsRecordIntervall': 50,
              'saveDatasetFigure': False, 'saveFinalNet': False, 'saveInitialNet': False, 'saveNetIntervall': 0,
              'saveProtoHistogram': False, 'visualizeTraining': False, 'dstDirectory': Paths.StatisticsClassificationDir(),
              'exportToOFRFormat': False, 'epochs': 1, 'expName': ''}

    netDef = {'netType': 'GLVQ', 'activFct': 'logistic', 'retrainFrequency': 0,
              'learnRatePerProto': True, 'learnRateInitial': 1, 'learnRateAnnealingSteps': 5000,
              'metricLearnRate': 0.03,
              'windowSize': 200, 'name': 'NoName', 'LIRAMLVQDimensions': '1'}

    netInsertionDef = {'insertionStrategy': 'SamplingCost', 'insertionTiming': 'trainStepCount', 'insertionTimingThresh': 1,
                       'sampling': 'random', 'protoAdds': 1, 'deleteNodes': False, 'deletionTimingThresh': 200,
                       'deletionStrategy': 'statistics'}
    netIndivDef = {}

    defCfg = {'net': netDef, 'insert': netInsertionDef, 'indiv': netIndivDef}

    noTrainCfg = copy.deepcopy(defCfg)
    noTrainCfg['indiv']['name'] = 'no updates'
    noTrainCfg['indiv']['learnRateInitial'] = 0

    SamplingCostCfg = copy.deepcopy(defCfg)
    SamplingCostCfg['indiv']['name'] = 'standard'
    #SamplingCostCfg['indiv']['learnRateInitial'] = 15

    SamplingCostRetrain1Cfg = copy.deepcopy(defCfg)
    SamplingCostRetrain1Cfg['indiv']['name'] = 'WU-1'
    SamplingCostRetrain1Cfg['indiv']['retrainFrequency'] = 1
    SamplingCostRetrain1Cfg['indiv']['learnRateInitial'] = 12

    SamplingCostRetrain5Cfg = copy.deepcopy(defCfg)
    SamplingCostRetrain5Cfg['indiv']['name'] = 'WU-5'
    SamplingCostRetrain5Cfg['indiv']['retrainFrequency'] = 5
    #SamplingCostRetrain5Cfg['indiv']['learnRateInitial'] = 8

    SamplingCostRetrain10Cfg = copy.deepcopy(defCfg)
    SamplingCostRetrain10Cfg['indiv']['name'] = 'WU-10'
    SamplingCostRetrain10Cfg['indiv']['retrainFrequency'] = 10
    SamplingCostRetrain10Cfg['indiv']['learnRateInitial'] = 6

    SamplingCostRetrain20Cfg = copy.deepcopy(defCfg)
    SamplingCostRetrain20Cfg['indiv']['name'] = 'WU-20'
    SamplingCostRetrain20Cfg['indiv']['retrainFrequency'] = 20
    SamplingCostRetrain20Cfg['indiv']['learnRateInitial'] = 5

    '''window100Cfg = copy.deepcopy(defCfg)
    window100Cfg['indiv']['name'] = 'window-100'
    window100Cfg['indiv']['retrainFrequency'] = 5
    window100Cfg['indiv']['windowSize'] = 100

    window500Cfg = copy.deepcopy(defCfg)
    window500Cfg['indiv']['name'] = 'window-500'
    window500Cfg['indiv']['retrainFrequency'] = 5
    window500Cfg['indiv']['windowSize'] = 500

    window1000Cfg = copy.deepcopy(defCfg)
    window1000Cfg['indiv']['name'] = 'window-1000'
    window1000Cfg['indiv']['retrainFrequency'] = 5
    window1000Cfg['indiv']['windowSize'] = 1000

    window2000Cfg = copy.deepcopy(defCfg)
    window2000Cfg['indiv']['name'] = 'window-2000'
    window2000Cfg['indiv']['retrainFrequency'] = 5
    window2000Cfg['indiv']['windowSize'] = 2000

    window4000Cfg = copy.deepcopy(defCfg)
    window4000Cfg['indiv']['name'] = 'window-4000'
    window4000Cfg['indiv']['retrainFrequency'] = 5
    window4000Cfg['indiv']['windowSize'] = 4000

    window10000Cfg = copy.deepcopy(defCfg)
    window10000Cfg['indiv']['name'] = 'window-10000'
    window10000Cfg['indiv']['retrainFrequency'] = 5
    window10000Cfg['indiv']['windowSize'] = 10000'''

    #cfgs = [noTrainCfg, SamplingCostCfg, SamplingCostRetrain1Cfg, SamplingCostRetrain5Cfg, SamplingCostRetrain10Cfg, SamplingCostRetrain20Cfg]
    cfgs = [SamplingCostRetrain5Cfg]
    #cfgs = [noTrainCfg, SamplingCostCfg, window100Cfg, window500Cfg, window1000Cfg, window2000Cfg, window4000Cfg, window10000Cfg]

    #doExperiment({'dsName': 'Gaussian', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'orderedByLabel', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 0, 'trainSetSize': 0.7, 'insertionTimingThresh': 10000, 'learnRateInitial': 8, 'expName': 'Gaussian_orderedByLabel_6Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'Gaussian', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'orderedByLabel', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 50, 'trainSetSize': 0.7, 'insertionTimingThresh': 680, 'learnRateInitial': 8, 'expName': 'Gaussian_orderedByLabel_12Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'Gaussian', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'orderedByLabel', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 50, 'trainSetSize': 0.7, 'insertionTimingThresh': 220, 'learnRateInitial': 8, 'expName': 'Gaussian_orderedByLabel_24Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'Gaussian', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'random', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 50, 'trainSetSize': 0.7, 'insertionTimingThresh': 10000, 'learnRateInitial': 8, 'expName': 'Gaussian_random_6Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'Gaussian', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'random', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 50, 'trainSetSize': 0.7, 'insertionTimingThresh': 800, 'learnRateInitial': 8, 'expName': 'Gaussian_random_12Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'Gaussian', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'random', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 50, 'trainSetSize': 0.7, 'insertionTimingThresh': 260, 'learnRateInitial': 8, 'expName': 'Gaussian_random_24Protos'}, expCfg, cfgs)

    #doExperiment({'dsName': 'Gaussian', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'chunksRandom', 'stratified': True,
    #          'shuffle': True, 'chunkSize': 100, 'trainSetSize': 0.7, 'insertionTimingThresh': 1000000, 'learnRateInitial': 8, 'expName': 'Gaussian_chunksRandom100_6Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'Gaussian', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'chunksRandom', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 50, 'trainSetSize': 0.7, 'insertionTimingThresh': 1000000, 'learnRateInitial': 8, 'expName': 'Gaussian_chunksRandom50_6Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'Gaussian', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'chunksRandom', 'stratified': True,
    #          'shuffle': True, 'chunkSize': 25, 'trainSetSize': 0.7, 'insertionTimingThresh': 1000000, 'learnRateInitial': 8, 'expName': 'Gaussian_chunksRandom25_6Protos'}, expCfg, cfgs)



    #doExperiment({'dsName': 'USPS', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'random', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 50, 'trainSetSize': 0.7, 'insertionTimingThresh': 10000,
    #              'learnRateInitial': 90, 'expName': 'USPS_random_10Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'USPS', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'random', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 50, 'trainSetSize': 0.7, 'insertionTimingThresh': 370,
    #             'learnRateInitial': 90, 'expName': 'USPS_random_30Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'USPS', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'random', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 50, 'trainSetSize': 0.7, 'insertionTimingThresh': 148,
    #              'learnRateInitial': 90, 'expName': 'USPS_random_60Protos'}, expCfg, cfgs)







    #doExperiment({'dsName': 'USPS', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'orderedByLabel', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 1, 'trainSetSize': 0.7, 'insertionTimingThresh': 10000,
    #              'learnRateInitial': 90, 'expName': 'USPS_orderedByLabel_10Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'USPS', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'orderedByLabel', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 1, 'trainSetSize': 0.7, 'insertionTimingThresh': 350,
    #              'learnRateInitial': 90, 'expName': 'USPS_orderedByLabel_30Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'USPS', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'orderedByLabel', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 1, 'trainSetSize': 0.7, 'insertionTimingThresh': 130,
    #              'learnRateInitial': 90, 'expName': 'USPS_orderedByLabel_60Protos'}, expCfg, cfgs)'''

    #doExperiment({'dsName': 'USPS', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'chunksRandom', 'stratified': True,
    #          'shuffle': True, 'chunkSize': 100, 'trainSetSize': 0.7, 'insertionTimingThresh': 1000000, 'learnRateInitial': 90, 'expName': 'USPS_chunksRandom100_6Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'USPS', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'chunksRandom', 'stratified': True,
    #              'shuffle': True, 'chunkSize': 50, 'trainSetSize': 0.7, 'insertionTimingThresh': 1000000, 'learnRateInitial': 90, 'expName': 'USPS_chunksRandom50_6Protos'}, expCfg, cfgs)
    #doExperiment({'dsName': 'USPS', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'chunksRandom', 'stratified': True,
    #          'shuffle': True, 'chunkSize': 25, 'trainSetSize': 0.7, 'insertionTimingThresh': 1000000, 'learnRateInitial': 90, 'expName': 'USPS_chunksRandom25_6Protos'}, expCfg, cfgs)


    doExperiment({'dsName': 'USPS', 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'random', 'stratified': True,
              'shuffle': True, 'chunkSize': 100, 'trainSetSize': 0.7, 'insertionTimingThresh': 7, 'learnRateInitial': 90}, expCfg, cfgs)


