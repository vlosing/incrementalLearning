__author__ = 'vlosing'
from generateCutInTrainData import  CutInDataGenerator
from HystereseEvaluation import HystereseEvaluation
from Base import Paths
import json
import logging
from Experiment.doExperiment import  doExperiment
from CBPInPhysicalPredictionWriter import CBPInPhysicalPredictionWriter
import os

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    maxTTCPred = 50
    minTTCPred = 0.01
    trainHorizon = 64000
    negSamplesRatio = 10
    minConfidence = 0.7
    os.environ['IACC_VEHICLE_TYPE'] = '2X_EU'

    trainStreamNames = ['Stream001', 'Stream002', 'Stream003', 'Stream004', 'Stream005', 'Stream006', 'Stream007', 'Stream008', 'Stream009']
    trainSetName = CutInDataGenerator.getTrainSetName(os.environ['IACC_VEHICLE_TYPE'], trainStreamNames, trainHorizon, maxTTCPred, negSamplesRatio)

    testStreamNames = ['Stream010', 'Stream011', 'Stream012', 'Stream013', 'Stream014', 'Stream015', 'Stream016', 'Stream017', 'Stream018',
                       'Stream019', 'Stream020', 'Stream021', 'Stream022', 'Stream023', 'Stream024', 'Stream025', 'Stream026', 'Stream027']

    #testStreamNames =  ['Stream021', 'Stream022', 'Stream023', 'Stream024', 'Stream025', 'Stream026', 'Stream027']


    trainPrefixDir = '/hri/localdisk/vlosing/ACC_Data/'
    dstPrefixDir = '/hri/localdisk/vlosing/ACC_Data/'

    os.environ['IACC_WORKING_DIR'] = dstPrefixDir + os.environ['IACC_VEHICLE_TYPE'] + '/'
    hystereseSrcPrefix = '/hri/storage/user/vlosing/HystereseCutIn/' + os.environ['IACC_VEHICLE_TYPE'] + '_'
    hystereseDstPrefix = os.environ['IACC_WORKING_DIR'] + 'Results_c/'

    trainSrcDir = trainPrefixDir+ os.environ['IACC_VEHICLE_TYPE'] + '/Results_c/'
    dataGen = CutInDataGenerator(trainSrcDir, os.environ['IACC_VEHICLE_TYPE'], trainStreamNames, maxTTCPredPos=maxTTCPred, minTTCPred=minTTCPred, seqLength=trainHorizon, negSamplesRatio=negSamplesRatio)
    dataGen.genAndSaveSamples()

    expCfg = {'iterations': 1, 'statisticsLevel': 0, 'statisticsRecordIntervall': 50,
              'saveDatasetFigure': False, 'saveFinalNet': False, 'saveInitialNet': False, 'saveNetIntervall': 0,
              'saveProtoHistogram': False, 'visualizeTraining': False, 'dstDirectory': Paths.StatisticsClassificationDir(),
              'exportToOFRFormat': False, 'epochs': 1, 'expName': ''}

    RFDefault = {'classifierType': 'RF', 'numEstimators': 500, 'max_depth': 10, 'name': 'RF'}

    defaultCfg = {'classifier': RFDefault, 'indiv': {}}

    cfgs = [defaultCfg]

    classifiers = doExperiment({'dsName': trainSetName, 'splitType': 'kFold', 'folds': 5, 'trainOrder': 'random', 'stratified': False,
                                'shuffle': True, 'chunkSize': 100, 'trainSetSize': 0.7, 'insertionTimingThresh': 50, 'learnRateInitial': 0, 'LIRAMLVQDimensions': 13}, expCfg, cfgs)
    classifier = classifiers['RF'][0]

    hystEval = HystereseEvaluation(hystereseSrcPrefix, testStreamNames, classifier, 2, 20, maxTTCPred=maxTTCPred, minTTCPred=minTTCPred, minConfidence=minConfidence)
    hystEval.evaluate()

    cbpWriter = CBPInPhysicalPredictionWriter(hystereseSrcPrefix, hystereseDstPrefix, testStreamNames, classifier, maxTTCPred=maxTTCPred, minTTCPred=minTTCPred)
    cbpWriter.writePredictions()

    os.system('python /home/vlosing/OnlineLearning/Software/sources/ToolBosStuff/trunk/Evaluation/EvalEngine.py -s CBP ' + ' '.join(testStreamNames))


