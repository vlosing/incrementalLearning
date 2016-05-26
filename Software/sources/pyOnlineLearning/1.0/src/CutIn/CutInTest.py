__author__ = 'vlosing'

import logging
import copy
from DataGeneration import DataSetFactory
from Experiment.doExperiment import doExperiment
from Base import Paths
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
import CutInCommon
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def pltROCCurve(allConfidences, orgLabels, classierNames):
    legendPlotList = []
    plotList = range(len(allConfidences))
    for i in range(len(allConfidences)):
        confidences = allConfidences[i]
    #for confidences in allConfidences:
        fpr, tpr, thresholds = metrics.roc_curve(orgLabels, confidences, pos_label=1)
        logging.info('auc-Test' + str(metrics.auc(fpr, tpr)))
        plotList[i] = plt.plot(fpr, tpr)
        plt.xlim([0, 0.2])
        legendPlotList.append(plotList[i][0])
    plt.legend(legendPlotList, classierNames, loc=4)
    plt.show()


def performTest(vehicleType, trainStreams, trainSetNames,
                testHorizon, testMaxTTCPredPos, testMaxTTCPredNeg, testNegSamplesRatio, testStreams, cfgs):
    allConfidences = []
    classierNames = []
    for i in range(len(cfgs)):
        classifiers = doExperiment({'dsName': trainSetNames[i], 'streams': trainStreams, 'splitType': 'simple', 'folds': 5, 'trainOrder': 'random', 'stratified': False,
                                'shuffle': False, 'chunkSize': 100, 'trainSetSize': 0.99}, expCfg, [cfgs[i]])
        classifier = classifiers[cfgs[i]['classifier']['name']][0]
        #datasetTrain = DataSetFactory.getDataSet(trainSetName, trainStreams)

        #scaler = StandardScaler().fit(datasetTrain.samples)
        #trainSamples = scaler.transform(datasetTrain.samples)
        #trainSamples = datasetTrain.samples
        #logging.info('Accuracy Train-Set ' + str(classifier.getAccuracy(trainSamples, datasetTrain.labels, scoreType='acc', labelSpecific=True)))
        #logging.info('LogLoss Train-Set ' + str(classifier.getAccuracy(trainSamples, datasetTrain.labels, scoreType='logLoss')))
        testSetName = CutInCommon.getTrainSetName(vehicleType, testStreams, testHorizon, testMaxTTCPredPos, testMaxTTCPredNeg, testNegSamplesRatio)

        #datasetTest = DataSetFactory.getDataSet(testSetName, testStreams)

        #testSamples = scaler.transform(datasetTest.samples)
        #testSamples = datasetTest.samples
        #logging.info('Label distribution Test' + str(datasetTest.getLabelDistributions()))
        #logging.info('Label distribution RealTest' + str(datasetCutInRealTest.getLabelDistributions()))

        #logging.info('Accuracy Test-Set ' + str(classifier.getAccuracy(testSamples, datasetTest.labels, scoreType='acc', labelSpecific=True)))
        #logging.info('LogLoss Test-Set ' + str(classifier.getAccuracy(testSamples, datasetTest.labels, scoreType='logLoss')))

        #logging.info('Accuracy RealTest-Set ' + str(classifier.getAccuracy(realTestSamples, datasetCutInRealTest.labels, scoreType='acc', labelSpecific=True)))
        #logging.info('LogLoss RealTest-Set ' + str(classifier.getAccuracy(realTestSamples, datasetCutInRealTest.labels, scoreType='logLoss')))

        #confidences = np.atleast_2d(classifier.predict_confidence2Class(trainSamples)).T
        #allConfidences.append(confidences)

        datasetCutInRealTest = DataSetFactory.getDataSet('CutInTest', testStreams)
        realTestSamples = datasetCutInRealTest.samples
        confidences = np.atleast_2d(classifier.predict_confidence2Class(realTestSamples)).T
        allConfidences.append(confidences)
        classierNames.append(cfgs[i]['classifier']['name'])

    allConfidences.append(np.atleast_2d(np.loadtxt('orfEvalCutInTest.txt')).T)
    classierNames.append('on-line RF')
    allConfidences.append(np.atleast_2d(np.loadtxt('OldCutInConfidences.txt')).T)
    classierNames.append('baseline')
    pltROCCurve(allConfidences, datasetCutInRealTest.labels, classierNames)




    #logging.info('Accuracy Test-Set Baseline' + str(np.sum(np.zeros(shape=(len(datasetTest.labels))) == datasetTest.labels)/float(len(datasetTest.labels))))


    '''clf = CalibratedClassifierCV(base_estimator=classifiers['RF'][0].getClassifier(), method='sigmoid', cv=2)
    clf.fit(datasetTrain2.samples, datasetTrain2.labels)

    yPredProba = clf.predict_proba(datasetTest.samples)
    yPred = clf.predict(datasetTest.samples)

    logging.info('after calibration')
    logging.info('Accuracy Test-Set ' + str(accuracy_score(datasetTest.labels, yPred)))
    logging.info('LogLoss Test-Set ' + str(log_loss(datasetTest.labels, yPredProba)))'''

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    expCfg = {'iterations': 1, 'trainOfflineSVM': False, 'statisticsLevel': 0, 'statisticsRecordIntervall': 50,
              'saveDatasetFigure': False, 'saveFinalNet': False, 'saveInitialNet': False, 'saveNetIntervall': 0,
              'saveProtoHistogram': False, 'visualizeTraining': False, 'dstDirectory': Paths.StatisticsClassificationDir(),
              'exportToOFRFormat': False, 'epochs': 1, 'expName': ''}

    LVQDefault = {'classifierType': 'LVQ', 'netType': 'GLVQ', 'activFct': 'linear', 'retrainFrequency': 0,
                 'learnRatePerProto': True, 'learnRateInitial': 1, 'learnRateAnnealingSteps': 1000,
                 'metricLearnRate': 0.03,
                 'windowSize': 1000, 'name': 'NoName', 'LIRAMLVQDimensions': '1',
                 'insertionStrategy': 'SamplingCost', 'insertionTiming': 'errorCount', 'insertionTimingThresh': 0,
                 'sampling': 'random', 'protoAdds': 1, 'deleteNodes': False, 'deletionTimingThresh': 200,
                 'deletionStrategy': 'statistics'}

    RFDefault = {'classifierType': 'RF', 'numEstimators': 100, 'max_depth': 10, 'max_features': 'auto', 'criterion': 'gini', 'name': 'off-line RF'}
    SVMDefault = {'classifierType': 'SVM', 'C': 1000, 'gamma': 0.001, 'name': 'NoName'}

    netIndivDef = {}

    defaultCfg = {'classifier': RFDefault, 'indiv': netIndivDef}

    rfCfg = copy.deepcopy(defaultCfg)

    rfCfg2 = copy.deepcopy(defaultCfg)
    rfCfg2['classifier']['numEstimators'] = 1000
    rfCfg2['classifier']['max_features'] = 2
    rfCfg2['classifier']['criterion'] = 'entropy'
    rfCfg2['classifier']['max_depth'] = None

    rfCfg3 = copy.deepcopy(defaultCfg)
    rfCfg3['classifier']['numEstimators'] = 1000
    rfCfg3['classifier']['criterion'] = 'entropy'
    rfCfg3['classifier']['max_depth'] = None


    cfg1 = copy.deepcopy(defaultCfg)
    cfg1['indiv']['learnRateInitial'] = 0
    cfg1['indiv']['netType'] = 'GLVQ'

    #cfg2 = copy.deepcopy(defaultCfg)
    #cfg2['indiv']['learnRateInitial'] = 2
    #cfg2['indiv']['netType'] = 'GMLVQ'

    '''cfg3 = copy.deepcopy(defaultCfg)
    cfg3['indiv']['learnRateInitial'] = 3
    cfg3['indiv']['netType'] = 'GMLVQ'

    cfg4 = copy.deepcopy(defaultCfg)
    cfg4['indiv']['learnRateInitial'] = 4
    cfg4['indiv']['netType'] = 'GMLVQ'

    cfg5 = copy.deepcopy(defaultCfg)
    cfg5['indiv']['learnRateInitial'] = 5
    cfg5['indiv']['netType'] = 'GMLVQ'''

    LVQCfg = copy.deepcopy(defaultCfg)
    LVQCfg['classifier'] = copy.deepcopy(LVQDefault)
    LVQCfg['indiv']['learnRateInitial'] = 10
    LVQCfg['indiv']['netType'] = 'GMLVQ'
    LVQCfg['indiv']['metricLearnRate'] = 0.1
    LVQCfg['classifier']['name'] = 'off-line GMLVQ'

    LVQOnlCfg = copy.deepcopy(defaultCfg)
    LVQOnlCfg['classifier'] = copy.deepcopy(LVQDefault)
    LVQOnlCfg['indiv']['learnRateInitial'] = 10
    LVQOnlCfg['indiv']['netType'] = 'GMLVQ'
    LVQOnlCfg['indiv']['metricLearnRate'] = 0.1
    LVQOnlCfg['indiv']['insertionTimingThresh'] = 2
    LVQOnlCfg['classifier']['name'] = 'on-line GMLVQ'


    #cfgs = [LVQCfg, LVQCfg]

    allStreams = np.array([1,2,3,4,5,6,7,8,9,10,11,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86])
    print len(allStreams)
    trainStreams = np.array([1,2,3,5,6,7,9,10,11,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44,45,46,48,49,50,52,53,54,56,57,58,60,61,62,64,65,66,68,69,70,72,73,74,76,77,78,80,81,82,84,85,86])
    print len(trainStreams)

    testStreams = np.setdiff1d(allStreams, trainStreams, assume_unique=True)
    #testStreams = np.array([4,8,23,27,31,35])

    vehicleType = '2X_EU'

    trainMaxTTCPredPos = 100
    trainMaxTTCPredNeg = 100
    trainHorizon = 64000
    trainNegSamplesRatio = 1


    rfTrainSetName = CutInCommon.getTrainSetName(vehicleType, trainStreams, trainHorizon, trainMaxTTCPredPos, trainMaxTTCPredNeg, 5)
    rfTrainSetName2 = CutInCommon.getTrainSetName(vehicleType, trainStreams, trainHorizon, trainMaxTTCPredPos, trainMaxTTCPredNeg, 1)
    rfTrainSetName3 = CutInCommon.getTrainSetName(vehicleType, trainStreams, 0, trainMaxTTCPredPos, trainMaxTTCPredNeg, 1)
    lvqTrainSetName  = CutInCommon.getTrainSetName(vehicleType, trainStreams, trainHorizon, trainMaxTTCPredPos, trainMaxTTCPredNeg, 1)
    lvqTrainSetName2  = CutInCommon.getTrainSetName(vehicleType, trainStreams, 0, trainMaxTTCPredPos, trainMaxTTCPredNeg, 1)

    cfgs = [rfCfg, LVQCfg, LVQOnlCfg]
    trainSetNames= [rfTrainSetName, lvqTrainSetName, lvqTrainSetName]
    #trainSetNames= [rfTrainSetName2, rfTrainSetName]
    #trainSetNames= [lvqTrainSetName, lvqTrainSetName2]

    testMaxTTCPredPos = 100
    testMaxTTCPredNeg = 100
    testNegSamplesRatio = 0

    testHorizon = 64000
    performTest(vehicleType, trainStreams, trainSetNames, testHorizon, testMaxTTCPredPos, testMaxTTCPredNeg, testNegSamplesRatio, testStreams, cfgs)

    #performTest(vehicleType, trainHorizon, trainMaxTTCPredPos, trainMaxTTCPredNeg, 100, trainStreams,
    #            testMaxTTCPredPos, testMaxTTCPredNeg, testNegSamplesRatio, testStreams)
    #performTest(vehicleType, trainHorizon, trainMaxTTCPredPos, trainMaxTTCPredNeg, 10000, trainStreams,
    #            testMaxTTCPredPos, testMaxTTCPredNeg, testNegSamplesRatio, testStreams)













