__author__ = 'vlosing'
import numpy as np
from Experiment.doExperiment import doExperiment
from Base import Paths
from Base import Serialization
import logging
import time
import CutInCommon
import matplotlib.pyplot as plt
from Visualization import GLVQPlot
import json
import cProfile

class HystereseEvaluation(object):
    GTC_NONE = 0
    GTC_PRED = 1
    GTC_IGNORE = 2
    #TTC_PRED_IDX = 12

    #vehicleID, lastHystereseStart, classifiedCount, lastTimeStampCategory, lastGroundTruthIdx

    def __init__(self, srcPathPrefix, streams, classifier, histOnSet=2, histOffSet=20, maxTTCPred=50, minTTCPred=0.1, minConfidences=[0.5]):
        self.srcPathPrefix = srcPathPrefix
        self.streams = streams
        self.classifier = classifier
        self.histOnSet = histOnSet
        self.histOffSet = histOffSet
        self.maxTTCPred = maxTTCPred
        self.minTTCPred = minTTCPred
        self.minConfidences = minConfidences
        self.allSamples = []
        self.allLabels = []

        self.FP = 0
        self.FN = 0
        self.TP = 0
        self.TN = 0

    def initGroundTruthData(self, groundTruthFileData):
        groundTruthData = np.empty(shape=(0, 6))
        for groundTruth in groundTruthFileData:
            predCase = groundTruth['pred_case']
            startTime = groundTruth['pred_time_start']
            ignoreEndTime = groundTruth['pred_time_stop']
            endTime = groundTruth['time_reference']
            vehicleID = groundTruth['vehicle_id']
            processed = 0
            groundTruthData = np.vstack([groundTruthData, [vehicleID, startTime, endTime, ignoreEndTime, processed, predCase]])
        return groundTruthData

    def getGroundTruthIdxToTimeStamp(self, timeStamp, vehicleID, groundTruthData):
        #np.where((self.groundTruthData[:, 0] == vehicleID) & (timeStamp > self.groundTruthData[:, 1]) & (timeStamp <= self.groundTruthData[:, 3]))
        for i in range(len(groundTruthData)):
            data = groundTruthData[i]
            if vehicleID == data[0] and timeStamp >= data[1] and timeStamp <= data[3]:
                return i
        return None

    def getTimeStampGroundTruthCategory(self, groundTruthIdx, timeStamp, groundTruthData):
        data = groundTruthData[groundTruthIdx]
        if timeStamp >= data[1] and timeStamp <= data[2]:
            return HystereseEvaluation.GTC_PRED
        elif timeStamp >= data[2] and timeStamp <= data[3]:
            return HystereseEvaluation.GTC_IGNORE
        return HystereseEvaluation.GTC_NONE

    def getTimeStampFromFileName(self, fileName):
        fileNameSplit = fileName.split('_')
        timeStamp = int(fileNameSplit[1].split('.')[0])
        return timeStamp

    def evaluate(self):
        tic = time.time()
        allConfidenceEvaluations = []
        numberOfSamples = 0
        for stream in self.streams:
            logging.info(stream)
            streamData = Serialization.loadPickleFile(self.srcPathPrefix + "%03d" % stream + '.pickle')
            groundTruthData = self.initGroundTruthData(streamData[1])
            data, timeStampCounterMapping = self.prepareStreamData(streamData[0], groundTruthData)
            if len(data) > 0:
                numberOfSamples += len(data)
                confidenceEvaluations = self.evaluateStream(data, groundTruthData, timeStampCounterMapping)
                if allConfidenceEvaluations == []:
                    allConfidenceEvaluations = confidenceEvaluations
                else:
                    allConfidenceEvaluations[:, [1, 2, 3]] += confidenceEvaluations[:, [1, 2, 3]]

        #f = open(Paths.CutInTestFeaturesPath(), 'w')
        #json.dump(self.allSamples, f)
        #f.close()
        #f = open(Paths.CutInTestLabelsPath(), 'w')
        #json.dump(self.allLabels, f)
        #f.close()

        logging.info(str(time.time() - tic) + " seconds")
        logging.info(str(numberOfSamples) + " samples")

        return allConfidenceEvaluations

    def prepareStreamData(self, streamData, groundTruthData):
        streamSamples = []
        timeStampCounter = 0
        data = []
        timeStampCounterMapping = []
        for fileData in streamData:
            samples = fileData[1]
            vehicleIds = fileData[2]
            timeStamp = fileData[0]
            timeStampCounter += 1
            timeStampCounterMapping.append([timeStampCounter, timeStamp])
            for sample, vehicleId in zip(samples, vehicleIds):
                if sample[CutInCommon.TTC_PRED_IDX] >= self.minTTCPred and sample[CutInCommon.TTC_PRED_IDX] <= self.maxTTCPred:

                    groundTruthIdx = self.getGroundTruthIdxToTimeStamp(timeStamp, vehicleId, groundTruthData)
                    if groundTruthIdx is None:
                        timeStampCategory = HystereseEvaluation.GTC_NONE
                    else:
                        timeStampCategory = self.getTimeStampGroundTruthCategory(groundTruthIdx, timeStamp, groundTruthData)
                    self.allSamples.append(sample)
                    if timeStampCategory==self.GTC_PRED:
                        self.allLabels.append(1)
                    else:
                        self.allLabels.append(0)
                    data.append([timeStampCounter, timeStamp, vehicleId, groundTruthIdx, timeStampCategory])
                    streamSamples.append(sample)
        data = np.array(data)


        if len(data) > 0:
            confidences = np.atleast_2d(self.classifier.predict_confidence2Class(streamSamples)).T
            data = np.hstack((data, confidences))
        return data, np.array(timeStampCounterMapping)

    def getTimeStampToTimeStampCounter(self, timeStampCounterMapping, timeStampCounter):
        return timeStampCounterMapping[timeStampCounterMapping[:, 0] == timeStampCounter][0][0]

    '''def getPosPredictions(self, streamData):
        uniqueVehicleIds = np.unique(streamData[:, 2])
        posPredictionData = []
        for minConfidence in self.minConfidences:
            posConfidencePredictionData = []
            for vehicleId in uniqueVehicleIds:
                dataIndices = np.where((streamData[:, 2] == vehicleId) & (streamData[:, 5] >= minConfidence))[0]
                #XXVL garuantee that dataIndices are sorted!
                if len(dataIndices) > 0:
                    filteredData = streamData[dataIndices, :]
                    #print filteredData
                    if self.histOnSet > 1:
                        timeStampCounterDiff = np.ediff1d(filteredData[:, 0])
                        consecutivePosPredictions = 0
                        minTimeStampCounter = 0
                        for i in range(len(timeStampCounterDiff)):
                            if filteredData[i][0] >= minTimeStampCounter:
                                if timeStampCounterDiff[i] == 1:
                                    consecutivePosPredictions += 1
                                    if consecutivePosPredictions == self.histOnSet - 1:
                                        #print 'pos Prediction', i,
                                        posConfidencePredictionData.append(filteredData[i + 1])
                                        minTimeStampCounter = filteredData[i + 1][0] + self.histOffSet
                                        consecutivePosPredictions = 0
                                else:
                                    consecutivePosPredictions = 0
                    else:
                        print 'histOnset 1 not implemented yet'
            posPredictionData.append([minConfidence, np.array(posConfidencePredictionData)])
        #print posPredictionData
        return posPredictionData'''

    def getPosPredictions(self, streamData):
        uniqueVehicleIds = np.unique(streamData[:, 2])
        posPredictionData = []

        posConfidencePredictionData = []
        posConfidences = []
        vehicleCount = 0
        for vehicleId in uniqueVehicleIds:
            #print 'vehicleId', vehicleId
            vehicleIndices = np.where(streamData[:, 2] == vehicleId)[0]

            relData = np.empty(shape=(len(self.minConfidences), len(vehicleIndices), streamData.shape[1]), dtype=object)
            for i in range(len(self.minConfidences)):
                relData[i, :, :] = streamData[vehicleIndices, :]
            confArray = (self.minConfidences * np.ones(shape=(streamData.shape[1], len(vehicleIndices), 1))).T

            timeStampCounterMatrix = relData[:, :, 0]
            #print 'minConfidences', len(self.minConfidences)
            #print 'relData.shape', relData.shape
            #print 'confArray.shape', confArray.shape

            #print 'relData[:,:,5]', relData[:,:,5]
            tooLowConfidencesEntries = (relData[:,:,5] < confArray[:,:,5])
            #print 'tooLowConfidencesEntries', tooLowConfidencesEntries
            timeStampCounterMatrix[tooLowConfidencesEntries] = -1

            #print 'timeStampCounterMatrix', timeStampCounterMatrix
            timeStampCounterDiff = np.diff(timeStampCounterMatrix, axis=1)
            #print 'timeStampCounterDiff', timeStampCounterDiff

            consecutivePosPredictions = np.zeros(shape=(len(self.minConfidences)))
            minTimeStampCounter = np.zeros(shape=(len(self.minConfidences)))
            #print 'relData[0, :, 0]', relData[0, :, 0]
            for i in range(timeStampCounterDiff.shape[1]):
                validIndices = relData[:, i, 0] >= minTimeStampCounter
                #print 'validIndices', validIndices
                #print 'timeStampCounterDiff[:, i] == 1', timeStampCounterDiff[:, i] == 1
                consecutivePosPredictions[(validIndices) & (timeStampCounterDiff[:, i] == 1)] += 1
                consecutivePosPredictions[(validIndices) & (timeStampCounterDiff[:, i] != 1)] = 0
                #print 'consecutivePosPredictions', consecutivePosPredictions

                posPredictions = np.where(consecutivePosPredictions == self.histOnSet - 1)[0]
                if len(posPredictions) > 0:
                    minTimeStampCounter[posPredictions] = relData[0, i+1, 0] + self.histOffSet
                    consecutivePosPredictions[posPredictions] = 0
                    for j in range(len(posPredictions)):
                        posConfidencePredictionData.append(relData[0, i+1, :])
                        #print posConfidencePredictionData
                        posConfidences.append(self.minConfidences[posPredictions[j]])


                #print 'minTimeStampCounter', minTimeStampCounter
                #print 'posConfidencePredictionData', posConfidencePredictionData
            vehicleCount += 1
            #if vehicleCount == 15:
            #    break
        #print posConfidencePredictionData
        #posConfidencePredictionData = sorted(posConfidencePredictionData, key=lambda item: item[0])
        #print posConfidencePredictionData
        posConfidencePredictionData = np.array(posConfidencePredictionData)
        lastConfidence = 0
        for confidence in self.minConfidences:
            indices = np.where(posConfidences==confidence)[0]
            posPredictionData.append([confidence, posConfidencePredictionData[indices]])

        #print posPredictionData
        return posPredictionData

    def getConfidenceEvaluations(self, posPredictionData, groundTruthData, timeStampCounterMapping):
        confidenceEvaluations = np.empty(shape=(0, 4))
        for posPredictionDataConfidence in posPredictionData:
            #print 'jo'
            #print posPredictionDataConfidence
            FP = 0
            TP = 0
            groundTruthDataTmp = groundTruthData.copy()
            for posPrediction in posPredictionDataConfidence[1]:
                #print 'jo2'
                #print posPrediction
                timeStampCounter = posPrediction[0]
                timeStamp = posPrediction[1]
                vehicleId = posPrediction[2]
                groundTruthIdx = posPrediction[3]
                timeStampCategory = posPrediction[4]
                confidence = posPrediction[5]
                if timeStampCategory == HystereseEvaluation.GTC_PRED:
                    if groundTruthDataTmp[groundTruthIdx, 5] == 1:
                        if groundTruthDataTmp[groundTruthIdx][4] == 0:
                            TP += 1
                            logging.debug('TP conf %.2f, vID %d, timeStampCounter %d, ts %d' % (confidence, vehicleId, timeStampCounter, timeStamp))
                            groundTruthDataTmp[groundTruthIdx][4] = 1
                elif timeStampCategory == HystereseEvaluation.GTC_IGNORE:
                    logging.debug('prediction in ignoreTime ts %d' % timeStamp)
                    tsCategoryAfterHyst = self.getTimeStampGroundTruthCategory(groundTruthIdx, self.getTimeStampToTimeStampCounter(timeStampCounterMapping, timeStampCounter + self.histOffSet), groundTruthDataTmp)
                    if tsCategoryAfterHyst == HystereseEvaluation.GTC_NONE:
                        logging.debug('prediction reaches past ignoreTime - FP vID %d, timeStampCounter %d, ts %d' % (vehicleId, timeStampCounter, timeStamp))
                        FP += 1
                else:
                    FP += 1
                    logging.debug('FP conf %.2f, vID %d, timeStampCounter %d, ts %d' % (confidence, vehicleId, timeStampCounter, timeStamp))
            FN = len(np.where((groundTruthDataTmp[:, 5] == 1) & (groundTruthDataTmp[:, 4] == 0))[0])
            confidenceEvaluations = np.vstack([confidenceEvaluations, [posPredictionDataConfidence[0], FP, TP, FN]])
        #print confidenceEvaluations
        return confidenceEvaluations

    def evaluateStream(self, streamData, groundTruthData, timeStampCounterMapping):
        posPredictionData = self.getPosPredictions(streamData)
        return self.getConfidenceEvaluations(posPredictionData, groundTruthData, timeStampCounterMapping)

def doHystEval(trainStreams, testStreams, trainMaxTTCPredPos, trainMaxTTCPredNeg, evalMaxTTCPred, minTTCPred, minConfidences, trainHorizon, vehicleType, negSamplesRatio, rounds=1):
    expCfg = {'iterations': 1, 'trainOfflineSVM': False, 'statisticsLevel': 0, 'statisticsRecordIntervall': 50,
              'saveDatasetFigure': False, 'saveFinalNet': False, 'saveInitialNet': False, 'saveNetIntervall': 0,
              'saveProtoHistogram': False, 'visualizeTraining': False, 'dstDirectory': Paths.StatisticsClassificationDir(),
              'exportToOFRFormat': False, 'epochs': 1, 'expName': ''}

    RFDefault = {'classifierType': 'RF', 'numEstimators': 250, 'max_depth': None, 'name': 'NoName'}

    LVQDefault = {'classifierType': 'LVQ', 'netType': 'GMLVQ', 'activFct': 'linear', 'retrainFrequency': 0,
                 'learnRatePerProto': True, 'learnRateInitial': 10, 'learnRateAnnealingSteps': 1000,
                 'metricLearnRate': 0.1,
                 'windowSize': 1000, 'name': 'NoName', 'LIRAMLVQDimensions': '1',
                 'insertionStrategy': 'SamplingCost', 'insertionTiming': 'errorCount', 'insertionTimingThresh': 2,
                 'sampling': 'random', 'protoAdds': 1, 'deleteNodes': False, 'deletionTimingThresh': 200,
                 'deletionStrategy': 'statistics'}


    netIndivDef = {}

    defaultCfg = {'classifier': RFDefault, 'indiv': netIndivDef}

    cfgs = [defaultCfg]

    hystereseSrcPrefix = Paths.HystereseCutInDir() + vehicleType + '_Stream'

    trainSetName = CutInCommon.getTrainSetName(vehicleType, trainStreams, trainHorizon, trainMaxTTCPredPos, trainMaxTTCPredNeg, negSamplesRatio)

    ROCValuesAVg = np.zeros(shape=(len(minConfidences), 2))
    for i in range(rounds):
        classifiers = doExperiment({'dsName': trainSetName, 'streams': trainStreams, 'splitType': 'simple', 'folds': 5, 'trainOrder': 'random', 'stratified': False,
                                    'shuffle': True, 'chunkSize': 100, 'trainSetSize': .99}, expCfg, cfgs)
        hystEval = HystereseEvaluation(hystereseSrcPrefix, testStreams, classifiers['NoName'][0], 2, 20, minTTCPred=minTTCPred, maxTTCPred=evalMaxTTCPred, minConfidences=minConfidences)
        confidenceEvaluationData = hystEval.evaluate()

        experimentName = trainSetName + '_test%03d-%03d_%d' % (testStreams[0], testStreams[-1], evalMaxTTCPred)
        savePath = Paths.HystereseCutInResultsDir() + experimentName + '.json'
        f = open(savePath, 'w')
        json.dump(confidenceEvaluationData.tolist(), f)
        f.close()
        ROCValues = np.hstack((np.atleast_2d(confidenceEvaluationData[:, 1]).T, np.atleast_2d(confidenceEvaluationData[:, 2] / (confidenceEvaluationData[:, 2] + confidenceEvaluationData[:, 3])).T))
        ROCValuesAVg += ROCValues
    ROCValuesAVg /= float(rounds)

    return [experimentName, ROCValuesAVg]

if __name__ == '__main__':
    #official CBP-setting
    allStreams = np.array([1,2,3,4,5,6,7,8,9,10,11,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86])

    trainStreams = np.array([1,2,3,5,6,7,9,10,11,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44,45,46,48,49,50,52,53,54,56,57,58,60,61,62,64,65,66,68,69,70,72,73,74,76,77,78,80,81,82,84,85,86])
    #trainStreams = np.array([1,2,3,5,6,7,9,10])
    testStreams = np.setdiff1d(allStreams, trainStreams, assume_unique=True)
    #testStreams = np.array([4, 8])

    #trainStreams = np.arange(1, 21, 1)
    #testStreams = np.arange(50, 90, 1)
    #testStreams = np.arange(50, 90, 1)

    #trainMaxTTCPredPos = 20
    #trainMaxTTCPredNeg = 10000
    minTTCPred = 0.1
    vehicleType = '2X_EU'
    #negSamplesRatio = 10000

    #trainSetName = CutInDataGenerator.getTrainSetName(vehicleType, trainStreams, trainHorizon, trainMaxTTCPredPos, trainMaxTTCPredNeg, negSamplesRatio)
    #logging.basicConfig(filename='hyst_' + trainSetName + '.log', filemode='w', format='%(message)s', level=logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    ROCValues = []
    #minConfidences = [0.5,]
    minConfidences = np.arange(0.01, 1, 0.01)

    #negSamplesRatio = 3
    #cProfile.run('ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 500, minTTCPred, minConfidences, trainHorizon, vehicleType, 10))')

    #ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, 160000, vehicleType, 1))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 50, 50, 50, minTTCPred, minConfidences, 128000, vehicleType, 1))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, 112000, vehicleType, 1))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 50, 50, 50, minTTCPred, minConfidences, 96000, vehicleType, 1))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, 80000, vehicleType, 1))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 50, 50, 50, minTTCPred, minConfidences, 64000, vehicleType, 1))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, 48000, vehicleType, 1))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, 32000, vehicleType, 1))

    '''ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 1))
    ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 2))
    ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 3))
    ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 5))
    ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 8))
    ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 10))'''

    '''ROCValues.append(doHystEval(np.arange(1, 2, 1), testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 1))
    ROCValues.append(doHystEval(np.arange(1, 3, 1), testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 1))
    ROCValues.append(doHystEval(np.arange(1, 6, 1), testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 1))
    ROCValues.append(doHystEval(np.arange(1, 11, 1), testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 1))
    ROCValues.append(doHystEval(np.arange(1, 21, 1), testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 1))
    ROCValues.append(doHystEval(np.arange(1, 31, 1), testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 1))
    ROCValues.append(doHystEval(np.arange(1, 41, 1), testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 1))
    ROCValues.append(doHystEval(np.arange(1, 51, 1), testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 1))'''

    #ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, trainHorizon, vehicleType, 1))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, trainHorizon, vehicleType, 2))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, trainHorizon, vehicleType, 3))

    #ROCValues.append(doHystEval(trainStreams, testStreams, 10000, 10000, 10000, minTTCPred, minConfidences, trainHorizon, vehicleType, 10000))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 50, 50, 50, minTTCPred, minConfidences, trainHorizon, vehicleType, 5))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, trainHorizon, vehicleType, 5))

    '''ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 128000, vehicleType, 5))
    ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 96000, vehicleType, 5))
    ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 64000, vehicleType, 5))'''

    '''ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 64000, vehicleType, 10))
    ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 64000, vehicleType, 5))
    ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 64000, vehicleType, 3))
    ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 64000, vehicleType, 2))
    ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 64000, vehicleType, 1))'''


    #ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, 64000, vehicleType, 5))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 50, 50, 50, minTTCPred, minConfidences, 64000, vehicleType, 5))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 64000, vehicleType, 5))
    ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 64000, vehicleType, 1))
    ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 64000, vehicleType, 5))
    ROCValues.append(doHystEval(trainStreams, testStreams, 100, 100, 100, minTTCPred, minConfidences, 0, vehicleType, 1))


    '''ROCValues.append(doHystEval(trainStreams, testStreams, 50, 50, 50, minTTCPred, minConfidences, 96000, vehicleType, 1))
    ROCValues.append(doHystEval(trainStreams, testStreams, 50, 50, 50, minTTCPred, minConfidences, 96000, vehicleType, 3))
    ROCValues.append(doHystEval(trainStreams, testStreams, 50, 50, 50, minTTCPred, minConfidences, 96000, vehicleType, 5))
    ROCValues.append(doHystEval(trainStreams, testStreams, 50, 50, 50, minTTCPred, minConfidences, 96000, vehicleType, 8))
    ROCValues.append(doHystEval(trainStreams, testStreams, 50, 50, 50, minTTCPred, minConfidences, 96000, vehicleType, 10))'''

    #ROCValues.append(doHystEval(trainStreams, testStreams, 20, 20, 20, minTTCPred, minConfidences, 128000, vehicleType, 1))
    #ROCValues.append(doHystEval(trainStreams, testStreams, 10000, 10000, 10000, minTTCPred, minConfidences, 128000, vehicleType, 10000))

    #print ROCValues
    cIdx = 0
    for evalValue in ROCValues:
        label = evalValue[0]
        X = []
        Y = []
        cIdx += 1
        for rocValue in evalValue[1]:
            X.append(rocValue[0])
            Y.append(rocValue[1])
        plt.plot(X, Y, label=label, color=GLVQPlot.getDefColors()[cIdx], linestyle='-')
        plt.ylim([0, 0.7])
        plt.xlim([0, 25])
        plt.legend(loc=0)
    plt.show()






