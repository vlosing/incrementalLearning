import json
import time
import numpy as np
import logging
from Experiment.doExperiment import doExperiment
from Base import Paths
from Base import Serialization
import CutInCommon

class CBPInPhysicalPredictionWriter(object):
    def __init__(self, srcPathPrefix, ppDstPathPrefix, streams, classifier=None, confidences=None, maxTTCPred=50, minTTCPred=0.1):
        self.srcPathPrefix = srcPathPrefix
        self.classifier = classifier
        self.ppDstPathPrefix = ppDstPathPrefix
        self.streams = streams
        self.maxTTCPred = maxTTCPred
        self.minTTCPred = minTTCPred
        self.sampleCounter = 0
        self.confidences = confidences
        #self.recordedConfidences = []

    def writeStreamPredictions(self, streamData):
        for fileData in streamData:
            timeStamp = fileData[0]
            samples = np.array(fileData[1])
            vehicleIds = fileData[2]
            allVehicleIds = fileData[3]
            ppData = fileData[4]
            ppFileName = fileData[5]
            for idx in range(len(vehicleIds)):
                foundVehicleId = False
                confidence = 0
                ttcPred = samples[idx, 12]
                #relevantCase = False
                #np.allclose(samples[idx], np.zeros(shape=(len(samples[idx]), 1))) and
                if ttcPred >= self.minTTCPred and ttcPred <= self.maxTTCPred:
                    relevantCase = True
                    if self.confidences is None:
                        predictions = self.classifier.predict_proba(samples[idx])
                        confidence = predictions[0][1]
                    else:
                        confidence = self.confidences[self.sampleCounter]
                        self.sampleCounter += 1

                for ppVehicleIdx in range(len(ppData['OtherVehicles'])):
                    if ppData['OtherVehicles'][ppVehicleIdx]['Id'] == vehicleIds[idx]:
                        #if ppData['OtherVehicles'][ppVehicleIdx]['Relevance'] == 0:
                        #    confidence = 0
                        #if relevantCase:
                        #    self.recordedConfidences.append(ppData['OtherVehicles'][ppVehicleIdx]['ContextCutInProbability'])
                        ppData['OtherVehicles'][ppVehicleIdx]['ContextCutInProbability'] = confidence
                        foundVehicleId = True
                        break
                if not foundVehicleId:
                    raise Exception('VehicleID ' + vehicleIds[idx] + ' not found in PhysicalPrediction data')

            f = open(self.ppDstPathPrefix + ppFileName, 'w')
            json.dump(ppData, f)
            f.close()

    def writePredictions(self):
        tic = time.time()
        for stream in self.streams:
            logging.info(stream)
            streamData = Serialization.loadPickleFile(self.srcPathPrefix + "%03d" % stream + '.pickle')
            self.writeStreamPredictions(streamData[0])
        #print len(self.recordedConfidences)
        #np.savetxt('OldCutInConfidences', self.recordedConfidences)
        logging.info(str(time.time() - tic) + " seconds")

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    expCfg = {'iterations': 1, 'trainOfflineSVM': False, 'statisticsLevel': 0, 'statisticsRecordIntervall': 50,
              'saveDatasetFigure': False, 'saveFinalNet': False, 'saveInitialNet': False, 'saveNetIntervall': 0,
              'saveProtoHistogram': False, 'visualizeTraining': False, 'dstDirectory': Paths.StatisticsClassificationDir(),
              'exportToOFRFormat': False, 'epochs': 1, 'expName': ''}

    LVQDefault = {'classifierType': 'LVQ', 'netType': 'GMLVQ', 'activFct': 'linear', 'retrainFrequency': 0,
                 'learnRatePerProto': True, 'learnRateInitial': 10, 'learnRateAnnealingSteps': 1000,
                 'metricLearnRate': 0.1,
                 'windowSize': 1000, 'name': 'NoName', 'LIRAMLVQDimensions': '1',
                 'insertionStrategy': 'SamplingCost', 'insertionTiming': 'errorCount', 'insertionTimingThresh': 2,
                 'sampling': 'random', 'protoAdds': 1, 'deleteNodes': False, 'deletionTimingThresh': 200,
                 'deletionStrategy': 'statistics'}
    RFDefault = {'classifierType': 'RF', 'numEstimators': 1000, 'max_depth': None, 'max_features': 'auto', 'criterion': 'entropy', 'name': 'NoName'}
    netIndivDef = {}
    defaultCfg = {'classifier': RFDefault, 'indiv': netIndivDef}
    cfgs = [defaultCfg]

    vehicleType = '2X_EU'
    #vehicleType = '3X_V2'

    trainMaxTTCPredPos = 100
    trainMaxTTCPredNeg = 100
    trainHorizon = 64000
    trainNegSamplesRatio = 5

    maxTTCPredCBP = 100
    minTTCPred = 0.1
    allStreams = np.array([1,2,3,4,5,6,7,8,9,10,11,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86])
    trainStreams = np.array([1,2,3,5,6,7,9,10,11,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44,45,46,48,49,50,52,53,54,56,57,58,60,61,62,64,65,66,68,69,70,72,73,74,76,77,78,80,81,82,84,85,86])
    testStreams = np.setdiff1d(allStreams, trainStreams, assume_unique=True)
    #trainStreams = np.array([1,2,5,6,9,10,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84])
    #trainStreams = np.array([1,5,9,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78,82])
    #trainStreams = np.array([5,22,30,38,46,54,62,70,78])
    #trainStreams = np.array([22,38,54,70])

    trainSetName = CutInCommon.getTrainSetName(vehicleType, trainStreams, trainHorizon, trainMaxTTCPredPos, trainMaxTTCPredNeg, trainNegSamplesRatio)
    print trainSetName

    classifiers = doExperiment({'dsName': trainSetName, 'streams': trainStreams, 'splitType': 'simple', 'folds': 5, 'trainOrder': 'random', 'stratified': False,
                               'shuffle': True, 'chunkSize': 100, 'trainSetSize': 0.99}, expCfg, cfgs)
    classifier = classifiers['NoName'][0]
    hystereseSrcPrefix = Paths.HystereseCutInDir() + vehicleType + '_Stream'

    dstPrefix = '/hri/localdisk/vlosing/ACC_Data/'
    cbpWriter = CBPInPhysicalPredictionWriter(hystereseSrcPrefix, dstPrefix + trainSetName + '/Results_c/',
                                              testStreams,
                                              classifier=classifier,
                                              maxTTCPred=maxTTCPredCBP,
                                              minTTCPred=minTTCPred,
                                              confidences=None
                                              #confidences=np.loadtxt('orfEvalCutInTest.txt')
                                              )

    cbpWriter.writePredictions()

