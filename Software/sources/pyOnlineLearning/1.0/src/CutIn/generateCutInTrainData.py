__author__ = 'vlosing'
import json
from Base.Serialization import SimpleFileIterator
from Base.Serialization import loadJsonFile
from Base import Paths
import os
import numpy as np
import time
import CutInCommon
import logging

class CutInDataGenerator(object):
    def __init__(self, prefixDirectory, vehicleType, streams, srcPrefix='CBPDebugData_', srcNumberLength=10, seqLength=96000, maxTTCPredPos=50, maxTTCPredNeg=50, minTTCPred=0.1, negSamplesRatio=0):
        self.prefixDirectory = prefixDirectory
        self.vehicleType = vehicleType
        self.streams = streams
        self.srcPrefix = srcPrefix
        self.srcNumberLength = srcNumberLength
        self.count = 0
        self.seqLength = seqLength
        self.maxTTCPredPos = maxTTCPredPos
        self.maxTTCPredNeg = maxTTCPredNeg
        self.minTTCPred = minTTCPred
        self.negSamplesRatio = negSamplesRatio
        trainSetName = CutInCommon.getTrainSetName(self.vehicleType, self.streams, self.seqLength, self.maxTTCPredPos, self.maxTTCPredNeg, self.negSamplesRatio)
        print trainSetName

    def genJsonFileName(self, dataTime):
        fileName = self.srcPrefix + '%010d' % dataTime
        return fileName

    def getSampleFromFileName(self, fileNamePath, maxTTCPred, vehicleID=None):
        featureFile = loadJsonFile(fileNamePath)
        features = None
        if featureFile is not None:
            for vehicle in featureFile['Vehicles']:
                if vehicleID is not None:
                    if vehicle['Id'] == vehicleID:
                        #[perception_confidence_pred', 'ttc_lpred', 't_on_lane_pred', 'relV_pred', 'ttc_lsuc',
                        #'perception_confidence_v', 'g_size', 'abs_a', 't_last_lane_change', 'diff_ttc_ttg',
                        #'diff_ttc_ttca', 'ttg', 'ttc_pred']
                        #np.allclose(vehicle['IndicatorVariables'].values(), np.zeros(shape=(len(vehicle['IndicatorVariables'].values()), 1)))
                        ttcPred = vehicle['IndicatorVariables']['ttc_pred']
                        if self.minTTCPred <= ttcPred <= maxTTCPred:
                            features = vehicle['IndicatorVariables'].values()
                            break
                else:
                    #[perception_confidence_pred', 'ttc_lpred', 't_on_lane_pred', 'relV_pred', 'ttc_lsuc',
                    #'perception_confidence_v', 'g_size', 'abs_a', 't_last_lane_change', 'diff_ttc_ttg',
                    #'diff_ttc_ttca', 'ttg', 'ttc_pred']
                    ttcPred = vehicle['IndicatorVariables']['ttc_pred']
                    if self.minTTCPred <= ttcPred <= maxTTCPred:
                        features = vehicle['IndicatorVariables'].values()
                        break
        return features

    def genAndSaveSamples(self):
        fileNames = []
        for stream in self.streams:
            logging.info(stream)
            prefixDirectory = os.path.normpath(self.prefixDirectory) + os.sep + self.vehicleType + '/Results_c/' + "Stream%03d" % stream
            srcDirectoryCBP = prefixDirectory + '/CBP/data/'
            groundTruthPath = prefixDirectory + '/Evaluation/groundtruth.json'
            fileIterator = SimpleFileIterator(srcDirectoryCBP, 'json')
            posSamples, posLabels, posFileNames = self.generatePosSamples(fileIterator, srcDirectoryCBP, groundTruthPath)
            groundTruthFileNames = self.getGroundTruthFileNames(groundTruthPath, fileIterator)
            if self.negSamplesRatio==0:
                numberOfNegSamples = 0
            else:
                numberOfNegSamples = len(posSamples) * self.negSamplesRatio
            negSamples, negLabels, negFileNames = self.generateNegSamples(fileIterator, srcDirectoryCBP, groundTruthFileNames, numberOfNegSamples)
            trainSetStreamName = CutInCommon.getTrainSetStreamName(self.vehicleType, stream, self.seqLength, self.maxTTCPredPos, self.maxTTCPredNeg, self.negSamplesRatio)

            pathPrefix = Paths.CutInFeaturesDir() + self.vehicleType + '/' + trainSetStreamName
            f = open(pathPrefix + '_features.json', 'w')
            json.dump(posSamples + negSamples, f)
            f.close()
            f = open(pathPrefix + '_labels.json', 'w')
            json.dump(posLabels + negLabels, f)
            f.close()
            fileNames = fileNames + posFileNames + negFileNames
            logging.info('positive examples %d' % (len(posSamples)))
            logging.info('negative examples %d' % (len(negSamples)))

    def generateNegSamples(self, fileIterator, cbpDirectory, ignoreFileNames, numberOfSamples):
        samples = []
        labels = []
        fileNames = np.array(fileIterator.getFileNames())
        fileNames = np.setdiff1d(fileNames, ignoreFileNames, assume_unique=True)
        fileNames = fileNames[np.random.permutation(len(fileNames))]
        fileIdx = 0
        usedFileNames = []
        while (len(samples) < numberOfSamples or numberOfSamples == 0) and fileIdx < len(fileNames):
            sample = self.getSampleFromFileName(cbpDirectory + fileNames[fileIdx], self.maxTTCPredNeg)
            if sample is not None:
                samples.append(sample)
                labels.append(0)
                usedFileNames.append(fileNames[fileIdx])
            fileIdx += 1
        return samples, labels, usedFileNames

    def getGroundTruthFileNames(self, groundTruthPath, fileIterator):
        groundTruths = loadJsonFile(groundTruthPath)
        allFileNames = np.array([])
        for groundTruth in groundTruths:
            startTime = groundTruth['pred_time_start']
            endTime = groundTruth['pred_time_stop']
            startFileName = self.genJsonFileName(startTime)
            endFileName = self.genJsonFileName(endTime)
            fileNames = fileIterator.getFileNamesInRange(startFileName, endFileName)
            allFileNames = np.append(allFileNames, fileNames)
        return allFileNames

    '''def generatePosSamples(self, fileIterator, cbpDirectory, groundTruthPath):
        samples = []
        labels = []
        groundTruths = loadJsonFile(groundTruthPath)

        usedFileNames = []
        usedVehicleIDs = []
        for groundTruth in groundTruths:
            if groundTruth['pred_case'] == 1:
                if self.seqLength == 0:
                    startTime = groundTruth['pred_time_start']
                else:
                    startTime = max(groundTruth['time_reference'] - self.seqLength, groundTruth['pred_time_start'])
                endTime = groundTruth['time_reference']
                vehicleID = groundTruth['vehicle_id']
                startFileName = self.genJsonFileName(startTime)
                endFileName = self.genJsonFileName(endTime)
                fileNames = fileIterator.getFileNamesInRange(startFileName, endFileName)
                for fileName in fileNames:
                    sample = self.getSampleFromFileName(cbpDirectory + fileName, self.maxTTCPredPos, vehicleID=vehicleID)
                    if sample is not None:
                        samples.append(sample)
                        labels.append(1)
                        usedFileNames.append(fileName)
                        usedVehicleIDs.append(vehicleID)
        return samples, labels, usedFileNames, usedVehicleIDs'''

    def generatePosSamples(self, fileIterator, cbpDirectory, groundTruthPath):
        samples = []
        labels = []
        groundTruths = loadJsonFile(groundTruthPath)

        usedFileNames = []
        for groundTruth in groundTruths:
            if groundTruth['pred_case'] == 1:
                if self.seqLength == 0:
                    startTime = groundTruth['pred_time_start']
                else:
                    startTime = max(groundTruth['time_reference'] - self.seqLength, groundTruth['pred_time_start'])
                endTime = groundTruth['time_reference']
                vehicleID = groundTruth['vehicle_id']
                startFileName = self.genJsonFileName(startTime)
                endFileName = self.genJsonFileName(endTime)
                fileNames = fileIterator.getFileNamesInRange(startFileName, endFileName)
                for fileName in fileNames:
                    sample = self.getSampleFromFileName(cbpDirectory + fileName, self.maxTTCPredPos, vehicleID=vehicleID)
                    if sample is not None:
                        samples.append(sample)
                        labels.append(1)
                        usedFileNames.append(fileName)
        return samples, labels, usedFileNames

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    srcPrefix = '/hri/localdisk/vlosing/ACC_Data'
    vehicleType = '2X_EU'
    maxTTCPredPos = 100
    maxTTCPredNeg = 100
    seqLength = 0

    #streams = np.arange(1, 21, 1)
    #streams = np.arange(21, 51, 1)
    #streams = np.arange(1, 51, 1)
    allStreams = np.array([1,2,3,4,5,6,7,8,9,10,11,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86])
    trainStreams = np.array([1,2,3,5,6,7,9,10,11,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44,45,46,48,49,50,52,53,54,56,57,58,60,61,62,64,65,66,68,69,70,72,73,74,76,77,78,80,81,82,84,85,86])
    testStreams = np.setdiff1d(allStreams, trainStreams, assume_unique=True)

    streams = np.array([1,2,3,5,6,7,9,10,11,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44,45,46,48,49,50,52,53,54,56,57,58,60,61,62,64,65,66,68,69,70,72,73,74,76,77,78,80,81,82,84,85,86])

    dataGen = CutInDataGenerator(srcPrefix, vehicleType, trainStreams, seqLength=seqLength, maxTTCPredPos=maxTTCPredPos, maxTTCPredNeg=maxTTCPredNeg, negSamplesRatio=5)
    dataGen.genAndSaveSamples()
    '''dataGen = CutInDataGenerator(srcPrefix, vehicleType, streams, seqLength=seqLength, maxTTCPredPos=maxTTCPredPos, maxTTCPredNeg=maxTTCPredNeg, negSamplesRatio=2)
    dataGen.genAndSaveSamples()
    dataGen = CutInDataGenerator(srcPrefix, vehicleType, streams, seqLength=seqLength, maxTTCPredPos=maxTTCPredPos, maxTTCPredNeg=maxTTCPredNeg, negSamplesRatio=3)
    dataGen.genAndSaveSamples()
    dataGen = CutInDataGenerator(srcPrefix, vehicleType, streams, seqLength=seqLength, maxTTCPredPos=maxTTCPredPos, maxTTCPredNeg=maxTTCPredNeg, negSamplesRatio=5)
    dataGen.genAndSaveSamples()'''
    #dataGen = CutInDataGenerator(srcPrefix, vehicleType, streams, seqLength=seqLength, maxTTCPredPos=maxTTCPredPos, maxTTCPredNeg=maxTTCPredNeg, negSamplesRatio=8)
    #dataGen.genAndSaveSamples()
    #dataGen = CutInDataGenerator(srcPrefix, vehicleType, streams, seqLength=seqLength, maxTTCPredPos=maxTTCPredPos, maxTTCPredNeg=maxTTCPredNeg, negSamplesRatio=10)
    #dataGen.genAndSaveSamples()
    #dataGen = CutInDataGenerator(srcPrefix, vehicleType, streams, seqLength=seqLength, maxTTCPredPos=maxTTCPredPos, maxTTCPredNeg=maxTTCPredNeg, negSamplesRatio=100)
    #dataGen.genAndSaveSamples()
    #dataGen = CutInDataGenerator(srcPrefix, vehicleType, streams, seqLength=seqLength, maxTTCPredPos=maxTTCPredPos, maxTTCPredNeg=maxTTCPredNeg, negSamplesRatio=10000)
    #dataGen.genAndSaveSamples()





    #cProfile.run("samples, labels, featureFileNames = dataGen.generateSamples()")


