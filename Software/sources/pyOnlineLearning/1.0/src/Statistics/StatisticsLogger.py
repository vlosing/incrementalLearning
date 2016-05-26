import json
import os
import logging
import numpy as np
from Base import Paths
from ClassificationExperiment.ExperimentListener import ExpListener


class StatsItem:
    def __init__(self):
        self.item = None
        self.foldItemLst = []
        self.itemLst = []

        self.itemAll = {}

    def initNewCfg(self, cfgName):
        self.item = np.zeros(shape=(1, 1))
        if cfgName in self.itemAll.keys():
            self.itemLst = self.itemAll[cfgName]
        else:
            self.itemLst = []
        self.itemLst.append([])

        self.foldItemLst = []
        self.foldItemLst.append([])

    def assignItemToFoldLst(self, foldIdx):
        self.foldItemLst[foldIdx] = self.item
        self.foldItemLst.append([])
        self.item = np.zeros(shape=(1, 1))

    def assignFoldLstToItemLst(self, lstIdx, cfgName):
        self.itemLst[lstIdx] = self.foldItemLst
        self.itemAll[cfgName] = self.itemLst

    def appendValue(self, value):
        self.item = np.append(self.item, value)


class StatisticsLogger(ExpListener):
    TRAINSTEPS_FIRST_RECORD_INTERVALL = 0

    def __init__(self, doTrainStepStatistics=True, doPrototypeStatistics=False, generateRejectionData=False, recordIntervall = 300,
                 loggTrainCompleteAccuracy=False, loggTrainAccuracy=False, loggShortTermAccuracy=False, loggTestAccuracy=True):
        super(StatisticsLogger, self).__init__()
        self.doTrainStepStatistics = doTrainStepStatistics
        self.doPrototypeStatistics = doPrototypeStatistics
        self.recordIntervall = recordIntervall
        self.loggTrainCompleteAccuracy = loggTrainCompleteAccuracy
        self.loggTrainAccuracy = loggTrainAccuracy
        self.loggShortTermAccuracy = loggShortTermAccuracy
        self.loggTestAccuracy = loggTestAccuracy

        self.trainStepCount = 0
        self.developmentError = 0
        self.trainTestSplit = None
        self.trainTestSplitLst = []

        self.lastComplexitySize = -1

        self.statsItems = []
        self.generateRejectionData = generateRejectionData

        self.complexitySize = StatsItem()
        self.complexitySizeTrainAcc = StatsItem()
        self.complexitySizeTrainComplAcc = StatsItem()
        self.complexitySizeTestAcc = StatsItem()
        self.complexitySizeInternSampleAcc = StatsItem()
        self.complexitySizeTrainStepCount = StatsItem()
        self.trainStepCounts = StatsItem()
        self.trainStepCountTrainAcc = StatsItem()
        self.trainStepCountTrainComplAcc = StatsItem()
        self.trainStepCountTestAcc = StatsItem()
        self.trainStepCountInternSampleAcc = StatsItem()
        self.trainStepCountTrainCost = StatsItem()
        self.trainStepCountTrainComplCost = StatsItem()
        self.trainStepCountTestCost = StatsItem()
        self.trainStepCountInternSampleCost = StatsItem()

        self.trainStepCountComplexitySize = StatsItem()
        self.trainStepCountDevelopmentAcc = StatsItem()

        self.portionRejectionAll = {}
        self.classRateRejectionAll = {}

        self.statsItems.append(self.complexitySize)
        self.statsItems.append(self.complexitySizeTrainAcc)
        self.statsItems.append(self.complexitySizeTrainComplAcc)
        self.statsItems.append(self.complexitySizeTestAcc)
        self.statsItems.append(self.complexitySizeInternSampleAcc)
        self.statsItems.append(self.complexitySizeTrainStepCount)
        self.statsItems.append(self.trainStepCounts)
        self.statsItems.append(self.trainStepCountTrainAcc)
        self.statsItems.append(self.trainStepCountTrainComplAcc)
        self.statsItems.append(self.trainStepCountTestAcc)
        self.statsItems.append(self.trainStepCountInternSampleAcc)
        self.statsItems.append(self.trainStepCountTrainCost)
        self.statsItems.append(self.trainStepCountTrainComplCost)
        self.statsItems.append(self.trainStepCountTestCost)
        self.statsItems.append(self.trainStepCountInternSampleCost)
        self.statsItems.append(self.trainStepCountComplexitySize)
        self.statsItems.append(self.trainStepCountDevelopmentAcc)

        self.classifier = None

    def getClassRate(self, doPrint=False):
        trainComplClassRate = 0
        trainComplCost = 1
        testClassRate = 0
        testCost = 1
        internSampleClassRate = 0
        internSampleCost = 1
        trainClassRate = 0
        trainCost = 1

        if self.loggTrainCompleteAccuracy:
            trainComplClassRate, trainComplCost = self.classifier.getAccuracy(self.TrainSamples,
                                                                              self.TrainLabels)
        if self.loggTestAccuracy:
            testClassRate, testCost = self.classifier.getAccuracy(self.TestSamples,
                                                                  self.TestLabels)
        if self.loggShortTermAccuracy:
            internSampleClassRate, internSampleCost = self.classifier.getShortTermMemoryClassRate()
        if self.trainStepCount > 0 and self.loggTrainAccuracy:
            maxIdx = min(len(self.TrainSamples), self.trainStepCount)
            trainClassRate, trainCost = self.classifier.getAccuracy(self.TrainSamples[0:maxIdx, :],
                                                                    self.TrainLabels[0:maxIdx])
        if doPrint:
            #logging.info(str(trainComplClassRate) + " correct on Train complete")
            logging.info(str(trainClassRate) + " correct on Train seen")
            logging.info(str(testClassRate) + "correct on Test")
            #print str(internSampleClassRate) + ' correct on intern samples'

        return trainComplClassRate, testClassRate, trainClassRate, internSampleClassRate, trainComplCost, testCost, \
                trainCost, internSampleCost

    def newCfgIteration(self, cfgName):
        self.trainStepCount = 0
        self.developmentError = 0
        self.lastComplexitySize = -1

        for statItem in self.statsItems:
            statItem.initNewCfg(cfgName)

    def addComplexitySize(self, classifierComplexity):
        trainComplClassRate, testClassRate, trainClassRate, internClassRate, \
        trainComplCost, testCost, trainCost, internCost = self.getClassRate()

        self.complexitySize.appendValue(classifierComplexity)
        self.complexitySizeTrainAcc.appendValue(trainClassRate)
        self.complexitySizeTrainComplAcc.appendValue(trainComplClassRate)
        self.complexitySizeTestAcc.appendValue(testClassRate)
        self.complexitySizeInternSampleAcc.appendValue(internClassRate)
        self.complexitySizeTrainStepCount.appendValue(self.trainStepCount)

    def newTrainStep(self, classificationResult, trainStepCount):
        self.trainStepCount = trainStepCount
        if not classificationResult:
            self.developmentError += 1
        complexitySize = self.classifier.getComplexity()

        if self.doTrainStepStatistics and \
                ((self.trainStepCount >= self.TRAINSTEPS_FIRST_RECORD_INTERVALL and
                              self.trainStepCount % self.recordIntervall == 0) or
                     (self.trainStepCount == self.TRAINSTEPS_FIRST_RECORD_INTERVALL)):
            self.addTrainStepCount(self.trainStepCount, complexitySize)

    def newTrainStep2(self, protoCount, testRate):
        if self.doPrototypeStatistics:
            self.complexitySize.appendValue(protoCount)
            self.complexitySizeTestAcc.appendValue(testRate)

    def finishIteration(self):
        self.trainTestSplitLst.append(self.trainTestSplit) #XXVL

    def finishCfgIteration(self, cfgName, iterationIdx):
        for statItem in self.statsItems:
            statItem.assignFoldLstToItemLst(iterationIdx, cfgName)

    def finishCVCfgIteration(self, foldIdx):
        protoCount = self.classifier.getComplexity()

        if self.doPrototypeStatistics:
            if protoCount > self.lastComplexitySize:
                self.addComplexitySize(protoCount)
        else:
            self.addComplexitySize(protoCount)
        if self.doTrainStepStatistics:
            if self.trainStepCounts.item[-1] != self.trainStepCount:

                self.addTrainStepCount(self.trainStepCount, protoCount)
        else:
            self.addTrainStepCount(self.trainStepCount, protoCount)

        for statItem in self.statsItems:
            statItem.assignItemToFoldLst(foldIdx)
        self.trainStepCount = 0
        self.developmentError = 0
        self.lastComplexitySize = -1

    def finishCfgIteration2(self):
        for statItem in self.statsItems:
            statItem.assignItemToLst(self.currentIter)

    def addTrainStepCount(self, trainStepCount, classifierComplexity):
        trainComplClassRate, testClassRate, trainClassRate, internClassRate, trainComplCost, \
            testCost, trainCost, internCost = self.getClassRate()
        self.trainStepCounts.appendValue(trainStepCount)
        self.trainStepCountTrainAcc.appendValue(trainClassRate)

        self.trainStepCountTrainComplAcc.appendValue(trainComplClassRate)
        self.trainStepCountTestAcc.appendValue(testClassRate)
        self.trainStepCountInternSampleAcc.appendValue(internClassRate)

        self.trainStepCountTrainCost.appendValue(trainCost)
        self.trainStepCountTrainComplCost.appendValue(trainComplCost)
        self.trainStepCountTestCost.appendValue(testCost)
        self.trainStepCountInternSampleCost.appendValue(internCost)

        self.trainStepCountComplexitySize.appendValue(classifierComplexity)
        self.trainStepCountDevelopmentAcc.appendValue(
            (self.trainStepCount - self.developmentError) / float(self.trainStepCount))

    def getRejectionPlotData(self, GLVQS, stepSize=0.05):
        classRateRejectionAll = {}
        portionRejectionAll = {}
        for key in GLVQS.keys():
            classRateRejectionLst = []
            portionRejectionLst = []
            for i in range(len(GLVQS[key])):
                classRateRejection = np.array([])
                portionRejection = np.array([])
                classMatrix = GLVQS[key][i].getDistanceMatrix(self.trainTestSplitLst[i].TestSamples,
                                                                    self.trainTestSplitLst[i].TestLabels)
                relSimValues = GLVQS[key][i].getCostFunctionValuesByMatrix(classMatrix)

                relSimThresh = 0
                countAll = len(relSimValues)

                while relSimThresh <= 1:
                    unrejectedIndices = np.where(np.abs(relSimValues) > relSimThresh)
                    correctIndices = np.where(relSimValues > relSimThresh)
                    countUnrejected = len(unrejectedIndices[0])

                    if countUnrejected > 0:
                        classRateRejection = np.append(classRateRejection,
                                                       len(correctIndices[0]) / float(countUnrejected))
                        portionRejection = np.append(portionRejection, countUnrejected / float(countAll))
                    relSimThresh += stepSize
                classRateRejectionLst.append(classRateRejection[::-1])
                portionRejectionLst.append(portionRejection[::-1])

            classRateRejectionAll[key] = classRateRejectionLst
            portionRejectionAll[key] = portionRejectionLst

        return portionRejectionAll, classRateRejectionAll

    @classmethod
    def getAvgMinMax(cls, X):
        XMaxs = np.array([])
        XMins = np.array([])
        for i in range(len(X)):
            XMaxs = np.append(XMaxs, np.max(X[i]))
            XMins = np.append(XMins, np.min(X[i]))
        return np.average(XMins), np.average(XMaxs)

    @staticmethod
    def getMeanAndVariance(X, Y, XStepSize):
        XSampled = np.array([])
        YAverage = np.array([])
        YVariance = np.array([])
        avgXMin, avgXMax = StatisticsLogger.getAvgMinMax(X)
        x = avgXMin
        maxReached = False
        while x <= avgXMax:
            XSampled = np.append(XSampled, x)

            yValues = np.array([])
            for i in range(len(X)):
                indices = np.where(X[i] <= x)[0]
                if len(indices) > 0:
                    yValues = np.append(yValues, Y[i][np.max(indices)])
                else:
                    yValues = np.append(yValues, Y[i][0])
            YAverage = np.append(YAverage, np.average(yValues))
            YVariance = np.append(YVariance, np.std(yValues))
            x += XStepSize
            if x > avgXMax > XStepSize - x and not maxReached:
                x = avgXMax
                maxReached = True
        return XSampled, YAverage, YVariance

    @staticmethod
    def getMeanAndVarianceNested(X, Y, XStepSize):
        newX = []
        newY = []
        for i in range(len(X)):
            Xavg, Yavg, YVar = StatisticsLogger.getMeanAndVariance(X[i], Y[i], XStepSize)
            newX.append(Xavg)
            newY.append(Yavg)
        return StatisticsLogger.getMeanAndVariance(newX, newY, XStepSize)


    @staticmethod
    def arrayToList(data):
        result = []
        for datum in data:
            subResult = []
            for cvDatum in datum:
                if len(cvDatum) > 0:  # XXVL hack eigentlichen Fehler suchen!
                    subResult.append(cvDatum.tolist())
            result.append(subResult)
        return result

    @staticmethod
    def listToArray(data):
        result = []
        for datum in (data):
            subResult = []
            for cvDatum in datum:
                subResult.append(np.array(cvDatum))
            result.append(subResult)
        return result

    def serialize(self, prefix, path=Paths.StatisticsClassificationDir()):
        json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')
        if not os.path.exists(path):
            os.makedirs(path)
        for cfgName in self.complexitySize.itemAll.keys():  # XXVL unschoen
            filename = path + prefix + '_' + cfgName + '.json'
            data = list()
            for statItem in self.statsItems:
                data.append(StatisticsLogger.arrayToList(statItem.itemAll[cfgName]))
            if cfgName != 'SVM' and self.generateRejectionData:
                data.append(StatisticsLogger.arrayToList(self.portionRejectionAll[cfgName]))
                data.append(StatisticsLogger.arrayToList(self.classRateRejectionAll[cfgName]))

            f = open(filename, 'w')
            json.dump(data, f)
            f.close()

    def deSerialize(self, cfgNames, prefix, path=Paths.StatisticsClassificationDir()):
        for cfgName in cfgNames:
            filename = path + prefix + '_' + cfgName + '.json'
            f = open(filename, 'r')
            data = json.load(f)
            f.close()
            counter = 0
            for statItem in self.statsItems:
                statItem.itemAll[cfgName] = StatisticsLogger.listToArray(data[counter])
                counter += 1
            if cfgName != 'SVM' and self.generateRejectionData:
                self.portionRejectionAll[cfgName] = StatisticsLogger.listToArray(data[counter])
                counter += 1
                self.classRateRejectionAll[cfgName] = StatisticsLogger.listToArray(data[counter])

    def initRejectionData(self, GLVQS):
        self.portionRejectionAll, self.classRateRejectionAll = self.getRejectionPlotData(GLVQS, 0.01)

    def onTrainDataChange(self, trainSamples, testSamples, trainLabels, testLabels):
        self.TrainSamples = trainSamples
        self.TestSamples = testSamples
        self.TrainLabels = trainLabels
        self.TestLabels = testLabels


    def onNewClassifier(self, classifier):
        self.classifier = classifier

    def onNewCfgIteration(self, cfgName):
        self.newCfgIteration(cfgName)

    def onNewTrainStep(self, classifier, classificationResult, trainStep):
        self.newTrainStep(classificationResult, trainStep)

    def onFinishIteration(self):
        self.finishIteration()

    def onFinishCfgIteration(self, cfgName, iterationIdx):
        self.finishCfgIteration(cfgName, iterationIdx)

    def onFinishFoldCfgIteration(self, foldIdx):
        self.finishCVCfgIteration(foldIdx)

    def onFinishExperiment(self, classifiers, expPrefix):
        if self.generateRejectionData:
            self.initRejectionData(classifiers)
        self.serialize(expPrefix)

    def onComplexityChanged(self, classifier):
        if self.doPrototypeStatistics:
            self.addComplexitySize(classifier.getComplexity())

    def onNewPrototypes(self, classifier, protos, protoLabels):
        self.onComplexityChanged(classifier)


    def onNewWindowSize(self, classifier):
        pass

    def onDelPrototypes(self, classifier, protos, protoLabels):
        self.onComplexityChanged(classifier)