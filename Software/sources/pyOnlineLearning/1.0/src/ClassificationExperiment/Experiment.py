__author__ = 'vlosing'

import time
import logging

import numpy as np

from LVQPY.GLVQ import GLVQ
from Visualization import GLVQPlot
from LVQPY import LVQFactory
from DataGeneration import DataSetFactory
from HyperParameter.hyperParameterFactory import getHyperParams
from DataGeneration.TrainTestSplitsManager import TrainTestSplitsManager
from ClassifierCommon.ClassifierListener import DummyClassifierListener
from Visualization.ClassifierVisualizer import ClassifierVisualizer
from Classifier.RandomForest import RandomForest
from Classifier.SVM import SVM
import uuid
from sklearn.metrics import accuracy_score
from Base import Paths
from DataGeneration.DataSetFactory import isStationary
from ClassifierComparison.auxiliaryFunctions import getClassifier, KNearestLeaveOneOut
from Visualization.DefClassifierPlotter import DefClassifierPlotter
from Classifier.KNNWindow import DriftDetection


'''def saveMissClassifiedData(glvq, Samples, Labels, FileNames, classNames, srcImageDir, dstImageDir,
                           dstStatFileName):
    indices, confusedClassLabels = glvq.getMissClassifiedIndices(Samples, Labels)
    MissClassifiedFileNames = FileNames[indices]
    CorrectClassLabelNames = classNames[Labels[indices]]
    ConfusedClassLabelNames = classNames[confusedClassLabels]

    missClassifiedStatLst = []
    if not os.path.exists(dstImageDir):
        os.makedirs(dstImageDir)
    for i in range(len(MissClassifiedFileNames)):
        entry = [MissClassifiedFileNames[i], CorrectClassLabelNames[i], ConfusedClassLabelNames[i]]
        missClassifiedStatLst.append(entry)

        maskedImage = CVFcts.getMaskedImageFromFile(srcImageDir + MissClassifiedFileNames[i])
        parts = MissClassifiedFileNames[i].split('.')

        dstFileName = dstImageDir + parts[0] + '_masked.' + parts[1]
        Serial.saveImage(maskedImage, dstFileName)
        dstFileName = dstImageDir + parts[0] + '_rg.' + parts[1]
        CVFcts.saveObjectPixelsRGChromacityFig(srcImageDir + MissClassifiedFileNames[i], dstFileName,
                                               MissClassifiedFileNames[i])
        dstFileName = dstImageDir + parts[0] + '_hist.' + parts[1]
        CVFcts.saveHistFig(srcImageDir + MissClassifiedFileNames[i], dstFileName, MissClassifiedFileNames[i])

    Serial.saveImages(srcImageDir, MissClassifiedFileNames, dstImageDir)

    missClassifiedStatLst = natsort.natsorted(missClassifiedStatLst, key=lambda missClassified: missClassified[0])
    f = open(dstStatFileName, 'w')
    json.dump(missClassifiedStatLst, f)
    f.close()'''


class Experiment:
    def __init__(self, trainingSetCfg, expCfg, cfgs, expListener):
        self.trainingSetCfg = trainingSetCfg
        self.expCfg = expCfg
        self.cfgs = cfgs
        self.dataSet = DataSetFactory.getDataSet(self.trainingSetCfg['dsName'])
        self.trainTestSplitsManager = TrainTestSplitsManager(self.dataSet.samples, self.dataSet.labels, self.dataSet.featureFileNames,
                                                     trainingSetCfg['trainOrder'], trainingSetCfg['chunkSize'],
                                                     trainingSetCfg['shuffle'], trainingSetCfg['stratified'],
                                                     trainingSetCfg['splitType'], trainingSetCfg['folds'],
                                                     trainingSetCfg['trainSetSize'])
        self.expListener = expListener
        self.classifiers = {}
        self.hyperParams = None

    @staticmethod
    def getExpName(expCfg, trainingSetCfg):
        expName = ''
        if 'expName' in trainingSetCfg.keys() and trainingSetCfg['expName'] != '':
            expName = trainingSetCfg['expName']
        elif expCfg is not None and 'expName' in expCfg.keys() and expCfg['expName'] != '':
            expName = expCfg['expName']
        return expName

    @staticmethod
    def getExpPrefix(expCfg, trainingSetCfg):
        expName = Experiment.getExpName(expCfg, trainingSetCfg)
        if expName != '':
            return expName
        else:
            return (trainingSetCfg['dsName'] + '_' + trainingSetCfg['trainOrder'] + '_' + str(
                    trainingSetCfg['trainSetSize'])).replace('.', '')

    @staticmethod #XXVL!
    def getExpPrefix2(expCfg, trainingSetCfg):
        expName = Experiment.getExpName(expCfg, trainingSetCfg)
        if expName != '':
            return expName
        else:
            return (trainingSetCfg['dsName'] + '_' + trainingSetCfg['trainOrder'] + '_' +  (trainingSetCfg['splitType'] or 'None'))

    def initIteration(self, iterationIdx):
        self.trainTestSplitsManager.generateSplits()
        logging.info(str(len(self.trainTestSplitsManager.TrainLabelsLst[0])) + ' TrainSamples')
        logging.info(str(len(self.trainTestSplitsManager.TestLabelsLst[0])) + ' TestSamples')
        if iterationIdx == 0:
            if self.expCfg['saveDatasetFigure'] and self.dataSet.dataDimensions == 2:
                dummy = GLVQ(self.dataSet.dataDimensions)
                fig, dummy = GLVQPlot.plotAll(dummy, dummy.prototypes, dummy.prototypesLabels, self.dataSet.samples,
                                              self.dataSet.labels, GLVQPlot.getDefColors(),
                                              self.trainingSetCfg['dsName'])
                fig.savefig(self.expCfg['dstDirectory'] + self.trainingSetCfg['dsName'] + '.pdf', bbox_inches='tight')

    def genNewClassifier(self, cfg, criteria, foldIdx):
        classifierName = cfg['classifierName']
        self.hyperParams = getHyperParams(self.trainingSetCfg['dsName'], classifierName, False)
        for key in cfg['indiv'].keys():
            self.hyperParams[key] = cfg['indiv'][key]
        logging.info(self.hyperParams)

        if self.expCfg['visualizeTraining']:
            visualizer = ClassifierVisualizer(DefClassifierPlotter(), self.trainTestSplitsManager.TrainSamplesLst[foldIdx], self.trainTestSplitsManager.TrainLabelsLst[foldIdx],
                                                   self.trainTestSplitsManager.TestSamplesLst[foldIdx], self.trainTestSplitsManager.TestLabelsLst[foldIdx], criteria)
        else:
            visualizer = DummyClassifierListener()

        classifier = getClassifier(classifierName, self.hyperParams, listener=[self.expListener, visualizer])

        #if classifierName == 'ILVQ':
            #if self.expCfg['visualizeTraining'] and self.dataSet.dataDimensions == 2:

        if criteria in self.classifiers.keys():
            classifierLst = self.classifiers[criteria]
        else:
            classifierLst = []
        classifierLst.append(classifier)
        self.classifiers[criteria] = classifierLst
        self.expListener.onNewClassifier(classifier)
        return classifier

    def getFileName(self, fileNameSuffix, iterationIdx, criteria, foldIdx):
        fileNamePrefix = self.expCfg['dstDirectory'] + Experiment.getExpPrefix(self.expCfg, self.trainingSetCfg) + '_' + str(iterationIdx) + '_' + str(foldIdx)
        if criteria != '':
            fileNamePrefix += '_' + criteria
        return fileNamePrefix + fileNameSuffix + '.pdf'

    def saveNet(self, classifier, title, fileName, foldIdx):
        fig, dummy = GLVQPlot.plotAll(classifier,
                                      classifier.prototypes,
                                      classifier.prototypesLabels,
                                      self.trainTestSplitsManager.TrainSamplesLst[foldIdx],
                                      self.trainTestSplitsManager.TrainLabelsLst[foldIdx], GLVQPlot.getDefColors(),
                                      title, plotBoundary=True)
        fig.savefig(fileName, bbox_inches='tight')

    def finishCVCfgIteration(self, foldIdx, classifier, criteria, iterationIdx):
        self.expListener.onFinishFoldCfgIteration(foldIdx)

        if self.expCfg['saveProtoHistogram']:
            protoDist = classifier.getProtoDistribution()
            fig = GLVQPlot.plotProtoHist(protoDist, criteria)
            fig.savefig(self.getFileName('_protoHist', iterationIdx, criteria, foldIdx), bbox_inches='tight')

        if self.expCfg['saveFinalNet'] and self.dataSet.dataDimensions == 2:
            self.saveNet(classifier, criteria, self.getFileName('_finalNet', iterationIdx, criteria, foldIdx), foldIdx)

        #if self.expCfg['exportToOFRFormat']:
        #    self.trainTestSplitsManager.exportToORFFormat(Paths.FeaturesORFDirPrefix() + self.getExpPrefix(self.expCfg, self.trainingSetCfg) + '_' + str(iterationIdx))

    def finishCfgIteration(self, criteria, iterationIdx):
        self.expListener.onFinishCfgIteration(criteria, iterationIdx)

    @staticmethod
    def getPropValue(propName, cfg, expCfg, trainingSetCfg):
        preSet = uuid.uuid1()
        value = preSet
        if propName in expCfg.keys():
            value = expCfg[propName]
        if propName in cfg['classifier'].keys():
            value = cfg['classifier'][propName]
        if propName in trainingSetCfg.keys():
            value = trainingSetCfg[propName]
        if propName in cfg['indiv'].keys():
            value = cfg['indiv'][propName]
        if value == preSet:
            raise ValueError('property name ' + propName + ' not found')
        return value

    def execute(self):
        expName = Experiment.getExpName(self.expCfg, self.trainingSetCfg)
        if expName != '':
            logging.info(expName)
        logging.info(self.trainingSetCfg['dsName'])
        logging.info(str(self.dataSet.dimensions) + ' dimensions')
        logging.info('Label distribution ' + str(self.dataSet.getLabelDistributions()))
        for iterationIdx in range(self.expCfg['iterations']):
            logging.info(str(iterationIdx) + '. iteration')
            self.initIteration(iterationIdx)
            firstCfg = True
            for cfg in self.cfgs:
                epochs = self.expCfg['epochs']
                cfgName= cfg['indiv']['name']
                logging.info(cfgName + ', ' + cfg['classifierName'])
                criteria = cfgName
                self.expListener.onNewCfgIteration(criteria)
                tic = time.time()
                # glvq.initProtosFromSamples(self.trainTestSplit.TrainSamples, self.trainTestSplit.TrainLabels, 20)

                scores = np.array([])
                costValues = np.array([])
                aucValues = np.array([])
                complexities = np.array([])
                scoresDetailed = []
                for foldIdx in range(len(self.trainTestSplitsManager.TrainSamplesLst)):
                    #KNearestLeaveOneOut(self.trainTestSplitsManager.TrainSamplesLst[foldIdx], self.trainTestSplitsManager.TrainLabelsLst[foldIdx])
                    #print self.trainTestSplitsManager.TrainLabelsLst[foldIdx].tolist()
                    if self.expCfg['saveInitialNet'] and self.dataSet.dataDimensions == 2 and firstCfg:
                        initialClassifier = GLVQ(self.dataSet.dataDimensions)
                        initialClassifier.initProtosFromSamples(self.trainTestSplitsManager.TrainSamplesLst[foldIdx], self.trainTestSplitsManager.TrainLabelsLst[foldIdx], numProtoPerLabel=1, shuffle=False)
                        self.saveNet(initialClassifier, 'Initial', self.getFileName('_initialNet', iterationIdx, '', foldIdx), foldIdx)
                    if self.expCfg['exportToOFRFormat']:
                        self.trainTestSplitsManager.exportToORFFormat(Paths.stationaryFeaturesDirPrefix() + self.getExpPrefix(self.expCfg, self.trainingSetCfg) + '_' + str(iterationIdx))

                    self.expListener.onTrainDataChange(self.trainTestSplitsManager.TrainSamplesLst[foldIdx], self.trainTestSplitsManager.TestSamplesLst[foldIdx],
                                                       self.trainTestSplitsManager.TrainLabelsLst[foldIdx], self.trainTestSplitsManager.TestLabelsLst[foldIdx])

                    classifier = self.genNewClassifier(cfg, criteria, foldIdx)
                    if cfg['classifierName'] == 'ILVQ':
                        insertionTimingThresh = self.hyperParams['insertionTimingThresh']
                        if insertionTimingThresh == 0:
                            classifier.initProtosFromSamples(self.trainTestSplitsManager.TrainSamplesLst[foldIdx], self.trainTestSplitsManager.TrainLabelsLst[foldIdx], numProtoPerLabel=200, shuffle=True, cluster=True)

                    if isStationary(self.trainingSetCfg['dsName']):
                        classifier.fit(self.trainTestSplitsManager.TrainSamplesLst[foldIdx], self.trainTestSplitsManager.TrainLabelsLst[foldIdx], epochs)
                        score = classifier.getAccuracy(self.trainTestSplitsManager.TestSamplesLst[foldIdx], self.trainTestSplitsManager.TestLabelsLst[foldIdx])
                    else:
                        predictedTrainLabels = classifier.alternateFitPredict(self.trainTestSplitsManager.TrainSamplesLst[foldIdx], self.trainTestSplitsManager.TrainLabelsLst[foldIdx], np.unique(self.trainTestSplitsManager.TrainLabelsLst[foldIdx]))
                        score = accuracy_score(self.trainTestSplitsManager.TrainLabelsLst[foldIdx], predictedTrainLabels)

                        detailedStep = 10
                        for i in np.arange(0, len(predictedTrainLabels), detailedStep):
                            scoresDetailed.append(accuracy_score(self.trainTestSplitsManager.TrainLabelsLst[foldIdx][i:i+detailedStep], predictedTrainLabels[i:i+detailedStep]))


                    '''for epoch in range(epochs): XXVL
                        for sample in range(len(self.trainTestSplitsManager.TrainSamplesLst[foldIdx])):
                            if self.expCfg['saveNetIntervall'] > 0 and sample > 0 and sample % self.expCfg['saveNetIntervall'] == 0:
                                name = str(sample) + ' samples'
                                self.saveNet(classifier, criteria + ' ' + name, self.getFileName('_' + name, iterationIdx, criteria, foldIdx), foldIdx)
                            classifier.trainInc(self.trainTestSplitsManager.TrainSamplesLst[foldIdx][sample, :], self.trainTestSplitsManager.TrainLabelsLst[foldIdx][sample])'''

                    scores = np.append(scores, score)
                    costValues = np.append(costValues, np.mean(classifier.getCostValues(self.trainTestSplitsManager.TestSamplesLst[foldIdx], self.trainTestSplitsManager.TestLabelsLst[foldIdx])))
                    complexities = np.append(complexities, classifier.getComplexity())
                    #confidences = np.atleast_2d(classifier.predict_confidence2Class(self.trainTestSplitsManager.TestSamplesLst[foldIdx])).T
                    #fpr, tpr, thresholds = metrics.roc_curve(self.trainTestSplitsManager.TestLabelsLst[foldIdx], confidences, pos_label=1)
                    #auc = metrics.auc(fpr, tpr)
                    #aucValues = np.append(aucValues, auc)

                    self.finishCVCfgIteration(foldIdx, classifier, criteria, iterationIdx)
                    #logging.info(classifier.getDetailedClassRateByMatrix(classifier.getDistanceMatrix(self.trainTestSplitsManager.TestSamplesLst[foldIdx], self.trainTestSplitsManager.TestLabelsLst[foldIdx])))



                self.finishCfgIteration(criteria, iterationIdx)
                logging.info(scores)
                logging.info("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean() * 100, scores.std() * 100))
                logging.info("costValues: %0.5f (+/- %0.5f)" % (costValues.mean(), costValues.std()))
                logging.info("complexity: %0.5f (+/- %0.5f)" % (complexities.mean(), complexities.std()))
                #logging.info("auc: %0.5f (+/- %0.5f)" % (aucValues.mean(), aucValues.std()))
                logging.info(classifier.getInfos())
                logging.info(str(time.time() - tic) + " seconds")

                #correctWindowSizes = np.tile(np.arange(1, 3001, 1), 20)
                #correctWindowSizes = 60 * np.ones(shape=len(classifier.windowSizes))
                print 'windowSizes', np.mean(classifier.windowSizes), np.mean(classifier.LTMSizes)
                #print np.sum(np.abs(classifier.windowSizes - correctWindowSizes))/float(len(correctWindowSizes))
                #print np.sum(np.maximum(classifier.windowSizes - correctWindowSizes, 0))/float(len(correctWindowSizes))
                import matplotlib.pyplot as plt
                plt.ioff()
                fig, ax = plt.subplots(5, 1)
                ax[0].plot(np.arange(len(classifier.windowSizes)), classifier.windowSizes, c='r')
                ax[1].plot(np.arange(len(classifier.LTMSizes)), classifier.LTMSizes, c='r')
                ax[2].plot(np.arange(0, len(predictedTrainLabels), detailedStep), scoresDetailed)
                print 'corr STM ', classifier.STMCorrectCount, 'ltm', classifier.LTMCorrectCount,'both ', classifier.BothCorrectCount
                print 'delCount', classifier.allDeletedCount
                #ax[2].plot(range(len(DriftDetection.currentSizeAccs)), DriftDetection.currentSizeAccs, c='r')
                #ax[2].plot(range(len(DriftDetection.smallestSizeAccs)), DriftDetection.smallestSizeAccs, c='b')
                #ax[2].plot(range(len(DriftDetection.largestSizeAccs)), DriftDetection.largestSizeAccs, c='g')
                #ax.plot(np.arange(len(correctWindowSizes)), correctWindowSizes, c='b')
                plt.show()

                #print len(classifier.reducedWindowSizes), np.mean(classifier.reducedWindowSizes), np.std(classifier.reducedWindowSizes)
                #print 'shrinked', DriftDetection.shrinkedCount, DriftDetection.shrinkedCount + DriftDetection.notShrinkedCount, DriftDetection.shrinkedCount/ float(DriftDetection.shrinkedCount + DriftDetection.notShrinkedCount)

                firstCfg = False

            self.expListener.onFinishIteration()
        self.expListener.onFinishExperiment(self.classifiers, Experiment.getExpPrefix(self.expCfg, self.trainingSetCfg))


