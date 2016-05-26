__author__ = 'vlosing'
import numpy as np
from sklearn.metrics import log_loss
import logging
import numpy as np
import pandas as pd
from Base.MatlabEngine import MatlabEngine
from datetime import datetime
class BaseClassifier(object):
    def __init__(self):
        pass

    def fit(self, samples, labels, epochs):
        raise NotImplementedError()

    def partial_fit(self, samples, labels, classes):
        raise NotImplementedError()

    def alternateFitPredict(self, samples, labels, classes):
        raise NotImplementedError()

    def predict(self, samples):
        raise NotImplementedError()

    def predict_proba(self, samples):
        raise NotImplementedError()

    def getInfos(self):
        raise NotImplementedError()

    def getComplexity(self):
        raise NotImplementedError()

    def getComplexityNumParameterMetric(self):
        raise NotImplementedError()

    @staticmethod
    def getLabelAccuracy(predictedY, groundTruthY):
        for label in np.unique(groundTruthY):
            labelIndices = np.where(groundTruthY == label)[0]
            labelAcc = np.sum(predictedY[labelIndices] == groundTruthY[labelIndices]) / float(len(labelIndices))
            print label, labelAcc


    def getAccuracy(self, samples, labels, scoreType='acc', labelSpecific=False):
        if len(samples) == 0:
            logging.warning('no samples to predict')
        elif scoreType == 'acc':
            classifiedLabels = np.array(self.predict(samples))
            acc = np.sum(classifiedLabels == labels)/float(len(labels))
            if labelSpecific:
                BaseClassifier.getLabelAccuracy(classifiedLabels, labels)
                return acc
            else:
                return acc

        elif scoreType == 'logLoss':
            classifiedLabels = self.predict_proba(samples)
            return log_loss(labels, classifiedLabels, eps=1e-15, normalize=True, sample_weight=None)
        else:
            raise NameError('unknown scoreType ' + scoreType)

    def getCostValues(self, samples, labels):
        return -1

    def trainFromFileAndEvaluate(self, trainFeaturesFileName, trainLabelsFileName,
                                 testFeaturesFileName,
                                 evaluationStepSize, evalDstPathPrefix, splitTestfeatures):

        trainFeatures = pd.read_csv(trainFeaturesFileName, sep=' ', header=None, skiprows=1).values
        trainLabels = pd.read_csv(trainLabelsFileName, header=None, skiprows=1).values.ravel()


        if testFeaturesFileName != '':
            testFeatures = pd.read_csv(testFeaturesFileName, sep=' ', header=None, skiprows=1).values

        self.trainAndEvaluate(trainFeatures, trainLabels, testFeatures, evaluationStepSize, splitTestfeatures)

    def trainAndEvaluate(self, trainFeatures, trainLabels,
                        testFeatures,
                        evaluationStepSize, streamSetting=False):

        splitIndices = np.arange(evaluationStepSize, len(trainLabels), evaluationStepSize)

        trainFeaturesChunks = np.array_split(trainFeatures, splitIndices)
        trainLabelsChunks = np.array_split(trainLabels, splitIndices)
        complexities = []
        complexityNumParameterMetric = []
        classes = np.unique(trainLabels)
        allPredictedTrainLabels = []
        allPredictedTestLabels = []
        for i in range(len(trainFeaturesChunks)):
            print "chunk %d/%d %s" % (i+1, len(trainFeaturesChunks), datetime.now().strftime('%H:%M:%S'))
            if streamSetting:
                predictedTrainLabels = self.alternateFitPredict(trainFeaturesChunks[i], trainLabelsChunks[i], classes)
                allPredictedTrainLabels.append(predictedTrainLabels)
            else:
                self.partial_fit(trainFeaturesChunks[i], trainLabelsChunks[i], classes)

            if len(testFeatures) > 0:
                predictedLabels = self.predict(testFeatures)
                allPredictedTestLabels.append(predictedLabels)
            complexities.append(self.getComplexity())
            complexityNumParameterMetric.append(self.getComplexityNumParameterMetric())
        return allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric
