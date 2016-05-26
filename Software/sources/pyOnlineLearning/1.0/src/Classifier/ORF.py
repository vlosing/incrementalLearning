__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
import numpy as np
import liborfPythonIntf

class ORF(BaseClassifier):
    def __init__(self, numTrees=10, numRandomTests=100, maxDepth=50, counterThreshold=10):
        self.numTrees = int(numTrees)
        self.numRandomTests = int(numRandomTests)
        self.maxDepth = int(maxDepth)
        self.counterThreshold = int(counterThreshold)
        self.initialized = False

    def fit(self, samples, labels, epochs):
        raise NotImplementedError()

    def getClassifier(self):
        raise NotImplementedError()

    def getInfos(self):
        return ''

    def partial_fit(self, samples, labels, classes):
        samples = samples.astype(np.float)
        self.alternateFitPredict(samples, labels, classes)

    def alternateFitPredict(self, samples, labels, classes):
        if not self.initialized:
            self.classes = classes
            self.numInputFeatures = np.size(samples, 1)
            liborfPythonIntf.initORF(self.numTrees, self.numRandomTests, self.maxDepth, self.counterThreshold, len(self.classes), self.numInputFeatures)
            self.initialized = True
        predictedTrainLabels = liborfPythonIntf.fitORF(samples, labels, len(self.classes))
        return predictedTrainLabels


    def predict(self, samples):
        samples = samples.astype(np.float)
        predictedLabels = liborfPythonIntf.predictORF(samples, len(self.classes))
        return predictedLabels

    def predict_proba(self, samples):
        raise NotImplementedError()

    def getComplexity(self):
        return liborfPythonIntf.getComplexity()

    def getComplexityNumParameterMetric(self):
        return liborfPythonIntf.getComplexityNumParametric()
