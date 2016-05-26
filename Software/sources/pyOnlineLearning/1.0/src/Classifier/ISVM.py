__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
import numpy as np
import logging
from Base.MatlabEngine import MatlabEngine
import matlab

class ISVM(BaseClassifier):
    def __init__(self, kernel = 'RBF', C=1, sigma=0.1, maxReserveVectors = 500):
        self.kernel = kernel
        self.C = C
        self.sigma = sigma
        self.maxReserveVectors = maxReserveVectors
        self.initialized = False

    def fit(self, samples, labels, epochs):
        raise NotImplementedError()

    def getClassifier(self):
        raise NotImplementedError()

    def getInfos(self):
        return ''

    def partial_fit(self, samples, labels, classes):
        if not self.initialized:
            self.classes = classes
            self.numInputFeatures = np.size(samples, 1)
            MatlabEngine.getEng().initISVM(matlab.double(self.classes.tolist()), nargout=0)
            self.initialized = True
        samples = matlab.double(samples.T.tolist())
        labels = matlab.double(labels.tolist())
        MatlabEngine.getEng().fitISVM(samples, labels, self.kernel, self.C, self.sigma, self.maxReserveVectors, nargout=0)
        predictedTrainLabels = []
        return predictedTrainLabels

    def alternateFitPredict(self, samples, labels, classes):
        if not self.initialized:
            self.classes = classes
            self.numInputFeatures = np.size(samples, 1)
            MatlabEngine.getEng().initISVM(matlab.double(self.classes.tolist()), nargout=0)
            self.initialized = True
        samples = matlab.double(samples.T.tolist())
        labels = matlab.double(labels.tolist())
        predictedTrainLabels = MatlabEngine.getEng().alternateFitPredictISVM(samples, labels, self.kernel, self.C, self.sigma, self.maxReserveVectors, nargout=1)
        return np.array(predictedTrainLabels._data.tolist())

    def predict(self, samples):
        samples = matlab.double(samples.T.tolist())
        predictedLabels = MatlabEngine.getEng().predictISVM(samples, nargout=1)
        return np.array(predictedLabels._data.tolist())

    def predict_proba(self, samples):
        raise NotImplementedError()

    def getComplexity(self):
        return MatlabEngine.getEng().getNumSV(nargout=1)

    def getComplexityNumParameterMetric(self):
        return MatlabEngine.getEng().getNumSV(nargout=1) * (self.numInputFeatures + 1) + 1