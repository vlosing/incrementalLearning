__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
import numpy as np
import logging
from Base.MatlabEngine import MatlabEngine
import matlab

class IELM(BaseClassifier):
    def __init__(self, activFct='sig', numHiddenNeurons=100):
        self.activFct = activFct
        self.numHiddenNeurons = numHiddenNeurons
        self.initialized = False

    def fit(self, samples, labels, epochs):
        self.classifier.fit(samples, labels)

    def getClassifier(self):
        raise NotImplementedError()

    def getInfos(self):
        return ''

    def partial_fit(self, samples, labels, classes):
        if not self.initialized:
            self.numInputFeatures = np.size(samples, 1)
            self.classes = classes
            MatlabEngine.getEng().clearWorkspace(nargout=0)
            MatlabEngine.getEng().initIELM(matlab.double(self.classes.tolist()), self.numInputFeatures, self.activFct, self.numHiddenNeurons, nargout=0)
            self.initialized = True

        samples = matlab.double(samples.tolist())
        labels = matlab.double(labels.tolist())
        MatlabEngine.getEng().fitIELM(samples, labels, nargout=0)

        predictedTrainLabels = []
        return predictedTrainLabels

    def alternateFitPredict(self, samples, labels, classes):
        predictedTrainLabels = []
        samples = matlab.double(samples.tolist())
        labels = matlab.double(labels.tolist())
        if not self.initialized:
            self.numInputFeatures = np.size(samples, 1)
            self.classes = classes
            MatlabEngine.getEng().clearWorkspace(nargout=0)
            MatlabEngine.getEng().initIELM(matlab.double(self.classes.tolist()), self.numInputFeatures, self.activFct, self.numHiddenNeurons, nargout=0)
            self.initialized = True
        else:
            predictedLabels = MatlabEngine.getEng().predictIELM(samples, nargout=1);
            predictedTrainLabels = np.array(predictedLabels._data.tolist())

        MatlabEngine.getEng().fitIELM(samples, labels, nargout=0)
        return predictedTrainLabels

    def predict(self, samples):
        samples = matlab.double(samples.tolist())
        predictedLabels = MatlabEngine.getEng().predictIELM(samples, nargout=1);
        return np.array(predictedLabels._data.tolist())

    def predict_proba(self, samples):
        raise NotImplementedError()

    def getComplexity(self):
        return self.numHiddenNeurons

    def getComplexityNumParameterMetric(self):
        return self.numHiddenNeurons * self.numInputFeatures + self.numHiddenNeurons * len(self.classes)