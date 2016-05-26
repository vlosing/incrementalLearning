__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
import numpy as np
import logging
from Base.MatlabEngine import MatlabEngine
import matlab

class LPPNSE(BaseClassifier):
    def __init__(self, sigmoidSlope=0.5, sigmoidCutOff=10, errorThreshold=0.01):
        self.sigmoidSlope = sigmoidSlope
        self.sigmoidCutOff = sigmoidCutOff
        self.errorThreshold = errorThreshold
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
            MatlabEngine.getEng().clearWorkspace(nargout=0)
            MatlabEngine.getEng().initLPPNSE(matlab.double(self.classes.tolist()), self.sigmoidSlope, self.sigmoidCutOff, self.errorThreshold, nargout=0)
            self.initialized = True
        samples = matlab.double(samples.tolist())
        labels = matlab.double(labels.tolist())
        MatlabEngine.getEng().fitLPPNSE(samples, labels, nargout=0)

    def alternateFitPredict(self, samples, labels, classes):
        samples = matlab.double(samples.tolist())
        labels = matlab.double(labels.tolist())
        if not self.initialized:
            self.classes = classes
            MatlabEngine.getEng().clearWorkspace(nargout=0)
            MatlabEngine.getEng().initLPPNSE(matlab.double(self.classes.tolist()), self.sigmoidSlope, self.sigmoidCutOff, self.errorThreshold, nargout=0)
            self.initialized = True
        predictedLabels = MatlabEngine.getEng().predictLPPNSE(samples, nargout=1)
        predictedTrainLabels = np.array(predictedLabels._data.tolist())
        MatlabEngine.getEng().fitLPPNSE(samples, labels, nargout=0)
        return predictedTrainLabels

    def predict(self, samples):
        samples = matlab.double(samples.tolist())
        predictedLabels = MatlabEngine.getEng().predictLPPNSE(samples, nargout=1)
        return np.array(predictedLabels._data.tolist())

    def predict_proba(self, samples):
        raise NotImplementedError()

    def getComplexity(self):
        return MatlabEngine.getEng().getNumberOfNodes(nargout=1)

    def getComplexityNumParameterMetric(self):
        numNodes = MatlabEngine.getEng().getNumberOfNodes(nargout=1)
        numLeaves = MatlabEngine.getEng().getNumberOfLeaves(nargout=1)
        numClassifier = MatlabEngine.getEng().getNumberOfClassifiers(nargout=1)
        return numNodes + (numNodes - numLeaves) * 2 + numLeaves + numClassifier