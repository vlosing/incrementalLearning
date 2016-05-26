__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
import numpy as np
import logging
from Base.MatlabEngine import MatlabEngine
import matlab

class LPP(BaseClassifier):
    def __init__(self, classifierPerChunk=1):
        self.classifierPerChunk = classifierPerChunk
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
            MatlabEngine.getEng().initLPP(self.classifierPerChunk, matlab.double(self.classes.tolist()), nargout=0)
            self.initialized = True
        samples = matlab.double(samples.tolist())
        labels = matlab.double(labels.tolist())
        MatlabEngine.getEng().fitLPP(samples, labels, nargout=0)

    def alternateFitPredict(self, samples, labels, classes):
        predictedTrainLabels = []
        samples = matlab.double(samples.tolist())
        labels = matlab.double(labels.tolist())
        if not self.initialized:
            self.classes = classes
            MatlabEngine.getEng().clearWorkspace(nargout=0)
            MatlabEngine.getEng().initLPP(self.classifierPerChunk, matlab.double(self.classes.tolist()), nargout=0)
            self.initialized = True
        else:
            predictedLabels = MatlabEngine.getEng().predictLPP(samples, nargout=1)
            predictedTrainLabels = np.array(predictedLabels._data.tolist())

        MatlabEngine.getEng().fitLPP(samples, labels, nargout=0)
        return predictedTrainLabels

    def predict(self, samples):
        samples = matlab.double(samples.tolist())
        predictedLabels = MatlabEngine.getEng().predictLPP(samples, nargout=1)
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