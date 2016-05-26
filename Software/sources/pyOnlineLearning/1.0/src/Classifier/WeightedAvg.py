__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import logging

class WeightedAvg(BaseClassifier):
    def __init__(self, length=1):
        self.length = length
        self.lastLabels = np.empty(shape=(0, 1), dtype=int)
    def fit(self, samples, labels, epochs):
        raise NotImplementedError

    def getClassifier(self):
        raise NotImplementedError

    def getInfos(self):
        return ''

    def partial_fit(self, samples, labels, classes):
        self.lastLabels = np.append(labels, self.lastLabels)
        if len(self.lastLabels) > self.length:
            self.lastLabels = np.delete(self.lastLabels, self.length, 0)

    def alternateFitPredict(self, samples, labels, classes):
        predictedTrainLabels = []
        for j in range(len(samples)):
            if len(self.lastLabels) == 0:
                predictedTrainLabels.append(0)
            else:
                predictedTrainLabels.append(self.predict(samples[j,:].reshape(1, -1)))
            self.partial_fit(samples[j,:].reshape(1, -1), np.atleast_1d(labels[j]), classes)
        return predictedTrainLabels

    def predict(self, samples):
        labels, counts =  np.unique(self.lastLabels, return_counts=True)
        majorityLabel = labels[np.argmax(counts)]
        return np.repeat([majorityLabel], samples.shape[0])

    def predict_proba(self, samples):
        predictions = self.classifier.predict_proba(samples)
        return predictions

    def getComplexity(self):
        return 0

    def getComplexityNumParameterMetric(self):
        return 0