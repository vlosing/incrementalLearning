__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import logging

class SGD(BaseClassifier):
    def __init__(self, eta0=0, learningRate='constant'):
        self.classifier = SGDClassifier(eta0=eta0, learning_rate=learningRate)
    def fit(self, samples, labels, epochs):
        self.classifier.fit(samples, labels)

    def getClassifier(self):
        return self.classifier

    def getInfos(self):
        return ''

    def partial_fit(self, samples, labels, classes):
        self.classifier.partial_fit(samples, np.atleast_1d(labels), classes)

    def alternateFitPredict(self, samples, labels, classes):
        predictedTrainLabels = []
        for j in range(len(samples)):
            if self.classifier.coef_ is None:
                predictedTrainLabels.append(-1)
            else:
                predictedTrainLabels.append(self.classifier.predict(samples[j,:].reshape(1, -1)))
            self.classifier.partial_fit(samples[j,:].reshape(1, -1), np.atleast_1d(labels[j]), classes)
        return predictedTrainLabels

    def predict(self, samples):
        return self.classifier.predict(samples)

    def predict_proba(self, samples):
        predictions = self.classifier.predict_proba(samples)
        return predictions

    def getComplexity(self):
        return self.classifier.coef_.shape[0]

    def getComplexityNumParameterMetric(self):
        return len(np.ravel(self.classifier.coef_)) + len(np.ravel(self.classifier.intercept_))
