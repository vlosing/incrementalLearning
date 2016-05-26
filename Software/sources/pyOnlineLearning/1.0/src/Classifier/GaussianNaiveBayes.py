__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

class GaussianNaiveBayes(BaseClassifier):
    def __init__(self):
        self.classifier = GaussianNB()

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
            if not hasattr(self.classifier, 'theta_'):
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
        return self.classifier.theta_.shape[0]

    def getComplexityNumParameterMetric(self):
        return len(np.ravel(self.classifier.theta_)) * 2
