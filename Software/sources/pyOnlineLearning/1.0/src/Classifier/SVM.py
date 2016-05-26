__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
from sklearn import svm
import numpy as np


class SVM(BaseClassifier):
    def __init__(self, C=1.0, kernel='rbf', gamma=0.0, probability=True):
        self.classifier = svm.SVC(C=C, kernel=kernel, degree=3, gamma=gamma, coef0=0.0, shrinking=True,
                                  probability=probability, tol=0.001, cache_size=200, class_weight=None,
                                  verbose=False, max_iter=-1, random_state=None)

    def fit(self, samples, labels, epochs):
        self.classifier.fit(samples, labels)

    def getInfos(self):
        return 'no infos'

    def predict(self, samples):
        return self.classifier.predict(samples)

    def predict_proba(self, samples):
        predictions = self.classifier.predict_proba(samples)
        return predictions

    def predict_confidence(self, samples):
        predictions = self.classifier.predict_proba(samples)
        return np.argmax(predictions, axis=1), np.max(predictions, axis=1)

    def getComplexity(self):
        return len(self.classifier.support_)
