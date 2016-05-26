__author__ = 'vlosing'
from ClassifierCommon.BaseClassifier import BaseClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import logging
class RandomForest(BaseClassifier):
    def __init__(self, numEstimators=10, max_depth=None, max_features='auto', criterion='gini'):
        self.classifier = RandomForestClassifier(n_estimators=numEstimators,
                                                 criterion=criterion,
                                                 max_depth=max_depth,
                                                 min_samples_split=2,
                                                 min_samples_leaf=1,
                                                 min_weight_fraction_leaf=0.0,
                                                 max_features=max_features,
                                                 max_leaf_nodes=None, bootstrap=True,
                                                 oob_score=False,
                                                 n_jobs=-1,
                                                 random_state=None,
                                                 verbose=0,
                                                 warm_start=False,
                                                 class_weight=None)

    def fit(self, samples, labels, epochs):
        self.classifier.fit(samples, labels)

    def getClassifier(self):
        return self.classifier

    def getInfos(self):

        return self.classifier.feature_importances_

    def predict(self, samples):
        return self.classifier.predict(samples)

    def predict_proba(self, samples):
        if len(samples) == 0:
            logging.warning('no samples to predict')
        else:
            predictions = self.classifier.predict_proba(samples)
            return predictions

    def predict_confidence(self, samples):
        if len(samples) == 0:
            logging.warning('no samples to predict')
        else:
            predictions = self.classifier.predict_proba(samples)
            return np.argmax(predictions, axis=1), np.max(predictions, axis=1)

    def predict_confidence2Class(self, samples):
        if len(samples) == 0:
            logging.warning('no samples to predict')
        else:
            predictions = self.classifier.predict_proba(samples)
            return predictions[:, 1]

    def getComplexity(self):
        return -1
