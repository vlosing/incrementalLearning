import json

from sklearn import svm
from sklearn import grid_search
import numpy as np

from Base import Paths


def getLabelMapping(path):
    f = open(path, 'r')
    featureFileNames = np.array(json.load(f))
    f.close()
    labelMapping = {}
    counter = 0
    for i in np.unique(featureFileNames):
        labelMapping[i] = counter
        counter += 1
    return labelMapping


def getMappedLabels(labelMapping, Y):
    mappedLabels = np.empty(shape=(0, 1))
    for i in Y:
        mappedLabels = np.append(mappedLabels, labelMapping[i])
    return mappedLabels


def SVMExperiment(dataSet, dataSetOrder, orderTrainValue, labelsPath):
    config = dataPrep.DataSetConfig(dataSet, dataSetOrder, orderTrainValue, 0)
    labelMapping = getLabelMapping(labelsPath)
    SVMModels = trainSVM(config.TrainSamples, config.TrainLabels, labelMapping)
    print 'train ' + str(getClassRate(SVMModels, labelMapping, config.TrainSamples, config.TrainLabels))
    print 'test ' + str(getClassRate(SVMModels, labelMapping, config.TestSamples, config.TestLabels))


def trainSVM(X, Y, labelMapping):
    mappedY = getMappedLabels(labelMapping, Y)
    SVMModels = {}
    numVectors = 0
    for i in np.unique(mappedY):
        parameters = {'C': [pow(2, 10)], 'gamma': [pow(2, 3)], 'kernel': ['rbf']}
        # parameters = {'C':[100000], 'kernel':['linear']}
        svc = svm.SVC(cache_size=10000, probability=True)
        clf = grid_search.GridSearchCV(svc, parameters, scoring='f1')
        tmpY = mappedY.copy()
        tmpY[mappedY == i] = 1
        tmpY[mappedY != i] = 0
        if np.sum(tmpY) > 1:
            clf.fit(X, tmpY)
            #print clf.best_estimator_.n_support_
            numVectors += clf.best_estimator_.n_support_
            #print "best params: "+str(clf.best_params_)
            SVMModels[i] = clf
    print 'numVectors ' + str(numVectors)
    return SVMModels


def getClassRate(SVMModels, labelMapping, X, Y):
    mappedY = getMappedLabels(labelMapping, Y)

    classificationsTrain = np.zeros(shape=(len(X), len(labelMapping.keys())))
    for key in SVMModels.keys():
        classificationsTrain[:, int(key)] = SVMModels[key].predict_proba(X)[:, 1]
    return np.sum(np.argmax(classificationsTrain, axis=1) == mappedY) / float(len(mappedY))


if __name__ == '__main__':
    # SVMExperiment('OutdoorEasy', 'random', 0.7,Paths.outdoorEasyLabelsFileName())
    #SVMExperiment('OutdoorEasy', 'chunksRandomEqualCountPerLabel', 0.1, Paths.outdoorEasyLabelsFileName())
    SVMExperiment('COIL', 'randomRegular', 9, Paths.coilLabelsFileName())
