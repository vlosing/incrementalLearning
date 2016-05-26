import numpy as np
from sklearn.metrics import accuracy_score

from DataGeneration.DataSetFactory import isStationary
from LVQPY import LVQFactory
from Classifier.GaussianNaiveBayes import GaussianNaiveBayes
from Classifier.SGD import SGD
from Classifier.IELM import IELM
from Classifier.LPP import LPP
from Classifier.LPPNSE import LPPNSE
from Classifier.ISVM import ISVM
from Classifier.ORF import ORF
from Classifier.KNNWindow import KNNWindow
from Classifier.WeightedAvg import WeightedAvg
from moa import trainMOAAlgorithm
from sklearn.neighbors import KNeighborsClassifier


def getTrainSetCfg(dataSetName):
    trainSetSize = 1
    if isStationary(dataSetName):
        trainOrder = 'random'
        splitType = None
    else:
        trainOrder = 'original'
        splitType = 'simple'

    trainingSetCfg = {'dsName': dataSetName, 'splitType': splitType, 'folds': 2, 'trainOrder': trainOrder, 'stratified': False,
                      'shuffle': False, 'chunkSize': 10, 'trainSetSize': trainSetSize}
    return trainingSetCfg

def KNearestLeaveOneOut(samples, labels):
    accuracies = []
    classifier = KNeighborsClassifier(n_neighbors=5)
    for i in range(len(labels)):
        trainSamples = np.delete(samples, i, 0)
        trainLabels = np.delete(labels, i, 0)
        classifier.fit(trainSamples, trainLabels)
        predLabel = classifier.predict(samples[i, :].reshape(1, -1))
        accuracies.append(predLabel == labels[i])
    print np.mean(accuracies)


def getClassifier(classifierName, classifierParams, listener=[]):
    if classifierName == 'ILVQ':
        return LVQFactory.getLVQClassifierByCfg(classifierParams, listener)
    elif classifierName == 'SGD':
        return SGD(eta0=classifierParams['eta0'], learningRate=classifierParams['learningRate'])
    elif classifierName == 'GNB':
        return  GaussianNaiveBayes()
    elif classifierName == 'IELM':
        return IELM(numHiddenNeurons=classifierParams['numHiddenNeurons'])
    elif classifierName == 'LPP':
        return LPP(classifierPerChunk=classifierParams['classifierPerChunk'])
    elif classifierName == 'LPPNSE':
        return LPPNSE()
    elif classifierName == 'ISVM':
        return ISVM(kernel=classifierParams['kernel'], C=classifierParams['C'], sigma=classifierParams['sigma'], maxReserveVectors=classifierParams['windowSize'])
    elif classifierName == 'ORF':
        return ORF(numTrees=classifierParams['numTrees'], numRandomTests=classifierParams['numRandomTests'], maxDepth=classifierParams['maxDepth'], counterThreshold=classifierParams['counterThreshold'])
    elif classifierName == 'KNNWindow':
        return KNNWindow(n_neighbors=classifierParams['nNeighbours'], windowSize=classifierParams['windowSize'], weights=classifierParams['weights'], driftStrategy=classifierParams['driftStrategy'], listener=listener)
    elif classifierName == 'WAVG':
        return WeightedAvg(length=1)

def trainClassifier(classifierName, classifierParams, trainFeatures, trainLabels, testFeatures, testLabels, evaluationStepSize, streamSetting):
    if classifierName in ['LVGB', 'KNNPaw', 'HoeffAdwin', 'DACC']:
        return trainMOAAlgorithm(classifierName, classifierParams, trainFeatures, trainLabels, testFeatures, testLabels, evaluationStepSize, streamSetting)
    else:
        classifier = getClassifier(classifierName, classifierParams)

    allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric = classifier.trainAndEvaluate(trainFeatures, trainLabels,
                                    testFeatures,
                                    evaluationStepSize,
                                    streamSetting=streamSetting)
    #classifier.plotCostCourse()
    return allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric

def updateClassifierEvaluations(classifierEvaluations, classifierName, trainLabels, testLabels, allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric,
                                chunkSize, criterion, driftSetting):
    if not driftSetting:
        testAccuracyScores = []
        for classifiedLabels in allPredictedTestLabels:
            classifiedLabels = np.array(classifiedLabels).astype(np.int)
            testAccuracyScores.append(accuracy_score(testLabels, classifiedLabels))
        print testAccuracyScores, np.mean(testAccuracyScores)
        print complexities
        if classifierEvaluations['values'].has_key(classifierName):
            if classifierEvaluations['values'][classifierName].has_key(criterion):
                if classifierEvaluations['values'][classifierName][criterion].has_key('testAccuracies'):
                    classifierEvaluations['values'][classifierName][criterion]['testAccuracies'].append(testAccuracyScores)
                else:
                    classifierEvaluations['values'][classifierName][criterion]['testAccuracies'] = [testAccuracyScores]
            else:
                classifierEvaluations['values'][classifierName][criterion] = {}
                classifierEvaluations['values'][classifierName][criterion]['testAccuracies'] = [testAccuracyScores]
        else:
            classifierEvaluations['values'][classifierName] = {}
            classifierEvaluations['values'][classifierName][criterion] = {}
            classifierEvaluations['values'][classifierName][criterion]['testAccuracies'] = [testAccuracyScores]

    else:
        trainPredictionAccuracyScores = []
        idx = 0
        weights = []
        for classifiedLabels in allPredictedTrainLabels:
            trainPredictionAccuracyScores.append(accuracy_score(trainLabels[idx:idx+len(classifiedLabels)], classifiedLabels))
            idx += len(classifiedLabels)
            weights.append(len(classifiedLabels))
        if classifierName == 'LPPNSE' and len(trainPredictionAccuracyScores) > 0:
            trainPredictionAccuracyScores[0] = trainPredictionAccuracyScores[1]
        meanAcc = np.average(trainPredictionAccuracyScores, weights=weights)
        print np.round(trainPredictionAccuracyScores, 4), np.round(meanAcc, 4)
        print complexities
        if classifierEvaluations['values'].has_key(classifierName):
            if classifierEvaluations['values'][classifierName].has_key(criterion):
                if classifierEvaluations['values'][classifierName][criterion].has_key('trainPredictionAccuracies'):
                    classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'].append(trainPredictionAccuracyScores)
                    classifierEvaluations['values'][classifierName][criterion]['meanAcc'].append(meanAcc)
                else:
                    classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'] = [trainPredictionAccuracyScores]
                    classifierEvaluations['values'][classifierName][criterion]['meanAcc']= [meanAcc]
            else:
                classifierEvaluations['values'][classifierName][criterion] = {}
                classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'] = [trainPredictionAccuracyScores]
                classifierEvaluations['values'][classifierName][criterion]['meanAcc']= [meanAcc]
        else:
            classifierEvaluations['values'][classifierName] = {}
            classifierEvaluations['values'][classifierName][criterion] = {}
            classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'] = [trainPredictionAccuracyScores]
            classifierEvaluations['values'][classifierName][criterion]['meanAcc'] = [meanAcc]

    if classifierEvaluations['values'].has_key(classifierName):
        if classifierEvaluations['values'][classifierName].has_key(criterion):
            if classifierEvaluations['values'][classifierName][criterion].has_key('complexities'):
                classifierEvaluations['values'][classifierName][criterion]['complexities'].append(complexities)
            else:
                classifierEvaluations['values'][classifierName][criterion]['complexities'] = [complexities]
        else:
            classifierEvaluations['values'][classifierName][criterion] = {}
            classifierEvaluations['values'][classifierName][criterion]['complexities'] = [complexities]
    else:
        classifierEvaluations['values'][classifierName] = {}
        classifierEvaluations['values'][classifierName][criterion] = {}
        classifierEvaluations['values'][classifierName][criterion]['complexities'] = [complexities]

    if classifierEvaluations['values'].has_key(classifierName):
        if classifierEvaluations['values'][classifierName].has_key(criterion):
            if classifierEvaluations['values'][classifierName][criterion].has_key('complexitiesNumParamMetric'):
                classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'].append(complexityNumParameterMetric)
            else:
                classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'] = [complexityNumParameterMetric]
        else:
            classifierEvaluations['values'][classifierName][criterion] = {}
            classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'] = [complexityNumParameterMetric]
    else:
        classifierEvaluations['values'][classifierName] = {}
        classifierEvaluations['values'][classifierName][criterion] = {}
        classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'] = [complexityNumParameterMetric]
    return classifierEvaluations




