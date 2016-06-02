import logging
from DataGeneration.TrainTestSplitsManager import TrainTestSplitsManager
from DataGeneration import DataSetFactory
from HyperParameter.hyperParameterFactory import getHyperParams
from Base import Paths
from ClassificationExperiment.Experiment import Experiment
import numpy as np
from auxiliaryFunctions import trainClassifier, updateClassifierEvaluations, getTrainSetCfg
from DataGeneration.DataSetFactory import isStationary
import json
from comparisonPlot import doComparisonPlot, dataToCSV
from HyperParameter.hyperParameterTuning import determineHyperParams
from Base.MatlabEngine import MatlabEngine

def getBootstrapSample(samples, labels):
    indices = np.random.randint(len(labels), size=len(labels))
    indices = np.sort(indices)
    return samples[indices, :], labels[indices]

def permutateDataset(samples, labels):
    indices = np.random.permutation(len(labels))
    return samples[indices, :], labels[indices]

def doComparison(dataSetName, classifierNames, iterations, criterionName, criteria, scaleData=True, hyperParamTuning=False, bootStrapSampling=False, permutate=False, chunkSize=100, LVQInsertionTimingThreshs=[], iELMNumHiddenNeurons=[]):
    trainSetCfg = getTrainSetCfg(dataSetName)
    dataSet = DataSetFactory.getDataSet(trainSetCfg['dsName'])
    dataSet.printDSInformation()
    trainTestSplitsManager = TrainTestSplitsManager(dataSet.samples, dataSet.labels, featureFileNames=dataSet.featureFileNames,
                                                 dataOrder=trainSetCfg['trainOrder'], chunkSize=trainSetCfg['chunkSize'],
                                                 shuffle=trainSetCfg['shuffle'], stratified=trainSetCfg['stratified'],
                                                 splitType=trainSetCfg['splitType'], numberOfFolds=trainSetCfg['folds'],
                                                 trainSetSize=trainSetCfg['trainSetSize'], testSamples=dataSet.testSamples, testLabels=dataSet.testLabels)

    evalFilePrefix = Experiment.getExpPrefix2(None, trainSetCfg)
    evalDstDirectory = Paths.FeatureTmpsDirPrefix() + 'Evaluations/'
    streamSetting = not isStationary(dataSetName)
    logging.info('stream setting' if streamSetting  else 'train/test setting')
    logging.info('scaleData %d, bootStrapSampling %d, permutate %d' % (scaleData, bootStrapSampling, permutate))

    classifierEvaluations = {}
    for iteration in np.arange(iterations):
        logging.info('iteration %d/%d'% (iteration+1, iterations))
        trainTestSplitsManager.generateSplits()
        if scaleData:
            trainTestSplitsManager.scaleData(maxSamples=1000)
        trainSamples = trainTestSplitsManager.TrainSamplesLst[0]
        trainLabels = trainTestSplitsManager.TrainLabelsLst[0]
        if permutate:
            trainSamples, trainLabels = permutateDataset(trainSamples, trainLabels)
        if bootStrapSampling:
            trainSamples, trainLabels = getBootstrapSample(trainSamples, trainLabels)
        numTrainSamples = len(trainLabels)
        logging.info('train-samples %d' % numTrainSamples)
        logging.info('test-samples %d' % len(trainTestSplitsManager.TestSamplesLst[0]))
        if iteration == 0:
            classifierEvaluations['meta'] = {'dataSetName':dataSetName, 'numTrainSamples': len(trainLabels)}
            classifierEvaluations['values'] = {}
            hyperParams = {}
            for classifierName in classifierNames:
                hyperParams[classifierName] = getHyperParams(trainSetCfg['dsName'], classifierName, scaleData)
                if hyperParamTuning:
                    hyperParams[classifierName] = determineHyperParams(classifierName, trainSamples[:50000, :], trainLabels[:50000], evalDstDirectory + 'hyperParameter/', dataSetName, hyperParams[classifierName], streamSetting, scaleData)


        for critierion in criteria:
            if criterionName=='chunkSize':
                chunkSize = critierion
            splits = int(np.ceil(numTrainSamples / float(chunkSize)))
            logging.info('evaluationStepSize %d splits %d' % (chunkSize, splits))

            for classifierName in classifierNames:
                classifierParams = hyperParams[classifierName]
                if classifierName in ['ILVQ', 'ISVM', 'KNNPaw', 'KNNWindow']:
                    #classifierParams['windowSize'] = 10000
                    classifierParams['windowSize'] = min(5000, int(0.1 * len(trainLabels)))
                    #classifierParams['insertionTimingThresh'] = 10
                if criterionName == 'complexity':
                    if classifierName == 'LVQ':
                        classifierParams['insertionTimingThresh'] = LVQInsertionTimingThreshs[critierion]
                    elif classifierName == 'iELM':
                        classifierParams['numHiddenNeurons'] = iELMNumHiddenNeurons[critierion]
                logging.info(classifierName + ' ' + str(classifierParams))
                allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric = trainClassifier(classifierName, classifierParams, trainSamples, trainLabels, trainTestSplitsManager.TestSamplesLst[0], trainTestSplitsManager.TestLabelsLst[0], chunkSize, streamSetting)
                classifierEvaluations = updateClassifierEvaluations(classifierEvaluations, classifierName, trainLabels, trainTestSplitsManager.TestLabelsLst[0], allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric, chunkSize, critierion, streamSetting)
    json.encoder.FLOAT_REPR = lambda o: format(o, '.5f')
    json.dump(classifierEvaluations, open(evalDstDirectory + evalFilePrefix + 'evaluations.json', 'w'))
    #dataToCSV(dataSetName)

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    #randomState = 0
    #np.random.seed(randomState)

    #criterionName = 'complexity'
    #criteria = np.arange(6)
    #LVQInsertionTimingThreshs = [100, 50, 10, 7, 5, 1]
    #iELMNumHiddenNeurons = [10, 50, 100, 150, 200, 300]
    #chunkSize = 400

    criterionName = 'chunkSize'

    #
    #criteria = [50, 100, 250, 500, 1000]
    #criteria = [100, 500, 1000]
    criteria = [100000]


    #classifierNames = ['LVGB', 'ILVQ', 'ORF', 'LPP', 'LPPNSE', 'SGD', 'IELM', 'GNB', 'ISVM']
    #classifierNames = ['ILVQ', 'LVGB', 'ORF', 'LPP', 'SGD', 'IELM', 'ISVM']
    #classifierNames = ['IELM', 'ILVQ', 'LPP']
    #classifierNames = ['SGD', 'ILVQ', 'ORF', 'LVGB']
    #classifierNames = ['ILVQ', 'LVGB']
    #classifierNames = ['LPPNSE', 'LVGB', 'ILVQ', 'SGD', 'ORF']
    #classifierNames = ['LPPNSE', 'LVGB', 'ILVQ', 'SGD']
    #classifierNames = ['KNNPaw', 'HoeffAdwin', 'LVGB']
    #classifierNames = ['KNNPaw', 'HoeffAdwin', 'LVGB', 'DACC']

    #classifierNames = ['KNNWindow']
    classifierNames = ['KNNPaw', 'LVGB', 'DACC']
    iterations = 1
    scaleData = False
    bootStrapSampling=False
    permutate=False


    #doComparison('weather', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData, bootStrapSampling=bootStrapSampling, permutate=permutate)
    #doComparison('elec', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData, bootStrapSampling=bootStrapSampling)
    #doComparison('covType', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData, bootStrapSampling=bootStrapSampling)
    #doComparison('outdoorStream', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData, bootStrapSampling=bootStrapSampling)
    #doComparison('rialto', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData, bootStrapSampling=bootStrapSampling)

    #doComparison('sea', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('rbfSlowXXL', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('hypSlowXXL', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('squaresIncrXXL', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData, bootStrapSampling=bootStrapSampling)
    doComparison('rbfAbruptXXL', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('chessVirtualXXL', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData, bootStrapSampling=bootStrapSampling, permutate=permutate)
    #doComparison('allDriftXXL', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('chessIIDXXL', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData, bootStrapSampling=bootStrapSampling, permutate=permutate)

    #doComparison('chessFields', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData, bootStrapSampling=bootStrapSampling, permutate=permutate)
    #doComparison('rbfAbruptSmall', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)


    #doComparison('rbfAbruptSmall', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('chessVirtual', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('allDrift', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)

    #doComparison('squaresIncr', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)

    #doComparison('rbfSlow', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('hypSlow', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)

    ###stationary###
    #doComparison('border', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('overlap', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('coil', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('outdoor', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('USPS', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('DNA', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('isolet', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('letter', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('satImage', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('penDigits', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('HAR', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)

    #doComparison('gisette', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('mnist', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('news20', classifierNames, iterations, criterionName, criteria, hyperParamTuning=True, scaleData=scaleData)

    ###nonstationary###
    #doComparison('weather', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('elec', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('spam', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('covType', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('keystroke', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)

    #doComparison('sea', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('rbfSlow', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)

    #doComparison('souza2CHT', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('souza4CREV1', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('souzaGears2C2D', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('cbConst', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('cbSinus', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)

    #doComparison('rbfAbrupt', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('rectGradual', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)

    #doComparison('souza2CDT', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('souzaFG2C2D', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)
    #doComparison('hyperplaneSlow', classifierNames, iterations, criterionName, criteria, hyperParamTuning=False, scaleData=scaleData)




    MatlabEngine.stop()