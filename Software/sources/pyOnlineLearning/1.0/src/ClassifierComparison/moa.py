__author__ = 'viktor'
import numpy as np
from Base import Paths
import subprocess
import os
import uuid

def getLVGBComplexityNumParameterMetric(numLeaves, numNodes, numAttributes, numClasses):
    return numNodes + (numNodes - numLeaves) * 2 + numLeaves + numAttributes * numClasses * 2

def trainMOAAlgorithm(classifierName, classifierParams, trainFeatures, trainLabels, testFeatures, testLabels, evaluationStepSize, streamSetting):
    allPredictedTestLabels = []
    allPredictedTrainLabels = []
    complexities = []
    complexityNumParameterMetric = []
    uuidKey = uuid.uuid4().hex
    labelOutputFileName = Paths.FeatureTmpsDirPrefix() + 'outputLabels' + uuidKey + '.txt'
    metaOutputFileName = Paths.FeatureTmpsDirPrefix() + 'outputMeta' + uuidKey + '.csv'
    ARFFTrainFileName = Paths.FeatureTmpsDirPrefix() + 'train' + uuidKey + '.arff'

    generateARFFFile(trainFeatures, trainLabels, ARFFTrainFileName)

    if classifierName == 'LVGB':
        numClassifier = classifierParams['numClassifier']
        classifierStr = 'meta.LeveragingBag -l (trees.HoeffdingTree -g %d -c %.7f -t %f) -s %d' % (classifierParams['gracePeriod'],
                                                                                                classifierParams['splitConfidence'],
                                                                                                classifierParams['tieThresh'],
                                                                                                numClassifier)

    elif classifierName == 'KNNPaw':
        classifierStr = 'lazy.kNNwithPAWandADWIN -k 5 -w %d' % (classifierParams['windowSize'])
        #classifierStr = 'lazy.kNNwithPAW -k 5 -w %d' % (classifierParams['windowSize'])
        #classifierStr = 'lazy.kNN -k 5 -w %d' % (classifierParams['windowSize'])
    elif classifierName == 'HoeffAdwin':
        numClassifier = 1
        classifierStr = 'trees.HoeffdingAdaptiveTree -g %d -c %.7f -t %f' % (classifierParams['gracePeriod'],
                                                                             classifierParams['splitConfidence'],
                                                                             classifierParams['tieThresh'])
    elif classifierName == 'DACC':
#        classifierStr = 'meta.DACC'
         classifierStr = 'meta.DACC -l trees.HoeffdingTree -n 10.0'
    if not streamSetting:
        ARFFTestFileName = Paths.FeatureTmpsDirPrefix() + 'test' + uuidKey + '.arff'
        generateARFFFile(testFeatures, testLabels, ARFFTestFileName)
        splits = int(np.ceil(len(trainLabels) / float(evaluationStepSize)))
        step = 0
        for i in np.arange(splits):
            step += evaluationStepSize
            step = min(len(trainLabels), step)
            if os.path.isfile(labelOutputFileName):
                os.remove(labelOutputFileName)
            if os.path.isfile(metaOutputFileName):
                os.remove(metaOutputFileName)
            cmd = ['java -Xmx35000m -cp %s -javaagent:%s moa.DoTask "EvaluateModel -m (LearnModel -l (%s) -s (ArffFileStream -f %s) -m %d -p 1) -s (ArffFileStream -f %s) -o %s" > %s' % (Paths.moaDirectory() + '/moa.jar',
                                                                                                                                                                                    Paths.moaDirectory() + '/sizeofag.jar',
                                                                                                                                                                                    classifierStr,
                                                                                                                                                                                    ARFFTrainFileName,
                                                                                                                                                                                    step,
                                                                                                                                                                                    ARFFTestFileName,
                                                                                                                                                                                    labelOutputFileName,
                                                                                                                                                                                    metaOutputFileName)]
            subprocess.call(cmd, shell=True)
            allPredictedTestLabels.append(np.loadtxt(labelOutputFileName, delimiter=',')[:, 0])

            #numClassifier = np.float(tmp[6])
            if classifierName in ['LVGB', 'HoeffAdwin']:
                tmp = np.loadtxt(metaOutputFileName, delimiter='=', dtype=np.str)[:, 1]
                numLeaves = float(tmp[12].replace(',', '')) * numClassifier
                numNodes = float(tmp[14].replace(',', '')) * numClassifier
                complexities.append(numNodes)
                complexityNumParameterMetric.append(getLVGBComplexityNumParameterMetric(numLeaves, numNodes, testFeatures.shape[1], len(np.unique(trainLabels))))
        if os.path.isfile(ARFFTestFileName):
            os.remove(ARFFTestFileName)
    else:
        cmd = ['java -Xmx35000m -cp %s -javaagent:%s moa.DoTask "EvaluatePrequential -l (%s) -s (ArffFileStream -f %s) -e BasicClassificationPerformanceEvaluator -f %d -o %s -d %s" > tmp.txt' % (Paths.moaDirectory() + '/moa.jar',
                                                                                                                                                                                Paths.moaDirectory() + '/sizeofag.jar',
                                                                                                                                                                                classifierStr,
                                                                                                                                                                                ARFFTrainFileName,
                                                                                                                                                                                evaluationStepSize,
                                                                                                                                                                                labelOutputFileName,
                                                                                                                                                                                metaOutputFileName)]

        subprocess.call(cmd, shell=True)
        predictedLabels = np.loadtxt(labelOutputFileName, delimiter=',')[:, 0]
        splitIndices = np.arange(evaluationStepSize, len(predictedLabels), evaluationStepSize)
        allPredictedTrainLabels = np.array_split(predictedLabels, splitIndices)
        if classifierName in ['LVGB', 'HoeffAdwin']:
            tmp = np.atleast_2d(np.loadtxt(metaOutputFileName, delimiter=',', dtype=np.str, skiprows=1))
            #numClassifier = tmp[:,9].astype(np.float)
            if classifierName == 'LVGB':
                metaFileNumNodesIdx = 15
                metaFileNumLeavesIdx = 17
            else:
                metaFileNumNodesIdx = 9
                metaFileNumLeavesIdx = 10
            numNodes = tmp[:,metaFileNumNodesIdx].astype(np.float) * numClassifier
            numLeaves = tmp[:,metaFileNumLeavesIdx].astype(np.float) * numClassifier
            numChangeDetections = tmp[:,10].astype(np.float)[-1]
            print numChangeDetections
            complexities += numNodes.tolist()
            complexityNumParameterMetric += (getLVGBComplexityNumParameterMetric(numLeaves, numNodes, testFeatures.shape[1], len(np.unique(trainLabels)))).tolist()


    if os.path.isfile(labelOutputFileName):
        os.remove(labelOutputFileName)
    if os.path.isfile(metaOutputFileName):
        os.remove(metaOutputFileName)
    if os.path.isfile(ARFFTrainFileName):
        os.remove(ARFFTrainFileName)

    return allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric

def generateARFFFile(features, labels, dstFileName):
    data = np.hstack([features, np.atleast_2d(labels).T])
    header = '@relation %s \n' % ('tmpDataset')
    for i in np.arange(features.shape[1]):
        header += '@attribute attribute%d Numeric \n' % (i)
    header += '@attribute class {%s}\n' % (','.join(str(p) for p in np.sort(np.unique(labels).astype(int))))
    header += '@data'
    np.savetxt(dstFileName, data, header=header, comments='', fmt='%.6g')
