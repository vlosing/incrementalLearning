import json

import numpy as np

from Base import Paths
from Base import Serialization
from CutIn import CutInCommon
import pandas as pd

def getCoilAll():
    imagePath, dummy, featurePath, labelsPath, featureFileNamesPath = Paths.getCoilAllPaths()
    f = open(featurePath, 'r')
    AllFeatures = np.array(json.load(f))
    f.close()
    f = open(labelsPath, 'r')
    AllLabels = np.array(json.load(f), dtype=np.unicode)
    f.close()
    f = open(featureFileNamesPath, 'r')
    featureFileNames = np.array(json.load(f))
    f.close()
    return AllFeatures, AllLabels, featureFileNames


def getCoilAllRGB():
    imagePath, dummy, featurePath, labelsPath, featureFileNamesPath = Paths.getCoilAllRGBPaths()
    f = open(featurePath, 'r')
    AllFeatures = np.array(json.load(f))
    f.close()
    f = open(labelsPath, 'r')
    AllLabels = np.array(json.load(f), dtype=np.unicode)
    f.close()
    f = open(featureFileNamesPath, 'r')
    featureFileNames = np.array(json.load(f))
    f.close()
    return AllFeatures, AllLabels, featureFileNames


def getOutdoorAll():
    dummy, featurePath, labelsPath, featureFileNamesPath = Paths.getOutdoorAllPaths()
    f = open(featurePath, 'r')
    AllFeatures = np.array(json.load(f))
    f.close()
    f = open(labelsPath, 'r')
    AllLabels = np.array(json.load(f))
    f.close()
    f = open(featureFileNamesPath, 'r')
    featureFileNames = np.array(json.load(f))
    f.close()
    return AllFeatures, AllLabels, featureFileNames

def getOutdoorEasy():
    dummy, featurePath, labelsPath, featureFileNamesPath = Paths.getOutdoorEasyPaths()
    f = open(featurePath, 'r')
    AllFeatures = np.array(json.load(f))
    f.close()
    f = open(labelsPath, 'r')
    AllLabels = np.array(json.load(f))
    f.close()
    f = open(featureFileNamesPath, 'r')
    featureFileNames = np.array(json.load(f))
    f.close()
    return AllFeatures, AllLabels, featureFileNames


def getOutdoorEasyRGB():
    dummy, featurePath, labelsPath, featureFileNamesPath = Paths.getOutdoorEasyPathsRGB()
    f = open(featurePath, 'r')
    AllFeatures = np.array(json.load(f))
    f.close()
    f = open(labelsPath, 'r')
    AllLabels = np.array(json.load(f))
    f.close()
    f = open(featureFileNamesPath, 'r')
    featureFileNames = np.array(json.load(f))
    f.close()
    return AllFeatures, AllLabels, featureFileNames


def getDemoAppData():
    AllFeatures, AllLabels = Serialization.getAllFeaturesAndLabels(Paths.ALMDemoTrainSetDataDir() + 'images/',
                                                                   Paths.ALMDemoTrainSetMappingDir())
    return AllFeatures, AllLabels


def getDemoPreTrain():
    imagePath, featurePath, labelsPath, featureFileNamesPath = Paths.getDemoPreTrainPaths()
    f = open(featurePath, 'r')
    AllFeatures = np.array(json.load(f))
    f.close()

    f = open(labelsPath, 'r')
    AllLabels = np.array(json.load(f))
    f.close()

    f = open(featureFileNamesPath, 'r')
    featureFileNames = np.array(json.load(f))
    f.close()

    return AllFeatures, AllLabels, featureFileNames


def getUSPS():
    trainSamples = np.loadtxt(Paths.USPSTrainFeaturesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.USPSTrainLabelsPath(), skiprows=1, dtype=np.int8)

    testSamples = np.loadtxt(Paths.USPSTestFeaturesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.USPSTestLabelsPath(), skiprows=1, dtype=np.int8)

    '''trainSamples=pd.read_csv(Paths.USPSTrainFeaturesPath(), sep=' ').values
    trainLabels=pd.read_csv(Paths.USPSTrainLabelsPath(), sep=' ').values
    testSamples=pd.read_csv(Paths.USPSTestFeaturesPath(), sep=' ').values
    testLabels=pd.read_csv(Paths.USPSTestLabelsPath(), sep=' ').values'''

    return trainSamples, trainLabels, testSamples, testLabels


def getDNA():
    '''trainSamples = np.loadtxt(Paths.DNATrainFeaturesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.DNATrainLabelsPath(), skiprows=1, dtype=np.int8)

    testSamples = np.loadtxt(Paths.DNATestFeaturesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.DNATestLabelsPath(), skiprows=1, dtype=np.int8)'''

    trainSamples=pd.read_csv(Paths.DNATrainFeaturesPath(), sep=' ', header=None).values
    trainLabels=pd.read_csv(Paths.DNATrainLabelsPath(), sep=' ', header=None, dtype=np.int8).values.ravel()
    testSamples=pd.read_csv(Paths.DNATestFeaturesPath(), sep=' ', header=None).values
    testLabels=pd.read_csv(Paths.DNATestLabelsPath(), sep=' ', header=None, dtype=np.int8).values.ravel()
    trainLabels = trainLabels.ravel()
    return trainSamples, trainLabels, testSamples, testLabels

def getCutInTest():
    f = open(Paths.CutInTestFeaturesPath(), 'r')
    AllFeatures = np.array(json.load(f))
    f.close()
    f = open(Paths.CutInTestLabelsPath(), 'r')
    AllLabels = np.array(json.load(f))
    f.close()
    return AllFeatures, AllLabels


def getCutInDataSet(trainSetName, streams):
    vehicleType, dummy, seqLength, maxTTCPredPos, maxTTCPredNeg, negSamplesRatio = CutInCommon.getProperties2TrainSetName(trainSetName)
    AllFeatures = []
    AllLabels = []
    for stream in streams:
        trainSetStreamName = CutInCommon.getTrainSetStreamName(vehicleType, stream, seqLength, maxTTCPredPos, maxTTCPredNeg, negSamplesRatio)
        prefix = Paths.CutInFeaturesDir() + vehicleType + '/' + trainSetStreamName
        featurePath = prefix + '_features.json'
        labelsPath = prefix + '_labels.json'

        f = open(featurePath, 'r')
        AllFeatures = AllFeatures + json.load(f)
        f.close()

        f = open(labelsPath, 'r')
        AllLabels = AllLabels + json.load(f)
        f.close()
    AllFeatures = np.array(AllFeatures)
    AllLabels = np.array(AllLabels)
    return AllFeatures, AllLabels


def getLetter():
    trainSamples = np.loadtxt(Paths.LetterTrainFeaturesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.LetterTrainLabelsPath(), skiprows=1, dtype=np.int8)

    testSamples = np.loadtxt(Paths.LetterTestFeaturesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.LetterTestLabelsPath(), skiprows=1, dtype=np.int8)

    '''trainSamples=pd.read_csv(Paths.LetterTrainFeaturesPath(), sep=' ').values
    trainLabels=pd.read_csv(Paths.LetterTrainLabelsPath(), sep=' ').values
    testSamples=pd.read_csv(Paths.LetterTestFeaturesPath(), sep=' ').values
    testLabels=pd.read_csv(Paths.LetterTestLabelsPath(), sep=' ').values'''

    return trainSamples, trainLabels, testSamples, testLabels


def getPenDigits():
    trainSamples = np.loadtxt(Paths.PenDigitsTrainFeaturesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.PenDigitsTrainLabelsPath(), skiprows=1, dtype=np.int8)

    testSamples = np.loadtxt(Paths.PenDigitsTestFeaturesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.PenDigitsTestLabelsPath(), skiprows=1, dtype=np.int8)

    '''trainSamples=pd.read_csv(Paths.PenDigitsTrainFeaturesPath(), sep=' ').values
    trainLabels=pd.read_csv(Paths.PenDigitsTrainLabelsPath(), sep=' ', dtype=np.int8).values.ravel()
    testSamples=pd.read_csv(Paths.PenDigitsTestFeaturesPath(), sep=' ', dtype=np.int8).values
    testLabels=pd.read_csv(Paths.PenDigitsTestLabelsPath(), sep=' ').values.ravel()'''

    return trainSamples, trainLabels, testSamples, testLabels


def getGisette():
    '''trainSamples = np.loadtxt(Paths.GisetteTrainFeaturesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.GisetteTrainLabelsPath(), skiprows=0, dtype=np.uint8)

    testSamples = np.loadtxt(Paths.GisetteTestFeaturesPath(), skiprows=0)
    testLabels = np.loadtxt(Paths.GisetteTestLabelsPath(), skiprows=0, dtype=np.uint8)'''

    trainSamples=pd.read_csv(Paths.GisetteTrainFeaturesPath(), header=None, sep=' ').values
    trainLabels=pd.read_csv(Paths.GisetteTrainLabelsPath(), header=None, sep=' ', dtype=np.int8).values.ravel()
    testSamples=pd.read_csv(Paths.GisetteTestFeaturesPath(), header=None, sep=' ').values
    testLabels=pd.read_csv(Paths.GisetteTestLabelsPath(), header=None, sep=' ', dtype=np.int8).values.ravel()
    trainLabels = trainLabels.ravel()

    return trainSamples, trainLabels, testSamples, testLabels

def getIsolet():
    trainSamples = np.loadtxt(Paths.isoletTrainFeaturesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.isoletTrainLabelsPath(), skiprows=0, dtype=np.uint8).ravel()
    testSamples = np.loadtxt(Paths.isoletTestFeaturesPath(), skiprows=0)
    testLabels = np.loadtxt(Paths.isoletTestLabelsPath(), skiprows=0, dtype=np.uint8).ravel()
    '''trainSamples=pd.read_csv(Paths.isoletTrainFeaturesPath(), header=None, sep=' ').values
    trainLabels=pd.read_csv(Paths.isoletTrainLabelsPath(), header=None, sep=' ').values
    testSamples=pd.read_csv(Paths.isoletTestFeaturesPath(), header=None, sep=' ').values
    testLabels=pd.read_csv(Paths.isoletTestLabelsPath(), header=None, sep=' ').values'''
    return trainSamples, trainLabels, testSamples, testLabels

def getElec():
    trainSamples=pd.read_csv(Paths.elecTrainFeaturesPath(), sep=',', header=None).values[:, [0,1,3,4,5,6]]
    trainLabels=pd.read_csv(Paths.elecTrainLabelsPath(), sep=',', header=None, dtype=np.int8).values.ravel()


    '''trainSamples = np.loadtxt(Paths.elecTrainFeaturesPath(), skiprows=0, delimiter=',')[:, [0,1,3,4,5,6]]
    trainLabels = np.loadtxt(Paths.elecTrainLabelsPath(), skiprows=0, dtype=np.uint8)'''
    return trainSamples, trainLabels

def getSea():
    trainSamples=pd.read_csv(Paths.seaTrainFeaturesPath(), sep=',', header=None).values
    trainLabels=pd.read_csv(Paths.seaTrainLabelsPath(), sep=',', header=None, dtype=np.int8).values.ravel()
    testSamples=pd.read_csv(Paths.seaTestFeaturesPath(), sep=',', header=None).values
    testLabels=pd.read_csv(Paths.seaTestLabelsPath(), sep=',', header=None, dtype=np.int8).values.ravel()

    return trainSamples, trainLabels, testSamples, testLabels

def getMNist():
    trainSamples=pd.read_csv(Paths.MNistTrainSamplesPath(), header=None, sep=' ').values[:, :]
    trainLabels=pd.read_csv(Paths.MNistTrainLabelsPath(), header=None, sep=' ', dtype=np.int8).values.ravel()[:]
    testSamples=pd.read_csv(Paths.MNistTestSamplesPath(), header=None, sep=' ').values
    testLabels=pd.read_csv(Paths.MNistTestLabelsPath(), header=None, sep=' ', dtype=np.int8).values.ravel()
    return trainSamples, trainLabels, testSamples, testLabels

def getWeather():
    trainSamples=pd.read_csv(Paths.weatherTrainFeaturesPath(), sep=',', header=None).values
    trainLabels=pd.read_csv(Paths.weatherTrainLabelsPath(), sep=',', header=None, dtype=np.int8).values.ravel()
    #trainSamples=pd.read_csv(Paths.weatherTrainFeaturesPath(), sep=',', header=None).values[0:6000, :]
    #trainLabels=pd.read_csv(Paths.weatherTrainLabelsPath(), sep=',', header=None, dtype=np.int8).values.ravel()[0:6000]

    #permIndices = np.random.permutation(len(trainLabels))
    #trainSamples = trainSamples[permIndices, :]
    #trainLabels = trainLabels[permIndices]

    '''trainSamples = np.loadtxt(Paths.weatherTrainFeaturesPath(), skiprows=0, delimiter=',')
    trainLabels = np.loadtxt(Paths.weatherTrainLabelsPath(), skiprows=0, dtype=np.uint8)'''
    return trainSamples, trainLabels

def getSpam():
    trainSamples=pd.read_csv(Paths.spamTrainFeaturesPath(), header=None, sep=' ').values.astype(np.int8)
    trainLabels=pd.read_csv(Paths.spamTrainLabelsPath(), header=None, dtype=np.int8).values.ravel()
    return trainSamples, trainLabels

def getOutdoor():
    trainSamples = np.loadtxt(Paths.outdoorTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.outdoorTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.outdoorTestSamplesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.outdoorTestLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels, testSamples, testLabels

def getOutdoorStream():
    samples = np.loadtxt(Paths.outdoorStreamSamplesPath(), skiprows=0)
    labels = np.loadtxt(Paths.outdoorStreamLabelsPath(), skiprows=0, dtype=np.uint8)
    return samples, labels

def getCOIL():
    trainSamples = np.loadtxt(Paths.coilTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.coilTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.coilTestSamplesPath(), skiprows=0)
    testLabels = np.loadtxt(Paths.coilTestLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels, testSamples, testLabels


def getNews20():
    from sklearn.datasets import load_svmlight_file
    X_train, y_train = load_svmlight_file(Paths.news20TrainPath())
    X_train = X_train.toarray().astype(np.int16)[:,:]
    y_train = (y_train -1).astype(np.int16)[:]

    X_test, y_test = load_svmlight_file(Paths.news20TestPath())
    X_test = X_test.toarray().astype(np.int16)
    X_test = np.hstack([X_test, np.zeros((X_test.shape[0], 1))])
    y_test = (y_test -1).astype(np.int16)

    return X_train, y_train, X_test, y_test

def getBorder():
    trainSamples = np.loadtxt(Paths.borderTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.borderTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.borderTestSamplesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.borderTestLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels, testSamples, testLabels

def getOverlap():
    trainSamples = np.loadtxt(Paths.overlapTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.overlapTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.overlapTestSamplesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.overlapTestLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels, testSamples, testLabels

def getNoise():
    trainSamples = np.loadtxt(Paths.noiseTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.noiseTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.noiseTestSamplesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.noiseTestLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels, testSamples, testLabels

def getSatImage():
    trainSamples = np.loadtxt(Paths.satImageTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.satImageTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.satImageTestSamplesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.satImageTestLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels, testSamples, testLabels

def getHAR():
    trainSamples = np.loadtxt(Paths.HARTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.HARTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.HARTestSamplesPath(), skiprows=0)
    testLabels = np.loadtxt(Paths.HARTestLabelsPath(), skiprows=0, dtype=np.uint8)

    return trainSamples, trainLabels, testSamples, testLabels

def getOptDigits():
    trainSamples = np.loadtxt(Paths.optDigitsTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.optDigitsTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.optDigitsTestSamplesPath(), skiprows=0)
    testLabels = np.loadtxt(Paths.optDigitsTestLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels, testSamples, testLabels

def getSouza2CDT():
    trainSamples = np.loadtxt(Paths.souza2CDTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.souza2CDTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getSouza4CREV1():
    trainSamples = np.loadtxt(Paths.souza4CREV1TrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.souza4CREV1TrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getSouzaGEARS2C2D():
    trainSamples = np.loadtxt(Paths.souzaGEARS2C2DTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.souzaGEARS2C2DTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getSouzaFG2C2D():
    trainSamples = np.loadtxt(Paths.souzaFG2C2DTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.souzaFG2C2DTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getSouza2CHT():
    trainSamples = np.loadtxt(Paths.souza2CHTTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.souza2CHTTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getKeystroke():
    trainSamples = np.loadtxt(Paths.keystrokeTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.keystrokeTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getCovType():
    trainSamples = np.loadtxt(Paths.covTypeTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.covTypeTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getRBFFast():
    trainSamples = np.loadtxt(Paths.rbfFastTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.rbfFastTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getHyperplaneFast():
    trainSamples = np.loadtxt(Paths.hyperplaneFastTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.hyperplaneFastTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getHyperplaneSlow():
    trainSamples = np.loadtxt(Paths.hyperplaneSlowTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.hyperplaneSlowTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getHyperplaneSlowXXL():
    trainSamples = np.loadtxt(Paths.hyperplaneSlowLargeTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.hyperplaneSlowLargeTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getRBFFast2D():
    trainSamples = np.loadtxt(Paths.rbfFast2DTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.rbfFast2DTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getRBFSlow():
    trainSamples = np.loadtxt(Paths.rbfSlowTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.rbfSlowTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getRBFSlowXXL():
    trainSamples = np.loadtxt(Paths.rbfSlowLargeTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.rbfSlowLargeTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getRBFSlow2D():
    trainSamples = np.loadtxt(Paths.rbfSlow2DTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.rbfSlow2DTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getCBConstant():
    trainSamples = np.loadtxt(Paths.cbConstTrainSamplesPath(), skiprows=0, delimiter=',')
    trainLabels = np.loadtxt(Paths.cbConstTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getCBSinus():
    trainSamples = np.loadtxt(Paths.cbSinusTrainSamplesPath(), skiprows=0, delimiter=',')
    trainLabels = np.loadtxt(Paths.cbSinusTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getRBFAbruptXXL():
    trainSamples = np.loadtxt(Paths.rbfAbruptXXLTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.rbfAbruptXXLTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels

def getRBFAbruptSmall():
    trainSamples = np.loadtxt(Paths.rbfAbruptSmallTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.rbfAbruptSmallTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels

def getRialto():
    trainSamples = np.loadtxt(Paths.rialtoTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.rialtoTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels

def getSquaresIncr():
    trainSamples = np.loadtxt(Paths.squaresIncrTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.squaresIncrTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels

def getSquaresIncrXXL():
    trainSamples = np.loadtxt(Paths.squaresIncrXXLTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.squaresIncrXXLTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels

def getChessVirtual():
    trainSamples = np.loadtxt(Paths.chessVirtualTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.chessVirtualTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels

def getChessVirtualXXL():
    trainSamples = np.loadtxt(Paths.chessVirtualXXLTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.chessVirtualXXLTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels

def getAllDrift():
    trainSamples = np.loadtxt(Paths.allDriftTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.allDriftTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels

def getAllDriftXXL():
    trainSamples = np.loadtxt(Paths.allDriftXXLTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.allDriftXXLTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels
