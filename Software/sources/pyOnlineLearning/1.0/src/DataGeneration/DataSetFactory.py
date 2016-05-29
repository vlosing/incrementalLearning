from DataGeneration import realDataSets, toyDataSets
from DataSet import DataSet, TrainTestDataSet


def getDataSet(name, streams=[]):
    featureFileNames = []
    numObjectViews = None
    sequenceLength = None
    if name == 'COILAll':
        samples, labels, featureFileNames = realDataSets.getCoilAll()
        numObjectViews = 72
    elif name == 'COILAllRGB':
        samples, labels, featureFileNames = realDataSets.getCoilAllRGB()
        numObjectViews = 72
    elif name == 'OutdoorAll':
        samples, labels, featureFileNames = realDataSets.getOutdoorAllPaths()
        sequenceLength = 10
    elif name == 'OutdoorEasy':
        samples, labels, featureFileNames = realDataSets.getOutdoorEasy()
        sequenceLength = 10
    elif name == 'OutdoorEasyRGB':
        samples, labels, featureFileNames = realDataSets.getOutdoorEasyRGB()
        sequenceLength = 10
    elif name == 'demoAppData':
        samples, labels = realDataSets.getDemoAppData()
    elif name == 'demoPreTrain':
        samples, labels, featureFileNames = realDataSets.getDemoPreTrain()
    elif name == 'USPS':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getUSPS()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'DNA':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getDNA()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'letter':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getLetter()
        #scaler = StandardScaler().fit(trainSamples)
        #trainSamples = scaler.transform(trainSamples)
        #testSamples = scaler.transform(testSamples)
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'penDigits':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getPenDigits()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'gisette':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getGisette()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'isolet':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getIsolet()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'elec':
        samples, labels = realDataSets.getElec()
    elif name == 'weather':
        samples, labels = realDataSets.getWeather()
    elif name == 'souza2CDT':
        samples, labels = realDataSets.getSouza2CDT()
    elif name == 'souza4CREV1':
        samples, labels = realDataSets.getSouza4CREV1()
    elif name == 'souzaGears2C2D':
        samples, labels = realDataSets.getSouzaGEARS2C2D()
    elif name == 'souzaFG2C2D':
        samples, labels = realDataSets.getSouzaFG2C2D()
    elif name == 'souza2CHT':
        samples, labels = realDataSets.getSouza2CHT()
    elif name == 'rbfFast':
        samples, labels = realDataSets.getRBFFast()
    elif name == 'hyperplaneFast':
        samples, labels = realDataSets.getHyperplaneFast()
    elif name == 'hypSlow':
        samples, labels = realDataSets.getHyperplaneSlow()
    elif name == 'hypSlowXXL':
        samples, labels = realDataSets.getHyperplaneSlowXXL()
    elif name == 'rbfFast2D':
        samples, labels = realDataSets.getRBFFast2D()
    elif name == 'rbfSlow':
        samples, labels = realDataSets.getRBFSlow()
    elif name == 'rbfSlowXXL':
        samples, labels = realDataSets.getRBFSlowXXL()
    elif name == 'rbfSlow2D':
        samples, labels = realDataSets.getRBFSlow2D()
    elif name == 'rialto':
        samples, labels = realDataSets.getRialto()
    elif name == 'rbfIncr':
        samples, labels = toyDataSets.rbfIncr(5, 10, 1, 1000, 2)
    elif name == 'rbfAbruptXXL':
        #samples, labels = toyDataSets.rbfAbrupt2(10, 1, 200, 5, 2)
        samples, labels = realDataSets.getRBFAbruptXXL()
    elif name == 'rbfAbruptSmall':
        samples, labels = realDataSets.getRBFAbruptSmall()
    #elif name == 'rbfGradual':
    #    samples, labels = toyDataSets.rbfGradual(2, 1, 200, 3, 2)
    elif name == 'squaresIncr':
        samples, labels = toyDataSets.squaresIncr(4, 7000, [0, 1], 2, distBetween=0.5, velocity = 0.02)
    elif name == 'squaresIncrXXL':
        #samples, labels = toyDataSets.squaresIncr(4, 250000, [0, 1], 2, distBetween=0.5, velocity = 0.02)
        samples, labels = realDataSets.getSquaresIncrXXL()
    #elif name == 'rectGradual':
    #    samples, labels = toyDataSets.rectGradual(2, 200, 4, distBetween=0.1)
    elif name == 'chessVirtualXXL':
        #samples, labels = toyDataSets.getChessRandomVirtual(250, 1000, nTiles=8, repetitions=32)
        samples, labels = realDataSets.getChessVirtualXXL()
    elif name == 'chessVirtual':
        #samples, labels = toyDataSets.getChessRandomVirtual(300, 1000, nTiles=8)
        #samples, labels = toyDataSets.getChessRandomVirtual(200, 500, nTiles=8)
        samples, labels = realDataSets.getChessVirtual()

    elif name == 'chessIIDXXL':
        #samples, labels = toyDataSets.getChessIID(3125)
        samples, labels = realDataSets.getChessIIDXXL()
        #samples, labels = toyDataSets.getChessIID2(200000)

    elif name == 'chessFields':
        #samples, labels = toyDataSets.getChessRandomFieldOrder(200, repetitions=5)
        samples, labels = realDataSets.getChessFields()
    elif name == 'allDrift':
        samples, labels = realDataSets.getAllDrift()
    elif name == 'allDriftXXL':
        '''samples1, labels1 = realDataSets.getRBFAbruptXXL()
        samples2, labels2 = realDataSets.getChessVirtualXXL()
        samples2[:,0] += 1.25
        samples3, labels3 = realDataSets.getSquaresIncrXXL()
        samples3[:,0] += 2.5
        samples4, labels4 = realDataSets.getChessIIDXXL()
        samples4[:,0] += 3.75
        samples, labels = toyDataSets.getMixedDataset([samples1, samples2, samples3, samples4], [labels1, labels2, labels3, labels4])'''
        samples, labels = realDataSets.getAllDriftXXL()
    elif name == 'cbConst':
        samples, labels = realDataSets.getCBConstant()
    elif name == 'cbSinus':
        samples, labels = realDataSets.getCBSinus()
    elif name == 'keystroke':
        samples, labels = realDataSets.getKeystroke()
    elif name == 'powerSupply':
        samples, labels = realDataSets.getPowerSupply()
    elif name == 'covType':
        samples, labels = realDataSets.getCovType()
    elif name == 'sea':
        samples, labels, dummy1, dummy2 = realDataSets.getSea()
    elif name == 'news20':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getNews20()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'mnist':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getMNist()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'outdoor':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getOutdoor()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'outdoorStream':
        samples, labels = realDataSets.getOutdoorStream()
    elif name == 'coil':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getCOIL()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'border':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getBorder()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'overlap':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getOverlap()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'noise':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getNoise()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'satImage':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getSatImage()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'poker':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getPoker()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'HAR':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getHAR()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'optDigits':
        trainSamples, trainLabels, testSamples, testLabels = realDataSets.getOptDigits()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'spam':
        samples, labels = realDataSets.getSpam()
    elif name == 'borderGen':
        samples, labels = toyDataSets.getBorderSet(0, 0, 5.8, 10000, 10000, 10000, 0.0)
    elif name == 'Toy2':
        samples, labels = toyDataSets.getToySet2(0, 0, 150, 18, 500, (0.5, 5), (3.5, 3.5), 10, 0)
    elif name == 'overlapGen':
        samples, labels = toyDataSets.getOverlapSet(-19, 5, 8, 1, 10, 500, 1500, 0)
    elif name == 'noiseGen':
        samples, labels = toyDataSets.getNoiseSet(-15, -15, 30, 2500, 2500, 0)
    elif name == 'Chess':
        samples, labels = toyDataSets.getChess(-18, 14, 4, 50)
    elif name == 'Chess2':
        samples, labels = toyDataSets.getChess(-10, 6, 10, 100, rows=2, cols=2)
    elif name == 'Gaussian':
        samples, labels = toyDataSets.getManyClasses(6, 1, 1000)
    elif name == 'CutInTest':
        samples, labels = realDataSets.getCutInTest()
    else:
        samples, labels = realDataSets.getCutInDataSet(name, streams)

    '''
    else:
        raise Exception('unknown dataset ' + name)'''

    return DataSet(name, samples, labels, featureFileNames, numObjectViews, sequenceLength)


def isStationary(dataSetName):
    if dataSetName in ['border', 'overlap', 'coil', 'USPS', 'DNA',
                           'letter', 'isolet', 'gisette',
                           'mnist', 'outdoor', 'satImage',
                           'penDigits', 'HAR', 'optDigits', 'news20', 'noise']:
        return True
    elif dataSetName in ['weather', 'elec', 'spam', 'souza2CDT', 'souza4CREV1', 'souzaGears2C2D', 'souzaFG2C2D', 'souza2CHT', 'keystroke', 'covType', 'rbfFast',
                         'rbfSlow', 'rbfSlowXXL', 'sea', 'rbfSlow2D', 'rbfFast2D', 'hypSlow', 'hypSlowXXL', 'cbConst', 'cbSinus',
                         'rbfIncr', 'rbfAbruptSmall', 'rbfAbruptXXL', 'squaresIncr', 'squaresIncrXXL', 'rbfGradual', 'rectGradual',
                         'chessVirtual', 'chessFields', 'chessVirtualXXL', 'chessIIDXXL', 'outdoorStream', 'rialto', 'allDrift', 'allDriftXXL']:
        return False

    else:
        raise Exception('unknown dataset')