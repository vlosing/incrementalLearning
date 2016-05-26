__author__ = 'vlosing'

import numpy as np
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def getIndicesOrderedByLabel(labels):
    indices = np.array([])
    uniqueLabels = np.unique(labels)
    uniqueLabels = uniqueLabels[np.random.permutation(len(uniqueLabels))]
    for label in uniqueLabels:
        labelIndices = np.where(labels == label)[0]
        indices = np.append(indices, labelIndices)
    return indices


def getOrderedIndices(dataOrder, labels, chunkSize):
    indices = np.array([])
    if dataOrder == 'random':
        indices = np.random.permutation(len(labels))
    elif dataOrder == 'original':
        indices = np.arange(len(labels))
    elif dataOrder == 'orderedByLabel':
        #indices = np.argsort(labels)
        indices = getIndicesOrderedByLabel(labels)
    elif dataOrder == 'chunksRandom':
        '''self.checkForBalance(labels)
        indices = np.argsort(labels)
        numberOfChunks = len(indices) / self.chunkSize
        shuffledChunkIndices = np.random.permutation(numberOfChunks)
        indices = np.split(indices, numberOfChunks)
        indices = indices[shuffledChunkIndices]
        indices = np.concatenate(indices)'''
        uniqueLabels = np.unique(labels)
        splitIndices = []
        for label in uniqueLabels:
            labelIndices = np.where(labels == label)[0]
            numberOfChunks = len(labelIndices) / chunkSize

            splitIndices = splitIndices + np.array_split(labelIndices, numberOfChunks)
        splitIndices = np.array(splitIndices)
        splitIndices = splitIndices[np.random.permutation(len(splitIndices))]
        indices = np.concatenate(splitIndices)
    else:
        raise NameError('unknown data-order ' + dataOrder)
    indices = indices.astype(int)
    return indices



class TrainTestSplitsManager(object):
    def __init__(self, samples, labels, featureFileNames=[], dataOrder='random', chunkSize=1, shuffle=True, stratified=False, splitType='simple', numberOfFolds=2, trainSetSize=0.5, testSamples=None, testLabels=None):
        self.samples = samples
        self.labels = labels
        self.featureFileNames = featureFileNames
        self.TrainLabelsLst = []
        self.TrainSamplesLst = []
        self.TestLabelsLst = []
        self.TestSamplesLst = []
        self.TrainFileNames = []
        self.TestFileNames = []
        self.TrainShuffledIndices = []
        self.clearBeforeSplit = True
        self.dataOrder = dataOrder
        self.chunkSize = chunkSize
        self.shuffle = shuffle
        self.stratified = stratified
        self.splitType = splitType
        self.numberOfFolds = numberOfFolds
        self.trainSetSize = trainSetSize
        self.testSamples = testSamples
        self.testLabels = testLabels

    def clear(self):
        self.TrainLabelsLst = []
        self.TrainSamplesLst = []
        self.TestLabelsLst = []
        self.TestSamplesLst = []
        self.TrainFileNames = []
        self.TestFileNames = []
        self.TrainShuffledIndices = []

    def scaleData(self, maxSamples=None, maxProportion=None):
        if maxSamples and maxProportion:
            numScalingSamples = min(len(self.TrainLabelsLst[0]) * maxProportion, maxSamples)
        elif maxSamples:
            numScalingSamples = maxSamples
        elif maxProportion:
            numScalingSamples = len(self.TrainLabelsLst[0]) * maxProportion
        else:
            numScalingSamples = len(self.TrainLabelsLst[0])

        scaler = StandardScaler().fit(self.TrainSamplesLst[0][:numScalingSamples, :])
        #scaler = MinMaxScaler(feature_range=(-1,1)).fit(self.TrainSamplesLst[0][:numScalingSamples, :])

        self.TrainSamplesLst[0] = scaler.transform(self.TrainSamplesLst[0])

        if len(self.TestSamplesLst[0]) > 0:
            self.TestSamplesLst[0] = scaler.transform(self.TestSamplesLst[0])

    def generateSplits(self):
        if self.clearBeforeSplit: #XXVL kann vllt weg
            self.clear()
        samples = self.samples
        labels = self.labels
        if self.shuffle:
            shuffledIndices = np.random.permutation(len(labels))
            samples = samples[shuffledIndices]
            labels = labels[shuffledIndices]
        indices = getOrderedIndices(self.dataOrder, labels, self.chunkSize)
        samples = samples[indices]
        labels = labels[indices]
        if self.splitType == 'simple':
            dataSetSplitter = TrainTestSplitter(samples, labels, self.stratified, self.trainSetSize)
        elif self.splitType == 'kFold':
            dataSetSplitter = KFoldSplitter(samples, labels, self.stratified, self.numberOfFolds)
        elif self.splitType == None:
            dataSetSplitter = DummySplitter(samples, labels, self.testSamples, self.testLabels)
        else:
            raise NameError('unknown splitType ' + self.splitType)
        self.TrainSamplesLst, self.TrainLabelsLst, self.TestSamplesLst, self.TestLabelsLst = dataSetSplitter.getSplits()

    @staticmethod
    def getLabelMapping(labels):
        labelMapping = {}
        counter = 0
        for i in np.unique(labels):
            labelMapping[i] = counter
            counter += 1
        return labelMapping

    @staticmethod
    def getMappedLabels(labelMapping, Y):
        mappedLabels = np.empty(shape=(0, 1))
        for i in Y:
            mappedLabels = np.append(mappedLabels, labelMapping[i])
        return mappedLabels

    @staticmethod
    def exportDataToORF(XTrain, yTrain, XTest, yTest, fileNamePrefix):
        trainFeaturesFileName = fileNamePrefix + '-train.data'
        trainLabelsFileName = fileNamePrefix + '-train.labels'
        if np.issubdtype(yTrain.dtype, int) or np.issubdtype(yTrain.dtype, float):
            trainLabels = yTrain
            testLabels = yTest
        else:
            labelMapping = TrainTestSplitsManager.getLabelMapping(np.append(yTrain, yTest))
            trainLabels = TrainTestSplitsManager.getMappedLabels(labelMapping, yTrain)
            testLabels = TrainTestSplitsManager.getMappedLabels(labelMapping, yTest)


        if issubclass(XTrain.dtype.type, np.integer):
            #print 'save int'
            #df = pd.DataFrame(self.TrainSamplesLst[i])
            #df.to_csv(trainFeaturesFileName, header=False, sep=' ')
            #df = pd.DataFrame(trainLabels)
            #df.to_csv(trainLabelsFileName, header=False, sep=' ')

            np.savetxt(trainFeaturesFileName, XTrain, fmt='%i', header=str(len(XTrain)) + ' ' + str(XTrain.shape[1]), comments='')
            np.savetxt(trainLabelsFileName, trainLabels, fmt='%i', header=str(len(trainLabels)) + ' ' + str(1), comments='')
        else:
            np.savetxt(trainFeaturesFileName, XTrain, fmt='%.7f', header=str(len(XTrain)) + ' ' + str(XTrain.shape[1]), comments='')
            np.savetxt(trainLabelsFileName, trainLabels, fmt='%i', header=str(len(trainLabels)) + ' ' + str(1), comments='')

        if len(testLabels) > 0:
            testFeaturesFileName = fileNamePrefix + '-test.data'
            testLabelsFileName = fileNamePrefix + '-test.labels'
            if issubclass(XTest.dtype.type, np.integer):
                np.savetxt(testFeaturesFileName, XTest, fmt='%i',
                              header=str(len(XTest)) + ' ' + str(XTest.shape[1]), comments='')
                np.savetxt(testLabelsFileName, testLabels, fmt='%i',
                              header=str(len(testLabels)) + ' ' + str(1), comments='')
            else:
                np.savetxt(testFeaturesFileName, XTest, fmt='%.7f',
                              header=str(len(XTest)) + ' ' + str(XTest.shape[1]), comments='')
                np.savetxt(testLabelsFileName, testLabels, fmt='%i',
                              header=str(len(testLabels)) + ' ' + str(1), comments='')
            return trainFeaturesFileName, trainLabelsFileName, testFeaturesFileName, testLabelsFileName
        else:
            return trainFeaturesFileName, trainLabelsFileName, '', ''

    def exportToORFFormat(self, fileNamePrefix):
        print fileNamePrefix
        trainFeaturesFileNames = []
        trainLabelsFileNames = []
        testFeaturesFileNames = []
        testLabelsFileNames = []
        for i in range(len(self.TrainSamplesLst)):
            fileNamePrefix_ = fileNamePrefix + '_' + str(i)
            trainFeaturesFileName, trainLabelsFileName, testFeaturesFileName, testLabelsFileName =  TrainTestSplitsManager.exportDataToORF(self.TrainSamplesLst[i], self.TrainLabelsLst[i], self.TestSamplesLst[i], self.TestLabelsLst[i], fileNamePrefix_)
            trainFeaturesFileNames.append(trainFeaturesFileName)
            trainLabelsFileNames.append(trainLabelsFileName)
            if testFeaturesFileName != '':
                testFeaturesFileNames.append(testFeaturesFileName)
                testLabelsFileNames.append(testLabelsFileName)
        return trainFeaturesFileNames, trainLabelsFileNames, testFeaturesFileNames, testLabelsFileNames

class DataSetSplitter(object):
    def __init__(self, samples, labels, stratified):
        self.samples = samples
        self.labels = labels
        self.stratified = stratified
    def getSplits(self):
        raise NotImplementedError()


class DummySplitter(DataSetSplitter):
    def __init__(self, trainSamples, trainLabels, testSamples, testLabels):
        self.trainSamplesLst = []
        self.testSamplesLst = []
        self.trainLabelsLst = []
        self.testLabelsLst = []
        self.trainSamplesLst.append(trainSamples)
        self.trainLabelsLst.append(trainLabels)
        self.testSamplesLst.append(testSamples)
        self.testLabelsLst.append(testLabels)

    def getSplits(self):
        return self.trainSamplesLst, self.trainLabelsLst, self.testSamplesLst, self.testLabelsLst


class TrainTestSplitter(DataSetSplitter):
    def __init__(self, samples, labels, stratified, trainSetSize):
        super(TrainTestSplitter, self).__init__(samples, labels, stratified)
        self.trainSetSize = trainSetSize

    def getSplits(self):
        trainSamplesLst = []
        testSamplesLst = []
        trainLabelsLst = []
        testLabelsLst = []

        if self.stratified:
            numberOfSamplesPerClass = int(len(self.labels) * self.trainSetSize/ len(np.unique(self.labels)))
            trainIndices = []
            testIndices = []
            for className in np.unique(self.labels):
                indices = np.where(self.labels==className)[0]
                trainIndices = np.append(trainIndices, indices[0:numberOfSamplesPerClass])
                testIndices = np.append(testIndices, indices[numberOfSamplesPerClass:])
            trainIndices = trainIndices.astype(np.int)
            testIndices = testIndices.astype(np.int)
        else:
            indices = np.arange(len(self.labels))
            trainIndices = indices[0:int(len(indices) * self.trainSetSize)]
            testIndices = indices[len(trainIndices):len(indices)]
        trainSamplesLst.append(self.samples[trainIndices, :])
        testSamplesLst.append(self.samples[testIndices, :])
        trainLabelsLst.append(self.labels[trainIndices])
        testLabelsLst.append(self.labels[testIndices])
        return trainSamplesLst, trainLabelsLst, testSamplesLst, testLabelsLst


class KFoldSplitter(DataSetSplitter):
    def __init__(self, samples, labels, stratified, numberOfFolds):
        super(KFoldSplitter, self).__init__(samples, labels, stratified)
        self.numberOfFolds = numberOfFolds

    def getSplits(self):
        trainSamplesLst = []
        testSamplesLst = []
        trainLabelsLst = []
        testLabelsLst = []
        if self.stratified:
            kFold = cross_validation.StratifiedKFold(self.labels, n_folds=self.numberOfFolds, shuffle=False, random_state=None)
        else:
            kFold = cross_validation.KFold(len(self.labels), n_folds=self.numberOfFolds, shuffle=False, random_state=None)

        for train_index, test_index in kFold:
            trainSamplesLst.append(self.samples[train_index, :])
            testSamplesLst.append(self.samples[test_index, :])
            trainLabelsLst.append(self.labels[train_index])
            testLabelsLst.append(self.labels[test_index])
        return trainSamplesLst, trainLabelsLst, testSamplesLst, testLabelsLst


