import numpy as np
import logging

class DataSet(object):
    def __init__(self, name, samples, labels, featureFileNames=None, numObjectViews=None, sequenceLength=None, imagesDir=None, missClassifiedDir=None, missClassifiedStatisticsFileName=None):
        self.name = name
        self.samples = samples
        self.dimensions = samples.shape[1]
        self.labels = labels

        self.featureFileNames = featureFileNames
        self.numObjectViews = numObjectViews
        self.sequenceLength = sequenceLength
        self.imagesDir = imagesDir
        self.missClassifiedDir = missClassifiedDir
        self.missClassifiedStatisticsFileName = missClassifiedStatisticsFileName
        self.testSamples = None
        self.testLabels = None


    def getLabelDistributions(self):
        distribution = []
        uniqueLabels = np.unique(self.labels)
        for label in uniqueLabels:
            numberOfSamples = len(np.where(self.labels == label)[0])
            distribution.append([label, numberOfSamples])
        return distribution

    def printDSInformation(self):
        logging.info('name %s' %(self.name))
        logging.info('dimensions %s' %(self.dimensions))
        logging.info('samples %d' %(len(self.labels)))
        logging.info('classes %d' %(len(np.unique(self.labels))))
        logging.info('distribution %s' %(self.getLabelDistributions()))



class TrainTestDataSet(DataSet):
    def __init__(self, name, trainSamples, trainLabels, testSamples, testLabels):
        super(TrainTestDataSet, self).__init__(name, trainSamples, trainLabels)
        self.testSamples = testSamples
        self.testLabels = testLabels

