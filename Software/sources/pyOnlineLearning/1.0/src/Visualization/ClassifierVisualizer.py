__author__ = 'vlosing'
from ClassifierCommon.ClassifierListener import ClassifierListener
import time
import matplotlib.pyplot as plt
import numpy as np
import GLVQPlot

class ClassifierVisualizer(ClassifierListener):
    SKIP_TRAINSTEPS = 2500
    DRAW_WINDOW_DATA = True
    DRAW_LTM_DATA = True

    DRAW_TRAINING_DATA = False
    DRAW_TEST_DATA = False

    DRAW_ON_TRAINSTEP = True
    DRAW_ON_NEW_PROTOTYPE = False
    DRAW_ON_DEL_PROTOTYPE = False
    DRAW_ON_NEW_WINDOWSIZE = False
    BORDER = 0
    def __init__(self, classifierPlotter, TrainSamples, TrainLabels, TestSamples, TestLabels, criteria):
        super(ClassifierVisualizer, self).__init__()
        self.TrainSamples = TrainSamples
        self.TrainLabels = TrainLabels
        self.TestSamples = TestSamples
        self.TestLabels = TestLabels
        self.criteria = criteria
        self.classifierPlotter = classifierPlotter

        self.minX = np.min(TrainSamples[:,0]) - ClassifierVisualizer.BORDER
        self.maxX = np.max(TrainSamples[:,0]) + ClassifierVisualizer.BORDER
        self.minY = np.min(TrainSamples[:,1]) - ClassifierVisualizer.BORDER
        self.maxY = np.max(TrainSamples[:,1]) + ClassifierVisualizer.BORDER

        plt.ion()
        if self.DRAW_WINDOW_DATA:
            self.figWindowData = plt.figure(figsize=(8, 8))
            self.subplotWindowData = self.figWindowData.add_subplot(111, aspect='equal')
        if self.DRAW_LTM_DATA:
            self.figLTMData = plt.figure(figsize=(8, 8))
            self.subplotLTMData = self.figLTMData.add_subplot(111, aspect='equal')

        if self.DRAW_TRAINING_DATA:
            self.figTrainingData = plt.figure(figsize=(8, 8))
            self.subplotTrainingData = self.figTrainingData.add_subplot(111, aspect='equal')
        if self.DRAW_TEST_DATA:
            self.figTestData = plt.figure(figsize=(8, 8))
            self.subplotTestData = self.figTestData.add_subplot(111, aspect='equal')

    def draw(self, classifier):
        if self.DRAW_WINDOW_DATA:
            self.subplotWindowData.clear()
            self.classifierPlotter.plot(classifier, classifier.windowSamples[:,:], classifier.windowSamplesLabels[:], self.figWindowData,  self.subplotWindowData,
                                        'STM', GLVQPlot.getDefColors(), XRange=[self.minX, self.maxX], YRange=[self.minY, self.maxY])
            self.figWindowData.canvas.draw()

        if self.DRAW_LTM_DATA:
            self.subplotLTMData.clear()
            self.classifierPlotter.plot(classifier, classifier.LTMSamples[:,:], classifier.LTMLabels[:], self.figLTMData,  self.subplotLTMData,
                                        'LTM', GLVQPlot.getDefColors(), XRange=[self.minX, self.maxX], YRange=[self.minY, self.maxY])
            self.figLTMData.canvas.draw()

        if self.DRAW_TRAINING_DATA:
            self.subplotTrainingData.clear()
            self.classifierPlotter.plot(classifier, self.TrainSamples[:classifier.trainStepCount,:], self.TrainLabels[:classifier.trainStepCount], self.figTrainingData,  self.subplotTrainingData,
                                        'Training-data', GLVQPlot.getDefColors(), XRange=[self.minX, self.maxX], YRange=[self.minY, self.maxY])
            self.figTrainingData.canvas.draw()
        if self.DRAW_TEST_DATA:
            self.subplotTestData.clear()
            self.classifierPlotter.plot(classifier, self.TestSamples, self.TestLabels, self.figTestData,  self.subplotTestData,
                                        'Test-data', GLVQPlot.getDefColors(), XRange=[self.minX, self.maxX], YRange=[self.minY, self.maxY])
            self.figTestData.canvas.draw()

    def onNewTrainStep(self, classifier, classificationResult, trainStep):
        if ClassifierVisualizer.DRAW_ON_TRAINSTEP and trainStep % (self.SKIP_TRAINSTEPS + 1) == 0:
            winLen = len(classifier.windowSamplesLabels)
            self.draw(classifier)
            #filename = '/homes/vlosing/STM%d.pdf' %(trainStep)
            #self.figWindowData.savefig(filename, bbox_inches='tight')
            #filename = '/homes/vlosing/LTM%d.pdf' %(trainStep)
            #self.figLTMData.savefig(filename, bbox_inches='tight')
            #time.sleep(0.5)

    def onNewPrototypes(self, classifier, protos, protoLabels):
        if ClassifierVisualizer.DRAW_ON_NEW_PROTOTYPE and np.atleast_2d(classifier.prototypes).shape[0] > 1:
            self.draw(classifier)

    def onNewWindowSize(self, classifier):
        if ClassifierVisualizer.DRAW_ON_NEW_WINDOWSIZE:
            self.draw(classifier)

    def onDelPrototypes(self, classifier, protos, protoLabels):
        if ClassifierVisualizer.DRAW_ON_DEL_PROTOTYPE:
            self.draw(classifier)
            GLVQPlot.highlightPrototypes(self.subplotWindowData, protos, protoLabels, GLVQPlot.getDefColors(), size=600)
            self.figWindowData.canvas.draw()
        #time.sleep(0.5)

