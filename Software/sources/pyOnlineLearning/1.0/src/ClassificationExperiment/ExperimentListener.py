__author__ = 'vlosing'
from ClassifierCommon.ClassifierListener import ClassifierListener


class ExpListener(ClassifierListener):
    def __init__(self):
        super(ExpListener, self).__init__()
        pass

    def onNewCfgIteration(self, cfgName):
        raise NotImplementedError()

    def onTrainDataChange(self, trainSamples, testSamples, trainLabels, testLabels):
        raise NotImplementedError()

    def onNewClassifier(self, classifier):
        raise NotImplementedError()

    def onFinishIteration(self):
        raise NotImplementedError()

    def onFinishCfgIteration(self, cfgName, iterationIdx):
        raise NotImplementedError()

    def onFinishFoldCfgIteration(self, foldIdx):
        raise NotImplementedError()

    def onFinishExperiment(self, classifiers, expPrefix):
        raise NotImplementedError()


class DummyExpListener(ExpListener):
    def __init__(self):
        super(DummyExpListener, self).__init__()
        pass

    def onNewCfgIteration(self, cfgName):
        pass

    def onTrainDataChange(self, trainSamples, testSamples, trainLabels, testLabels):
        pass

    def onNewClassifier(self, classifier):
        pass

    def onNewTrainStep(self, classifier, classificationResult, trainStep):
        pass

    def onFinishIteration(self):
        pass

    def onFinishCfgIteration(self, cfgName, iterationIdx):
        pass

    def onFinishFoldCfgIteration(self, foldIdx):
        pass

    def onFinishExperiment(self, classifiers, expPrefix):
        pass

    def onNewPrototypes(self, classifier, protos, protoLabels):
        pass

    def onDelPrototypes(self, classifier, protos, protoLabels):
        pass
    def onNewWindowSize(self, classifier):
        pass

