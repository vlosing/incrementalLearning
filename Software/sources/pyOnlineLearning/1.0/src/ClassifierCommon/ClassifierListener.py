__author__ = 'vlosing'
class ClassifierListener(object):
    def __init__(self):
        pass

    def onNewTrainStep(self, classifier, classificationResult, trainStep):
        raise NotImplementedError()

    def onNewPrototypes(self, classifier, protos, protoLabels):
        raise NotImplementedError()

    def onDelPrototypes(self, classifier, protos, protoLabels):
        raise NotImplementedError()

    def onNewWindowSize(self, classifier):
        raise NotImplementedError()


class DummyClassifierListener(ClassifierListener):
    def onNewTrainStep(self, classifier, classificationResult, trainStep):
        pass

    def onNewPrototypes(self, classifier, protos, protoLabels):
        pass

    def onDelPrototypes(self, classifier, protos, protoLabels):
        pass

    def onNewWindowSize(self, classifier):
        pass