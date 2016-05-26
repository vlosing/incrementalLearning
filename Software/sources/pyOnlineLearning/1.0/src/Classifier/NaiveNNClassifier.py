import scipy
import scipy.spatial

class NaiveNNClassifier:
    def __init__(self):
        self.prototypes = list()
        self.featureDim = 0
        self.labels = list()
        self.kdtree = None

    def testAndLearn(self, feature, label):
        testLabel = self.classify(feature)
        if testLabel == label:
            return True
        else:
            self.learn(feature, label)
            return False

    def learn(self, feature, label, initTree=True):
        assert len(feature.shape) == 1
        if len(self.prototypes) == 0:
            self.featureDim = feature.shape[0]
        else:
            assert feature.shape[0] == self.featureDim
        self.prototypes.append(feature)
        self.labels.append(label)
        if initTree:
            self.kdtree = scipy.spatial.KDTree(self.prototypes, leafsize=100000)

    def initTree(self):
        # print self.prototypes

        self.kdtree = scipy.spatial.KDTree(self.prototypes, leafsize=100000)

    def classify(self, feature):
        if len(self.prototypes) == 0:
            # print "unlearned classifier"
            return -1
        else:
            assert len(feature.shape) == 1
            assert feature.shape[0] == self.featureDim
            dist, labelIdx = self.kdtree.query(feature)
            # print str(dist) + ' ' + str(labelIdx)
            return self.labels[labelIdx]
