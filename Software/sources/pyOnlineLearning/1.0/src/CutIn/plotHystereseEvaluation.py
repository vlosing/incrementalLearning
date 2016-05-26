import matplotlib.pyplot as plt
import CutInCommon
from Base.Serialization import loadJsonFile
from Base import Paths
from Visualization import GLVQPlot
import numpy as np

def plotHystereseData(expName):
    data = np.array(loadJsonFile(Paths.HystereseCutInResultsDir() + expName + '.json'))
    ROCValues = np.hstack((np.atleast_2d(data[:, 1]).T, np.atleast_2d(data[:, 2] / (data[:, 2] + data[:, 3])).T))
    cIdx = 0
    label = expName
    X = []
    Y = []
    cIdx += 1
    for rocValue in ROCValues:
        X.append(rocValue[0])
        Y.append(rocValue[1])
    plt.plot(X, Y, label=label, color=GLVQPlot.getDefColors()[cIdx], linestyle='-')
    plt.ylim([0, 1])
    plt.xlim([0, 50])
    plt.legend(loc=0)

def getExpName(trainStreams, testStreams, trainMaxTTCPredPos, trainMaxTTCPredNeg, evalMaxTTCPred, trainHorizon, vehicleType, negSamplesRatio):
    trainSetName = CutInCommon.getTrainSetName(vehicleType, trainStreams, trainHorizon, trainMaxTTCPredPos, trainMaxTTCPredNeg, negSamplesRatio)
    return trainSetName + '_test%03d-%03d_%d' % (testStreams[0], testStreams[-1], evalMaxTTCPred)

def plotExperiment(trainStreams, testStreams, trainMaxTTCPredPos, trainMaxTTCPredNeg, evalMaxTTCPred, trainHorizon, vehicleType, negSamplesRatio):
    expName = getExpName(trainStreams, testStreams, trainMaxTTCPredPos, trainMaxTTCPredNeg, evalMaxTTCPred, trainHorizon, vehicleType, negSamplesRatio)
    plotHystereseData(expName)

if __name__ == '__main__':
    trainStreams = np.arange(1, 21, 1)
    testStreams = np.arange(50, 90, 1)
    vehicleType = '2X_EU'
    minTTCPred = 0.1
    plotExperiment(trainStreams, testStreams, 10000, 10000, 10000, 64000, vehicleType, 1)
    plt.show()