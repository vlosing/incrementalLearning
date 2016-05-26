import numpy as np
from LVQCommon import LVQCommon
import logging

class DistanceMatrix(object):
    @staticmethod
    def addProtoToDistanceMatrix(distMatrix, samples, newPrototype, newPrototypeLabel, newPrototypeIdx,
                                 prototypeLabels, activFct, logisticFactor, getDistanceFct, protoStatistics=None, checkOverlapping=False):
        newDistMatrix = np.copy(distMatrix)
        if len(newDistMatrix) > 0:
            tmpProto = newPrototype.copy()
            tmpProto.shape = [len(newPrototype), 1]
            protoMat = tmpProto * np.ones(shape=[1, samples.shape[0]])
            samplesInt = np.transpose(samples)
            distances = getDistanceFct(samplesInt, protoMat)
            newProtoLabels = np.atleast_2d([newPrototypeLabel] * samplesInt.shape[0]).T
            winnerIndices = np.unique(np.where((distances < newDistMatrix[:, 1]) &
                                               (newProtoLabels == prototypeLabels[newDistMatrix[:, 0].astype(int)]))[1])
            looserIndices = np.unique(np.where((distances < newDistMatrix[:, 3]) &
                                               (newProtoLabels != prototypeLabels[newDistMatrix[:, 0].astype(int)]))[1])

            if protoStatistics is not None:
                indicesOldWinner = np.where((newDistMatrix[winnerIndices, 1] < newDistMatrix[winnerIndices, 3]))[0]
                protoIndices, counts = np.unique(newDistMatrix[winnerIndices[indicesOldWinner], 0], return_counts=True)
                protoIndices = protoIndices.astype(int)
                protoStatistics[protoIndices, 0] -= counts
                protoStatistics[protoIndices, 1] -= counts

                indicesOldLooser = np.where((newDistMatrix[looserIndices, 1] >= newDistMatrix[looserIndices, 3]))[0]
                protoIndices, counts = np.unique(newDistMatrix[looserIndices[indicesOldLooser], 2], return_counts=True)
                protoIndices = protoIndices.astype(int)
                protoStatistics[protoIndices, 0] -= counts
                protoStatistics[protoIndices, 2] -= counts

            newDistMatrix[winnerIndices, 0] = newPrototypeIdx
            newDistMatrix[winnerIndices, 1] = distances[winnerIndices]
            if checkOverlapping and not all(distances[winnerIndices] + newDistMatrix[winnerIndices, 3]):
                logging.warning('inserted 2. prototype at same position with different label ' + str(newPrototypeLabel))
            newDistMatrix[winnerIndices, 4] = LVQCommon.getCostFunctionValue(distances[winnerIndices],
                                                                             newDistMatrix[winnerIndices, 3],
                                                                             activFct=activFct,
                                                                             logisticFactor=logisticFactor)
            newDistMatrix[looserIndices, 2] = newPrototypeIdx
            newDistMatrix[looserIndices, 3] = distances[looserIndices]
            if checkOverlapping and not all(distances[looserIndices] + newDistMatrix[looserIndices, 1]):
                logging.warning('inserted 2. prototype at same position with different label ' + str(newPrototypeLabel))
            newDistMatrix[looserIndices, 4] = LVQCommon.getCostFunctionValue(newDistMatrix[looserIndices, 1],
                                                                             distances[looserIndices],
                                                                             activFct=activFct,
                                                                             logisticFactor=logisticFactor)
            if protoStatistics is not None:
                numCorrect = len(
                    np.where((newDistMatrix[:, 0] == newPrototypeIdx) & (newDistMatrix[:, 1] < newDistMatrix[:, 3]))[0])
                numWrong = len(
                    np.where((newDistMatrix[:, 2] == newPrototypeIdx) & (newDistMatrix[:, 1] >= newDistMatrix[:, 3]))[
                        0])
                protoStatistics[newPrototypeIdx, 0] = numCorrect + numWrong
                protoStatistics[newPrototypeIdx, 1] = numCorrect
                protoStatistics[newPrototypeIdx, 2] = numWrong
        return newDistMatrix, protoStatistics

    @staticmethod
    def deleteProtoFromDistanceMatrix(distMatrix, samples, samplesLabels, prototypes, prototypesLabels, protoIdx, getDistanceFct,
                                      protoStatistics=None):
        newDistMatrix = distMatrix.copy()
        tmpProtos = prototypes.copy()
        tmpLabels = prototypesLabels.copy()

        tmpProtos = np.delete(tmpProtos, protoIdx, 0)
        tmpLabels = np.delete(tmpLabels, protoIdx, 0)
        UpdateWinnerIndices = np.where(newDistMatrix[:, 0] > protoIdx)[0]

        UpdateLooserIndices = np.where(newDistMatrix[:, 2] > protoIdx)[0]

        winnerIndices = np.where(newDistMatrix[:, 0] == protoIdx)[0]

        if len(winnerIndices) > 0:
            minWinnerIndices = np.empty(shape=(1, 0))
            minWinnerDistances = np.empty(shape=(1, 0))
            for idx in winnerIndices:
                winnerIndex, winnerDist = LVQCommon._getWinnerPrototypeIdx(samples[idx], samplesLabels[idx], tmpProtos,
                                                                      tmpLabels, getDistanceFct)
                minWinnerIndices = np.append(minWinnerIndices, winnerIndex)
                minWinnerDistances = np.append(minWinnerDistances, winnerDist)

            newDistMatrix[winnerIndices, 0] = minWinnerIndices
            newDistMatrix[winnerIndices, 1] = minWinnerDistances
            newDistMatrix[winnerIndices, 4] = LVQCommon.getCostFunctionValue(minWinnerDistances,
                                                                                 newDistMatrix[winnerIndices, 3])

        looserIndices = np.where(newDistMatrix[:, 2] == protoIdx)[0]
        if len(looserIndices) > 0:
            minLooserIndices = np.empty(shape=(1, 0))
            minlooserDistances = np.empty(shape=(1, 0))
            for idx in looserIndices:
                looserIdx, looserDist = LVQCommon._getLooserPrototypeIdx(samples[idx], samplesLabels[idx], tmpProtos,
                                                                    tmpLabels, getDistanceFct)
                minLooserIndices = np.append(minLooserIndices, looserIdx)
                minlooserDistances = np.append(minlooserDistances, looserDist)

            newDistMatrix[looserIndices, 2] = minLooserIndices
            newDistMatrix[looserIndices, 3] = minlooserDistances
            newDistMatrix[looserIndices, 4] = LVQCommon.getCostFunctionValue(newDistMatrix[looserIndices, 1],
                                                                             minlooserDistances)

        newDistMatrix[UpdateWinnerIndices, 0] -= 1
        newDistMatrix[UpdateLooserIndices, 2] -= 1

        if protoStatistics is not None:
            protoStatistics = np.delete(protoStatistics, protoIdx, 0)

            correctIndicesBefore = winnerIndices[
                np.where(distMatrix[winnerIndices, 1] < distMatrix[winnerIndices, 3])[0]]
            correctIndicesAfter = winnerIndices[
                np.where(newDistMatrix[winnerIndices, 1] < newDistMatrix[winnerIndices, 3])[0]]
            protoIndices, counts = np.unique(newDistMatrix[correctIndicesAfter, 0], return_counts=True)
            protoIndices = protoIndices.astype(int)
            protoStatistics[protoIndices, 0] += counts
            protoStatistics[protoIndices, 1] += counts

            wrongIndicesAfter = np.setdiff1d(correctIndicesBefore, correctIndicesAfter)
            protoIndices, counts = np.unique(newDistMatrix[wrongIndicesAfter, 2], return_counts=True)
            protoIndices = protoIndices.astype(int)
            protoStatistics[protoIndices, 0] += counts
            protoStatistics[protoIndices, 2] += counts

            wrongIndicesBefore = looserIndices[
                np.where(distMatrix[looserIndices, 1] >= distMatrix[looserIndices, 3])[0]]
            wrongIndicesAfter = looserIndices[
                np.where(newDistMatrix[looserIndices, 1] >= newDistMatrix[looserIndices, 3])[0]]

            protoIndices, counts = np.unique(newDistMatrix[wrongIndicesAfter, 2], return_counts=True)
            protoIndices = protoIndices.astype(int)
            protoStatistics[protoIndices, 0] += counts
            protoStatistics[protoIndices, 2] += counts

            correctIndicesAfter = np.setdiff1d(wrongIndicesBefore, wrongIndicesAfter)
            protoIndices, counts = np.unique(newDistMatrix[correctIndicesAfter, 0], return_counts=True)
            protoIndices = protoIndices.astype(int)
            protoStatistics[protoIndices, 0] += counts
            protoStatistics[protoIndices, 1] += counts

        return newDistMatrix, protoStatistics

    @staticmethod
    def getDistanceMatrix(samples, samplesLabels, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct):
        resultList = np.empty(shape=(0, 5))
        for i in range(len(samples)):
            values = LVQCommon._getWinnerLooserPrototypeIndices(samples[i, :], samplesLabels[i], prototypes, prototypesLabels, getDistanceFct)
            winnerIdx = int(values[0][0])

            looserIdx = int(values[1][0])
            winnerDist = values[0][1]

            looserDist = values[1][1]
            resultList = np.vstack([resultList,
                                    np.array([winnerIdx, winnerDist, looserIdx, looserDist,
                                              LVQCommon.getCostFunctionValue(winnerDist, looserDist,
                                                                             activFct=activFct,
                                                                             logisticFactor=logisticFactor)])])
        return resultList

