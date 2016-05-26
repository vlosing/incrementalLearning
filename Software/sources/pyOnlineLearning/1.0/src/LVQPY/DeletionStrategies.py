__author__ = 'viktor'
import numpy as np
from LVQCommon import LVQCommon
import logging
from DistanceMatrix import DistanceMatrix
from sklearn.cross_validation import train_test_split
#XXVL refactoring not finished!
class DeletionStrategies(object):

    @staticmethod
    def getPrototypeDelIndices(deletionStrategy, distanceMatrix, samples, labels, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct, protoStatistics, trainStepCount):
        if deletionStrategy is None:
            delIndices = []
        elif deletionStrategy == 'statistics':
            delIndices = DeletionStrategies.getDeletionIndicesByStatistics(protoStatistics)
        elif deletionStrategy == 'obsolete':
            delIndices = DeletionStrategies.getDeletionIndicesOfObsolete(protoStatistics, trainStepCount)
        elif deletionStrategy == 'windowAccReg':
            delIndices = DeletionStrategies.getDeletionIdxByWindowAccReg2(1, samples, labels, distanceMatrix, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct)
        elif deletionStrategy == 'windowAccMax':
            #delIndices = DeletionStrategies.getDeletionIndicesOfObsolete(self._protoStatistics, self._trainStepCount)
            #self._removeProtoIndices(delIndices)
            delIndices = DeletionStrategies.getDeletionsByWindowAccMax(samples, labels, distanceMatrix, prototypes, prototypesLabels, getDistanceFct)
        elif deletionStrategy == 'accBelowChance':
            delIndices = DeletionStrategies.getDeletionsBelowChanceAcc(samples, labels, distanceMatrix, prototypes, prototypesLabels, getDistanceFct)
        else:
            raise Exception('unknown deletion strategy ' + str(deletionStrategy))
        return delIndices

    @staticmethod
    def getDeletionIndicesByStatistics(protoStatistics):
        return np.where(protoStatistics[:, 1] < protoStatistics[:, 2])[0]

    @staticmethod
    def getDeletionIndicesOfObsolete(protoStatistics, currentTrainStep):
        return np.where(protoStatistics[:, 3] < currentTrainStep - 5000)[0]


    @staticmethod
    def getDeletionIndicesOutOfWindow(distanceMatrix, prototypes):
        protoIndices = np.unique(distanceMatrix[:, [0, 2]])
        return np.setdiff1d(np.arange(len(prototypes)), protoIndices)

    @staticmethod
    def getDeletionsByWindowAcc(numDeletions, samples, sampleLabels, distMatrix, prototypes, prototypesLabels, getDistanceFct):
        distMatrixTmp = np.copy(distMatrix)
        tmpPrototypes = prototypes.copy()
        tmpPrototypesLabels = prototypesLabels.copy()

        for d in range(numDeletions):
            initialValue = LVQCommon.getAccuracyByDistanceMatrix(distMatrixTmp)
            maxValue = initialValue
            bestPrototypeIdx = None
            protoIndices, counts = np.unique(distMatrixTmp[:, [0, 2]], return_counts=True)
            protoIndices = protoIndices.astype(np.int)
            countIndices = np.where(counts > 0)
            protoIndices = protoIndices[countIndices]
            delIndices = []
            permutation = np.random.permutation(protoIndices)

            for i in range(len(permutation)):
                protoCandidateIdx = permutation[i]
                if len(np.where(tmpPrototypesLabels == tmpPrototypesLabels[protoCandidateIdx])[0]) > 1:
                    newDistMatrix, dummy = DistanceMatrix.deleteProtoFromDistanceMatrix(distMatrixTmp, samples, sampleLabels, tmpPrototypes, tmpPrototypesLabels,
                                                                                 protoCandidateIdx, getDistanceFct)
                    newValue = LVQCommon.getAccuracyByDistanceMatrix(newDistMatrix)
                    if newValue >= maxValue:
                        maxValue = newValue
                        bestPrototypeIdx = protoCandidateIdx
                        distMatrixTmp = newDistMatrix

            if bestPrototypeIdx is not None:  # and maxAvgRelSim > (averageBefore + 0.01):
                delIndices.append(bestPrototypeIdx)
                tmpPrototypesLabels = np.delete(tmpPrototypesLabels, bestPrototypeIdx, 0)
                tmpPrototypes = np.delete(tmpPrototypes, bestPrototypeIdx, 0)
            else:
                #logging.info('deletion not found')
                break

            delIndices = DeletionStrategies.fixDelIndices(delIndices)
        return delIndices

    @staticmethod
    def getDeletionsBelowChanceAcc(samples, sampleLabels, distMatrix, prototypes, prototypesLabels, getDistanceFct):
        majAccuracy = 1./len(np.unique(sampleLabels))
        correctProtoIndices = np.where(distMatrix[:, 1] < distMatrix[:, 3])[0]
        wrongProtoIndices = np.where(distMatrix[:, 1] >= distMatrix[:, 3])[0]
        binCountCorrect = np.bincount(distMatrix[correctProtoIndices][:, 0].astype(int), minlength=len(prototypesLabels))
        binCountWrong = np.bincount(distMatrix[wrongProtoIndices][:, 2].astype(int), minlength=len(prototypesLabels))
        accuracies = binCountCorrect / (1. * (binCountCorrect + binCountWrong))
        deletedIndices = np.where(accuracies <= majAccuracy)[0]
        return deletedIndices


    @staticmethod
    def getDeletionsByWindowAccMax(samples, sampleLabels, distMatrix, prototypes, prototypesLabels, getDistanceFct):
        distMatrixTmp = np.copy(distMatrix)
        tmpPrototypes = prototypes.copy()
        tmpPrototypesLabels = prototypesLabels.copy()
        delIndices = []
        while True:
            initialValue = LVQCommon.getAccuracyByDistanceMatrix(distMatrixTmp)
            maxValue = initialValue
            bestPrototypeIdx = None
            protoIndices, counts = np.unique(distMatrixTmp[:, [0, 2]], return_counts=True)
            #print protoIndices

            protoIndices = protoIndices.astype(np.int)
            countIndices = np.where(counts > 0)
            protoIndices = protoIndices[countIndices]
            permutation = np.random.permutation(protoIndices)
            numTries = min(len(permutation), 20)
            for i in range(numTries):
                protoCandidateIdx = permutation[i]
                if len(np.where(tmpPrototypesLabels == tmpPrototypesLabels[protoCandidateIdx])[0]) > 1:
                    newDistMatrix, dummy = DistanceMatrix.deleteProtoFromDistanceMatrix(distMatrixTmp, samples, sampleLabels, tmpPrototypes, tmpPrototypesLabels,
                                                                                 protoCandidateIdx, getDistanceFct)
                    newValue = LVQCommon.getAccuracyByDistanceMatrix(newDistMatrix)
                    if newValue >= maxValue:
                        maxValue = newValue
                        bestPrototypeIdx = protoCandidateIdx
                        distMatrixTmp = newDistMatrix
            if bestPrototypeIdx is not None:  # and maxAvgRelSim > (averageBefore + 0.01):
                delIndices.append(bestPrototypeIdx)
                tmpPrototypesLabels = np.delete(tmpPrototypesLabels, bestPrototypeIdx, 0)
                tmpPrototypes = np.delete(tmpPrototypes, bestPrototypeIdx, 0)
            else:
                #logging.info('deletion not found')
                break

        delIndices = DeletionStrategies.fixDelIndices(delIndices)
        return delIndices

    @staticmethod
    def getDeletionIdxByWindowAccReg(minDeletions, samples, sampleLabels, distMatrix, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct):
        X_train, X_test, y_train, y_test = train_test_split(samples, sampleLabels, test_size=0.33)
        distMatrixTrain = DistanceMatrix.getDistanceMatrix(X_train, y_train, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct)
        distMatrixTest = DistanceMatrix.getDistanceMatrix(X_test, y_test, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct)
        trainAcc = [LVQCommon.getAccuracyByDistanceMatrix(distMatrixTrain)]
        testAcc = [LVQCommon.getAccuracyByDistanceMatrix(distMatrixTest)]
        tmpPrototypesLabels = prototypesLabels.copy()
        tmpPrototypes = prototypes.copy()
        iteration = 0
        deletedIndices = []
        while iteration < minDeletions or testAcc[-1] > testAcc[-2]:
            bestPrototypeIdx = None
            bestDistMatrix = None
            maxTrainAcc = 0
            protoIndices, counts = np.unique(distMatrixTrain[:, [0, 2]], return_counts=True)
            protoIndices = protoIndices.astype(np.int)
            countIndices = np.where(counts > 0)
            protoIndices = protoIndices[countIndices]
            permutation = np.random.permutation(protoIndices)

            for i in range(len(permutation)):
                protoCandidateIdx = permutation[i]
                if len(np.where(tmpPrototypesLabels == tmpPrototypesLabels[protoCandidateIdx])[0]) > 1:
                    newDistMatrix, dummy = DistanceMatrix.deleteProtoFromDistanceMatrix(distMatrixTrain, samples, sampleLabels, tmpPrototypes, tmpPrototypesLabels,
                                                                                 protoCandidateIdx, getDistanceFct)
                    newAcc = LVQCommon.getAccuracyByDistanceMatrix(newDistMatrix)
                    if newAcc > maxTrainAcc:
                        maxTrainAcc = newAcc
                        bestPrototypeIdx = protoCandidateIdx
                        bestDistMatrix = np.copy(newDistMatrix)
            if bestPrototypeIdx is None:
                break
            distMatrixTrain = bestDistMatrix
            trainAcc.append(maxTrainAcc)
            distMatrixTest, dummy = DistanceMatrix.deleteProtoFromDistanceMatrix(distMatrixTest, samples, sampleLabels, tmpPrototypes, tmpPrototypesLabels,
                                                                                 bestPrototypeIdx, getDistanceFct)
            testAcc.append(LVQCommon.getAccuracyByDistanceMatrix(distMatrixTest))
            deletedIndices.append(bestPrototypeIdx)
            tmpPrototypesLabels = np.delete(tmpPrototypesLabels, bestPrototypeIdx, 0)
            tmpPrototypes = np.delete(tmpPrototypes, bestPrototypeIdx, 0)
            iteration += 1
        numDeletions = np.argmax(testAcc)
        delIndices = []
        if numDeletions > 0:
            delIndices = DeletionStrategies.getDeletionsByWindowAcc(numDeletions, samples, sampleLabels, distMatrix, prototypes, prototypesLabels, getDistanceFct)
            #print numDeletions, len(delIndices)
        return delIndices


    @staticmethod
    def getDeletionIdxByWindowAccReg2(minDeletions, samples, sampleLabels, distMatrix, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct):
        X_train, X_test, y_train, y_test = train_test_split(samples, sampleLabels, test_size=0.33)
        distMatrixTrain = DistanceMatrix.getDistanceMatrix(X_train, y_train, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct)
        distMatrixTest = DistanceMatrix.getDistanceMatrix(X_test, y_test, prototypes, prototypesLabels, activFct, logisticFactor, getDistanceFct)
        trainAcc = [LVQCommon.getAccuracyByDistanceMatrix(distMatrixTrain)]
        testAcc = [LVQCommon.getAccuracyByDistanceMatrix(distMatrixTest)]
        tmpPrototypesLabels = prototypesLabels.copy()
        tmpPrototypes = prototypes.copy()
        iteration = 0
        delIndices = []
        while iteration < minDeletions or testAcc[-1] > testAcc[-2]:
            bestPrototypeIdx = None
            bestDistMatrix = None
            maxTrainAcc = 0
            protoIndices, counts = np.unique(distMatrixTrain[:, [0, 2]], return_counts=True)
            protoIndices = protoIndices.astype(np.int)
            countIndices = np.where(counts > 0)
            protoIndices = protoIndices[countIndices]
            permutation = np.random.permutation(protoIndices)
            numTries = min(len(permutation), 50)
            for i in range(numTries):
                protoCandidateIdx = permutation[i]
                if len(np.where(tmpPrototypesLabels == tmpPrototypesLabels[protoCandidateIdx])[0]) > 1:
                    newDistMatrix, dummy = DistanceMatrix.deleteProtoFromDistanceMatrix(distMatrixTrain, samples, sampleLabels, tmpPrototypes, tmpPrototypesLabels,
                                                                                 protoCandidateIdx, getDistanceFct)
                    newAcc = LVQCommon.getAccuracyByDistanceMatrix(newDistMatrix)
                    if newAcc > maxTrainAcc:
                        maxTrainAcc = newAcc
                        bestPrototypeIdx = protoCandidateIdx
                        bestDistMatrix = np.copy(newDistMatrix)
            if bestPrototypeIdx is None:
                break
            distMatrixTrain = bestDistMatrix
            trainAcc.append(maxTrainAcc)
            distMatrixTest, dummy = DistanceMatrix.deleteProtoFromDistanceMatrix(distMatrixTest, samples, sampleLabels, tmpPrototypes, tmpPrototypesLabels,
                                                                                 bestPrototypeIdx, getDistanceFct)
            testAcc.append(LVQCommon.getAccuracyByDistanceMatrix(distMatrixTest))
            delIndices.append(bestPrototypeIdx)
            tmpPrototypesLabels = np.delete(tmpPrototypesLabels, bestPrototypeIdx, 0)
            tmpPrototypes = np.delete(tmpPrototypes, bestPrototypeIdx, 0)
            iteration += 1
        stopIdx = np.argmax(testAcc)
        delIndices = delIndices[:stopIdx]
        delIndices = DeletionStrategies.fixDelIndices(delIndices)

        return delIndices[:stopIdx]


    @staticmethod
    def fixDelIndices(delIndices):
        if len(delIndices) > 1:
            correctedIndices = [delIndices[0]]
            for i in range(1, len(delIndices)):
                correctedIndices.append(len(np.where(delIndices[:i] <= delIndices[i])[0]) + delIndices[i])
            return correctedIndices
        else:
            return delIndices

    @staticmethod
    def combineDelIndices(indices, indices2):
        if len(indices) > 0 and len(indices2) > 0:
            corrIndices2 = []
            for i in range(0, len(indices2)):
                corrIndices2.append(len(np.where(indices <= indices2[i])[0]) + indices2[i])
            combIndices = np.append(indices, corrIndices2)
        else:
            combIndices =  np.append(indices, indices2)
        #print indices, indices2, combIndices
        return combIndices

    '''@staticmethod
    def _removeWorstPrototype3(samples, sampleLabels, distMatrix, minOccurence=3):
        distMatrixTmp = np.copy(distMatrix)
        # minCost = sys.maxint
        initialCost = np.average(LVQCommon.getCostFunctionValuesByMatrix(distMatrixTmp))
        protoIndices, counts = np.unique(distMatrixTmp[:, [0, 2]], return_counts=True)
        protoIndices = protoIndices.astype(np.int)
        countIndices = np.where(counts > 0)
        protoIndices = protoIndices[countIndices]

        permutation = np.random.permutation(protoIndices)

        for i in range(len(permutation)):
            protoCandidateIdx = permutation[i]
            if len(np.where(self._prototypesLabels == self._prototypesLabels[protoCandidateIdx])[0]) > 1:
                newDistMatrix = self.deleteProtoFromDistanceMatrix(distMatrixTmp, samples, sampleLabels,
                                                                   protoCandidateIdx)
                costValue = np.average(LVQCommon.getCostFunctionValuesByMatrix(newDistMatrix))
                deltaCost = initialCost - costValue
                self._protoStatistics[protoCandidateIdx, 5] += deltaCost
                self._protoStatistics[protoCandidateIdx, 6] += 1

        indices = np.where((self._protoStatistics[:, 5] >= 0) & (self._protoStatistics[:, 6] >= minOccurence))[0]

        self._removeProtoIndices(indices)'''