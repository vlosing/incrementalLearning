#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# <description>
#
# Copyright (C)
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
#
# UNPUBLISHED PROPRIETARY MATERIAL.
# ALL RIGHTS RESERVED.
#
#
import numpy as np
import sys
import math
class LVQCommon(object):
    @staticmethod
    def logisticFunction(x, logisticFactor=1):
        return 1 / (1 + np.exp(-logisticFactor * x))

    @staticmethod
    def getMuValue(winnerDist, looserDist):
        if winnerDist + looserDist > 0:
            return (winnerDist - looserDist) / (winnerDist + looserDist)
        else:
            return 0

    @staticmethod
    def getCostFunctionValuesByMatrix(distMatrix, ignoreIdx=None):
        if not (ignoreIdx is None):
            result = distMatrix[0:ignoreIdx, 4]
            result = np.append(result, distMatrix[ignoreIdx + 1:len(distMatrix), 4])
            return result
        else:
            return distMatrix[:, 4]

    @staticmethod
    def getAvgCostValueByDistanceMatrix(distMatrix):
        return np.average(distMatrix[:, 4])


    @staticmethod
    def getAccuracyByDistanceMatrix(distMatrix):
        if len(distMatrix) > 0:
            numCorrect = len(np.where(distMatrix[:, 1] < distMatrix[:, 3])[0])
            return numCorrect / float(len(distMatrix))
        else:
            return 0

    @staticmethod
    def getWeightedAccuracyByDistanceMatrix(distMatrix):
        if len(distMatrix) > 0:
            #weights = np.linspace(1,100, distMatrix.shape[0])
            weights = np.logspace(0,1, distMatrix.shape[0])
            correctIndices = np.where(distMatrix[:, 1] < distMatrix[:, 3])[0]
            return np.sum(weights[correctIndices]) / float(np.sum(weights))
        else:
            return 0

    @staticmethod
    def getCostFunctionValue(winnerDist, looserDist, activFct='linear', logisticFactor=1):
        winnerDist = np.array(winnerDist, dtype=float)
        looserDist = np.array(looserDist, dtype=float)
        if len(winnerDist.shape) == 0:
            winnerDist = np.array([winnerDist], dtype=float)
            looserDist = np.array([looserDist], dtype=float)

        nonZeroIndices = np.where((winnerDist + looserDist) != 0)[0]

        costFunctionValue = np.zeros(len(winnerDist))
        if len(nonZeroIndices) > 0:
            costFunctionValue[nonZeroIndices] = (winnerDist[nonZeroIndices] - looserDist[nonZeroIndices]) / (
                winnerDist[nonZeroIndices] + looserDist[nonZeroIndices])
            if activFct == 'logistic':
                costFunctionValue[nonZeroIndices] = LVQCommon.logisticFunction(costFunctionValue[nonZeroIndices],
                                                                                    logisticFactor=logisticFactor)
        return costFunctionValue

    @staticmethod
    def getSquaredDistance(A, B):
        return np.sum((A - B) ** 2, 0)

    @staticmethod
    def getMetricDistance(A, B, omegaMetricWeights):
        diff = (A - B)
        tmp = np.dot(np.transpose(diff), omegaMetricWeights.T)
        return np.einsum('ij,ji->i', tmp, np.transpose(tmp))

    @staticmethod
    def getLinearAnnealedLearnRate(learnRateInitial, stepCount, learnRateAnnealingSteps):
        if learnRateAnnealingSteps == 0:
            return learnRateInitial
        else:
            return max(1 - (stepCount / float(learnRateAnnealingSteps)), 0) * learnRateInitial
    @staticmethod
    def doesProtoExist(prototype, prototypes):
        indices = np.where(np.all(prototypes == prototype, axis=1))[0]
        if len(indices) > 0:
            return True
        return False


    @staticmethod
    def _getWinnerPrototypeIdx(sample, sampleLabel, protos, protoLabels, getDistanceFct):
        protoIdx = np.where(protoLabels == sampleLabel)
        if len(protoIdx) == 0:
            return -1, sys.maxint
        protosWithSameClass = protos[protoIdx]

        [minIdx, minDist] = LVQCommon._getPrototypeIdxWithMinDistance(protosWithSameClass, sample, getDistanceFct)
        if minIdx == -1:
            return minIdx, minDist
        else:
            return protoIdx[0][minIdx].astype(int), minDist

    @staticmethod
    def _getLooserPrototypeIdx(sample, sampleLabel, protos, protoLabels, getDistanceFct):
        protoIdx = np.where(protoLabels != sampleLabel)
        if len(protoIdx) == 0 or len(protoLabels) == 0:
            return -1, sys.maxint
        protosDifferentLabel = protos[protoIdx]
        [minIdx, minDist] = LVQCommon._getPrototypeIdxWithMinDistance(protosDifferentLabel, sample, getDistanceFct)
        if minIdx == -1:
            return minIdx, minDist
        else:
            return protoIdx[0][minIdx].astype(int), minDist

    @staticmethod
    def _getWinnerLooserPrototypeIndices(sample, sampleLabel, prototypes, prototypesLabels, getDistanceFct):
        return LVQCommon._getWinnerPrototypeIdx(sample, sampleLabel, prototypes, prototypesLabels, getDistanceFct), \
               LVQCommon._getLooserPrototypeIdx(sample, sampleLabel, prototypes, prototypesLabels, getDistanceFct)

    @staticmethod
    def _getPrototypeIdxWithMinDistance(protos, sample, getDistanceFct, excludedLabel=None, forcedLabel=None,
                                        protoLabels=np.array([])):
        tmpSample = sample.copy()
        tmpSample.shape = [len(sample), 1]

        if not (excludedLabel is None):
            relevantProtos = protos[np.where(protoLabels != excludedLabel)[0]]
        elif not (forcedLabel is None):
            relevantProtos = protos[np.where(protoLabels == forcedLabel)[0]]
        else:
            relevantProtos = protos
        sampleMat = tmpSample * np.ones(shape=[1, len(relevantProtos)])
        distances = getDistanceFct(np.transpose(relevantProtos), sampleMat)

        if len(distances) == 0:
            return -1, sys.maxint
        else:
            minIdx = np.argmin(distances)
            return minIdx, distances[minIdx]
