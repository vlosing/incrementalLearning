#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Module containing implementations of GLVQ, GMLVQ, LIRAMLVQ. These
# models can be trained off- and on-line. In the on-line setting the prototype insertion strategy SamplingCost is used,
# which inserts prototype to minimize the cost of a window containing recent samples.
#
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
import unittest
import logging
import numpy as np
from LVQPY import LVQFactory
from fractions import Fraction
from grlvqPythonTest.compareToCalcValues.compareAllValues import LVQSimple
from LVQCommon import LVQCommon

class TestLVQ(unittest.TestCase):

    def getDefGLVQ(self):
        return LVQFactory.getLVQClassifier(3, netType='GLVQ', learnRateInitial=1, protoAdds=1, insertionTimingThresh=1)

    def getDefGMLVQ(self):
        return LVQFactory.getLVQClassifier(3, netType='GMLVQ', learnRateInitial=1, protoAdds=1, insertionTimingThresh=1,
                                           metricLearnRate=0.03)

    def getDefLIRAMLVQ(self, LIRAMLVQDimensions):
        return LVQFactory.getLVQClassifier(3, netType='LIRAMLVQ', LIRAMLVQDimensions=LIRAMLVQDimensions,
                                           learnRateInitial=1, protoAdds=1, insertionTimingThresh=1,
                                           metricLearnRate=0.03)
    def getTestPrototypes(self):
        protos = np.empty(shape=(0, 3))
        protos = np.vstack([protos, np.array([-2, -2, -2])])
        protos = np.vstack([protos, np.array([-2, -3, -4])])
        protos = np.vstack([protos, np.array([0, 0, 0])])
        protos = np.vstack([protos, np.array([1, 1, 1])])
        protos = np.vstack([protos, np.array([3, 3, 3])])
        protos = np.vstack([protos, np.array([3, 4, 5])])

        labels = np.array([0, 0, 1, 1, 2, 2])
        return protos, labels


    def getTestData(self):
        samples = np.empty(shape=(0, 3))
        samples = np.vstack([samples, np.array([-2, -3, -2])])
        samples = np.vstack([samples, np.array([-3, -4, -5])])
        samples = np.vstack([samples, np.array([1, 0, 0])])
        samples = np.vstack([samples, np.array([1, 2, 2])])
        samples = np.vstack([samples, np.array([4, 3, 4])])
        samples = np.vstack([samples, np.array([5, 6, 7])])

        labels = np.array([0, 0, 1, 1, 2, 2])
        return samples, labels


    def getTestDataExt(self):
        samples = np.empty(shape=(0, 3))
        cov = [[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]]
        samples = np.vstack([samples, np.random.multivariate_normal([0, 0, 0], cov, 100)])
        samples = np.vstack([samples, np.random.multivariate_normal([1, 1, 1], cov, 100)])
        samples = np.vstack([samples, np.random.multivariate_normal([-1, -1, -1], cov, 100)])

        labels = np.append(np.zeros(shape=(100)), np.ones(shape=(100)))
        labels = np.append(labels, 2 * np.ones(shape=(100)))
        return samples, labels

    def test_getDistance(self):

        A = np.atleast_2d(np.array([0, 0, 0])).T
        B = np.atleast_2d(np.array([0, 0, 0])).T

        omegaMetric = np.sqrt(np.array([[float(Fraction('1/3')), 0, 0],
                                        [0, float(Fraction('1/3')), 0],
                                        [0, 0, float(Fraction('1/3'))]]))

        self.assertEqual(LVQCommon.getSquaredDistance(A, B), 0.)
        self.assertEqual(LVQCommon.getMetricDistance(A, B, omegaMetric), 0.)

        B = np.atleast_2d(np.array([1, 1, 1])).T
        self.assertTrue(np.isclose(LVQCommon.getSquaredDistance(A, B), 3.))
        self.assertEqual(LVQCommon.getMetricDistance(A, B, omegaMetric), 1.)

        A = np.atleast_2d(np.array([1, 2, 3])).T
        B = np.atleast_2d(np.array([3, 2, 1])).T
        self.assertTrue(np.isclose(LVQCommon.getSquaredDistance(A, B), 8.))
        self.assertTrue(np.isclose(LVQCommon.getMetricDistance(A, B, omegaMetric), float(Fraction('8/3'))))

        A = np.array([[0, 0, 0], [0, 0, 0], [1, 2, 3]]).T
        B = np.array([[0, 0, 0], [1, 1, 1], [3, 2, 1]]).T
        self.assertTrue(np.all(np.isclose(LVQCommon.getSquaredDistance(A, B), [0, 3, 8])))
        self.assertTrue(np.all(np.isclose(LVQCommon.getMetricDistance(A, B, omegaMetric), [0, 1, float(Fraction('8/3'))])))

    def test_logisticFunction(self):
        X = np.array([0, float(Fraction('1/4')), 100, -100])
        Y = np.array([0.5, 0.5621765, 1, 0])
        for i in range(len(X)):
            self.assertTrue(np.isclose(LVQCommon.logisticFunction(X[i]), Y[i]))
        self.assertTrue(np.all(np.isclose(LVQCommon.logisticFunction(X), Y)))

        Y = np.array([0.5, 0.6224593, 1, 0])
        for i in range(len(X)):
            self.assertTrue(np.isclose(LVQCommon.logisticFunction(X[i], logisticFactor=2), Y[i]))
        self.assertTrue(np.all(np.isclose(LVQCommon.logisticFunction(X, logisticFactor=2), Y)))

    def test_muFunction(self):
        X = np.array([0, 1, 1])
        X2 = np.array([1, 0, 1])
        Y = np.array([-1, 1, 0])
        for i in range(len(X)):
            self.assertTrue(np.isclose(LVQCommon.getMuValue(X[i], X2[i]), Y[i]))

    def test_costFunction(self):
        X = np.array([0, 1, 1])
        X2 = np.array([1, 0, 1])
        Y = np.array([-1, 1, 0])
        for i in range(len(X)):
            self.assertTrue(np.isclose(LVQCommon.getCostFunctionValue(X[i], X2[i], activFct='linear'), Y[i]))

        Y = np.array([0.268941, 0.731058578, 0.5])
        for i in range(len(X)):
            self.assertTrue(np.isclose(LVQCommon.getCostFunctionValue(X[i], X2[i],
                                                                           activFct='logistic', logisticFactor=1), Y[i]))

    def test__getInternLearnRate(self):
        self.assertTrue(np.isclose(LVQCommon.getLinearAnnealedLearnRate(1, 1, 10), 0.9))
        self.assertTrue(np.isclose(LVQCommon.getLinearAnnealedLearnRate(1, 2, 10), 0.8))
        self.assertTrue(np.isclose(LVQCommon.getLinearAnnealedLearnRate(1, 10, 10), 0))
        self.assertTrue(np.isclose(LVQCommon.getLinearAnnealedLearnRate(10, 1, 10), 9))
        self.assertTrue(np.isclose(LVQCommon.getLinearAnnealedLearnRate(10, 5, 10), 5))
        self.assertTrue(np.isclose(LVQCommon.getLinearAnnealedLearnRate(10, 10, 10), 0))

    def test__trainGLVQ(self):
        classifier = self.getDefGLVQ()
        prototypeGroundTruth = np.array([[-2.00889649, -2.05855573, -2.005931],
                                         [-2.01774548, -3.01774548, -4.01774548],
                                         [ 0.06087249,  0.01281067,  0.01088751],
                                         [ 0.98327051,  1.07939006,  1.07433934],
                                         [ 3.10019049,  3.02999365,  3.07019684],
                                         [ 3.01977519,  4.01977519,  5.01977519]])
        protos, labels = self.getTestPrototypes()
        for i in range(len(protos)):
            classifier.addPrototype(protos[i], labels[i])
        samples, labels = self.getTestData()
        for i in range(len(samples)):
            classifier.train(samples[i], labels[i])
        self.assertTrue(np.all(np.isclose(classifier.prototypes, prototypeGroundTruth)))

    def test__compareGLVQWithSimpleImpl(self):
        self.assertTrue(LVQSimple.compareGLVQValues2(LVQFactory))

    def test__compareGMLVQWithSimpleImpl(self):
        self.assertTrue(LVQSimple.compareGMLVQOwnImplValues(LVQFactory))


    def test__trainGMLVQ(self):
        classifier = self.getDefGMLVQ()
        prototypeGroundTruth = np.array([[-2.00880014, -2.05848001, -2.0059488],
                                         [-2.01781789, -3.01762296, -4.01781789],
                                         [ 0.06049804,  0.01304282,  0.01108595],
                                         [ 0.98422346,  1.07900505,  1.07561777],
                                         [ 3.10039794,  3.03141866,  3.0701588 ],
                                         [ 3.02114027,  4.019389,    5.01935263]])
        metricWeightsGroundTruth = np.array([[  3.43684716e-01,   1.27685082e-02,   1.00008612e-02],
                                             [  1.27685082e-02,   3.25689889e-01,  -1.04991814e-04],
                                             [  1.00008612e-02,  -1.04991814e-04,  3.30625395e-01]])

        protos, labels = self.getTestPrototypes()
        for i in range(len(protos)):
            classifier.addPrototype(protos[i], labels[i])
        samples, labels = self.getTestData()
        for i in range(len(samples)):
            classifier.train(samples[i], labels[i])

        self.assertTrue(np.all(np.isclose(classifier.prototypes, prototypeGroundTruth)))
        self.assertTrue(np.all(np.isclose(classifier.metricWeights, metricWeightsGroundTruth)))

    def test__trainLIRAMLVQ(self):
        classifier = self.getDefLIRAMLVQ(3)
        prototypeGroundTruth = np.array([[-2.00880014, -2.05848001, -2.0059488 ],
                                         [-2.01781789, -3.01762296, -4.01781789],
                                         [ 0.06049804,  0.01304282,  0.01108595],
                                         [ 0.9842232,   1.07900512,  1.0756178 ],
                                         [ 3.10039853,  3.03141819,  3.07015868],
                                         [ 3.02114044,  4.01938902,  5.0193525 ]])
        metricWeightsGroundTruth = np.array([[  3.43686458e-01,   1.27675475e-02,   9.99945269e-03],
                                             [  1.27675475e-02,   3.25689831e-01,  -1.05997063e-04],
                                             [  9.99945269e-03,  -1.05997063e-04,   3.30623711e-01]])



        protos, labels = self.getTestPrototypes()
        for i in range(len(protos)):
            classifier.addPrototype(protos[i], labels[i])
        samples, labels = self.getTestData()
        for i in range(len(samples)):
            classifier.train(samples[i], labels[i])

        self.assertTrue(np.all(np.isclose(classifier.prototypes, prototypeGroundTruth)))
        self.assertTrue(np.all(np.isclose(classifier.metricWeights, metricWeightsGroundTruth)))

    def test__trainLIRAMLVQReducedDimensions(self):
        np.random.seed(0)
        classifier = self.getDefLIRAMLVQ(2)
        prototypeGroundTruth = np.array([[-2.0072655,  -2.05175259, -2.01159559],
                                         [-2.00789657, -3.02379036, -4.01854819],
                                         [ 0.01064089,  0.02457404,  0.02653574],
                                         [ 1.03245975,  1.11616701,  1.07553055],
                                         [ 3.04676224,  3.10141238,  3.11561118],
                                         [ 3.00969766,  4.02699959,  5.02098416]])
        metricWeightsGroundTruth = np.array([[ 0.05652329,  0.09112037,  0.13823173],
                                             [ 0.09112037,  0.5834361,   0.12487502],
                                             [ 0.13823173,  0.12487502,  0.36004062]])

        protos, labels = self.getTestPrototypes()
        for i in range(len(protos)):
            classifier.addPrototype(protos[i], labels[i])
        samples, labels = self.getTestData()
        for i in range(len(samples)):
            classifier.train(samples[i], labels[i])

        self.assertTrue(np.all(np.isclose(classifier.prototypes, prototypeGroundTruth)))
        self.assertTrue(np.all(np.isclose(classifier.metricWeights, metricWeightsGroundTruth)))


    def test__trainInc(self):
        np.random.seed(0)
        classifier = self.getDefGLVQ()
        prototypeGroundTruth = np.array([[ 2.43326851, -2.00828344, -1.37265431],
                                         [ 1.12967705,  0.79196225,  1.04036162],
                                         [-2.87081911,  0.60291708,  1.43261907],
                                         [-1.41571153, -0.48961244, -0.99363734],
                                         [ 1.95241979,  1.54465224,  1.72252102],
                                         [-0.76926878, -1.37253076, -1.10385463]])
        samples, labels = self.getTestDataExt()
        for i in range(len(samples)):
            classifier.trainInc(samples[i], labels[i])
        self.assertTrue(np.all(np.isclose(classifier.prototypes, prototypeGroundTruth)))


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    unittest.main()