import json
import math

import numpy
import scipy.misc
import random
import matplotlib
import matplotlib.pyplot as plt

import generateNetworkValues
import libpythoninterface
import numpy as np
import logging

class LVQSimple(object):
    @staticmethod
    def getWinnerLooser(proto1, proto1Label, proto2, proto2Label, sampleLabel):
        if (sampleLabel == proto1Label):
            winner = proto1
            looser = proto2
        elif (sampleLabel == proto2Label):
            winner = proto2
            looser = proto1
        return (winner, looser)

    @staticmethod
    def trainGLVQ(proto1, proto1Label, proto2, proto2Label, learnrate, sample, sampleLabel):
        values = LVQSimple.getWinnerLooser(proto1, proto1Label, proto2, proto2Label, sampleLabel)
        winner = values[0]
        looser = values[1]
        winnerDist = numpy.linalg.norm(winner - sample)
        winnerDist *= winnerDist
        looserDist = numpy.linalg.norm(looser - sample)
        looserDist *= looserDist

        deltaWinner = [sample[0] - winner[0], sample[1] - winner[1]]
        factorWinner = learnrate * looserDist / ((winnerDist + looserDist) * (winnerDist + looserDist))
        winner[0] = winner[0] + factorWinner * deltaWinner[0]
        winner[1] = winner[1] + factorWinner * deltaWinner[1]
        deltaLooser = [sample[0] - looser[0], sample[1] - looser[1]]
        factorLooser = learnrate * winnerDist / ((winnerDist + looserDist) * (winnerDist + looserDist))
        looser[0] = looser[0] - factorLooser * deltaLooser[0]
        looser[1] = looser[1] - factorLooser * deltaLooser[1]

    @staticmethod
    def getNewLearnRate(sampleCount, maxSamples, startLearnRate):
        return max(1 - (sampleCount / float(maxSamples)), 0) * startLearnRate

    @staticmethod
    def initValues():
        f = open('/hri/localdisk/vlosing/OnlineLearning/Software/pyOnlineLearning/1.0/src/grlvqPythonTest/compareToCalcValues/data/traindataClass1.json', 'r')
        dataClass1 = numpy.array(json.load(f))
        f.close()
        f = open('/hri/localdisk/vlosing/OnlineLearning/Software/pyOnlineLearning/1.0/src/grlvqPythonTest/compareToCalcValues/data/traindataClass2.json', 'r')
        dataClass2 = numpy.array(json.load(f))
        f.close()
        proto1 = dataClass1[0, :]
        proto2 = dataClass2[0, :]
        startLearnRate = 1.0
        startMetricLearnRate = 0.01
        return (dataClass1, dataClass2, proto1, proto2, startLearnRate, startMetricLearnRate)

    @staticmethod
    def genGLVQ_OwnImplLRFALSEValues2(LVQFactory):
        initValues = LVQSimple.initValues()
        dataClass1 = initValues[0]
        dataClass2 = initValues[1]
        startLearnRate = initValues[4]
        maxSamples = len(dataClass1) * 2
        glvq = LVQFactory.getLVQClassifier(2, learnRateInitial=startLearnRate, learnRateAnnealingSteps=maxSamples, netType='GLVQ')
        glvq.addPrototype(initValues[2], 0)
        glvq.addPrototype(initValues[3], 1)
        for sample in range(len(dataClass1) - 2):
            glvq.train(dataClass1[sample + 1, :], 0)
            glvq.train(dataClass2[sample + 1, :], 1)
        prototypes = glvq.prototypes
        prototypes = numpy.round(prototypes, 3)
        return prototypes

    @staticmethod
    def genGLVQ_LRFALSEValues():
        initValues = LVQSimple.initValues()
        dataClass1 = initValues[0]
        dataClass2 = initValues[1]

        proto1 = initValues[2]
        proto2 = initValues[3]
        startLearnRate = initValues[4]
        prototypes = proto1
        prototypes = numpy.vstack([prototypes, proto2])
        prototypes = numpy.round(prototypes, 3)

        maxSamples = len(dataClass1) * 2
        sampleCount = 0
        learnrate = startLearnRate

        for sample in range(len(dataClass1) - 2):
            LVQSimple.trainGLVQ(proto1, 0, proto2, 1, learnrate, dataClass1[sample + 1, :], 0)
            sampleCount += 1
            learnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startLearnRate)
            LVQSimple.trainGLVQ(proto1, 0, proto2, 1, learnrate, dataClass2[sample + 1, :], 1)
            sampleCount += 1
            learnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startLearnRate)

        prototypes = proto1
        prototypes = numpy.vstack([prototypes, proto2])
        prototypes = numpy.round(prototypes, 3)
        return prototypes

    @staticmethod
    def trainGRLVQ(proto1, proto1Label, proto2, proto2Label, learnrate, learnrateMetricWeights, sample, sampleLabel,
                   metricWeights):
        values = LVQSimple.getWinnerLooser(proto1, proto1Label, proto2, proto2Label, sampleLabel)
        winner = values[0]
        looser = values[1]
        winnerDist = (winner - sample)[0] * (winner - sample)[0] * metricWeights[0] + (winner - sample)[1] * \
                                                                                      (winner - sample)[1] * metricWeights[
                                                                                          1]

        looserDist = (looser - sample)[0] * (looser - sample)[0] * metricWeights[0] + (looser - sample)[1] * \
                                                                                      (looser - sample)[1] * metricWeights[
                                                                                          1]

        deltaWinner = [sample[0] - winner[0], sample[1] - winner[1]]
        facWin = looserDist / ((winnerDist + looserDist) * (winnerDist + looserDist));
        factorWinner = learnrate * facWin
        winner[0] = winner[0] + factorWinner * deltaWinner[0] * metricWeights[0]
        winner[1] = winner[1] + factorWinner * deltaWinner[1] * metricWeights[1]

        deltaLooser = [sample[0] - looser[0], sample[1] - looser[1]]
        facLos = winnerDist / ((winnerDist + looserDist) * (winnerDist + looserDist))
        factorLooser = learnrate * facLos
        looser[0] = looser[0] - factorLooser * deltaLooser[0] * metricWeights[0]
        looser[1] = looser[1] - factorLooser * deltaLooser[1] * metricWeights[1]

        # print 'winner ' + str(winner)
        #print 'looser ' + str(looser)
        #print 'learnrate ' + str(learnrate)
        #print 'winnerDist' + str(winnerDist)
        #print 'looserDist' + str(looserDist)
        #print 'deltaWinner[0]' + str(deltaWinner[0])
        #print 'deltaLooser[0]' + str(deltaLooser[0])
        #print 'deltaWinner[1]' + str(deltaWinner[1])
        #print 'deltaLooser[1]' + str(deltaLooser[1])

        updateMetric = learnrateMetricWeights * (
        facWin * deltaWinner[0] * deltaWinner[0] - facLos * deltaLooser[0] * deltaLooser[0])
        metricWeights[0] -= updateMetric;
        metricWeights[0] = max(metricWeights[0], 0)
        updateMetric = learnrateMetricWeights * (
        facWin * deltaWinner[1] * deltaWinner[1] - facLos * deltaLooser[1] * deltaLooser[1])
        metricWeights[1] -= updateMetric;
        metricWeights[1] = max(metricWeights[1], 0)
        metricWeightsSum = metricWeights[0] + metricWeights[1];

        metricWeights[0] /= metricWeightsSum;
        metricWeights[1] /= metricWeightsSum;
        #print 'metricWeights' + str(metricWeights)

    @staticmethod
    def genGRLVQ_LRFALSEValues():
        initValues = LVQSimple.initValues()
        dataClass1 = initValues[0]
        dataClass2 = initValues[1]
        proto1 = initValues[2]
        proto2 = initValues[3]
        startLearnRate = initValues[4]
        startMetricLearnRate = initValues[5]
        prototypes = proto1
        prototypes = numpy.vstack([prototypes, proto2])
        prototypes = numpy.round(prototypes, 3)

        metricWeights = [0.5, 0.5];

        maxSamples = len(dataClass1) * 2
        sampleCount = 0
        learnrate = startLearnRate
        metricWeightsLearnrate = startMetricLearnRate
        for sample in range(len(dataClass1) - 1):
            LVQSimple.trainGRLVQ(proto1, 0, proto2, 1, learnrate, metricWeightsLearnrate, dataClass1[sample + 1, :], 0, metricWeights)
            sampleCount += 1
            learnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startLearnRate)
            metricWeightsLearnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startMetricLearnRate)
            LVQSimple.trainGRLVQ(proto1, 0, proto2, 1, learnrate, metricWeightsLearnrate, dataClass2[sample + 1, :], 1, metricWeights)
            sampleCount += 1
            learnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startLearnRate)
            metricWeightsLearnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startMetricLearnRate)

        prototypes = proto1
        prototypes = numpy.vstack([prototypes, proto2])
        prototypes = numpy.round(prototypes, 3)
        metricWeights = numpy.round(metricWeights, 3)
        return (prototypes, metricWeights)

    @staticmethod
    def trainGMLVQ(proto1, proto1Label, proto2, proto2Label, learnrate, learnrateMetricWeights, sample, sampleLabel,
                   metricWeights, omega):
        # print metricWeights
        #print omega

        values = LVQSimple.getWinnerLooser(proto1, proto1Label, proto2, proto2Label, sampleLabel)
        winner = values[0]
        looser = values[1]

        winnerDist = (metricWeights[0][0] * (winner - sample)[0] + metricWeights[0][1] * (winner - sample)[1]) * \
                     (winner - sample)[0] + (metricWeights[1][0] * (winner - sample)[0] + metricWeights[1][1] *
                                             (winner - sample)[1]) * (winner - sample)[1]

        looserDist = (metricWeights[0][0] * (looser - sample)[0] + metricWeights[0][1] * (looser - sample)[1]) * \
                     (looser - sample)[0] + (metricWeights[1][0] * (looser - sample)[0] + metricWeights[1][1] *
                                             (looser - sample)[1]) * (looser - sample)[1]

        #maybe its omega*omegaTrans...
        omegaSquare = [
            [omega[0][0] * omega[0][0] + omega[0][1] * omega[1][0], omega[0][0] * omega[0][1] + omega[0][1] * omega[1][1]],
            [omega[1][0] * omega[0][0] + omega[1][1] * omega[1][0], omega[1][0] * omega[0][1] + omega[1][1] * omega[1][1]]]

        #deltaWinner = [sample[0] - winner[0], sample[1] - winner[1]]
        deltaWinner = sample - winner
        facWin = looserDist / ((winnerDist + looserDist) * (winnerDist + looserDist));
        factorWinner = learnrate * facWin
        winner[0] = winner[0] + factorWinner * (omegaSquare[0][0] * deltaWinner[0] + omegaSquare[0][1] * deltaWinner[1])
        winner[1] = winner[1] + factorWinner * (omegaSquare[1][0] * deltaWinner[0] + omegaSquare[1][1] * deltaWinner[1])

        #deltaLooser = [sample[0] - looser[0], sample[1] - looser[1]]
        deltaLooser = sample - looser
        facLos = winnerDist / ((winnerDist + looserDist) * (winnerDist + looserDist))
        factorLooser = learnrate * facLos
        looser[0] = looser[0] - factorLooser * (omegaSquare[0][0] * deltaLooser[0] + omegaSquare[0][1] * deltaLooser[1])
        looser[1] = looser[1] - factorLooser * (omegaSquare[1][0] * deltaLooser[0] + omegaSquare[1][1] * deltaLooser[1])





        #print 'winner ' + str(winner)
        #print 'looser ' + str(looser)
        #print 'learnrate ' + str(learnrate)
        #print 'winnerDist ' + str(winnerDist)
        #print 'looserDist ' + str(looserDist)
        #print 'deltaWinner[0]' + str(deltaWinner[0])
        #print 'deltaLooser[0]' + str(deltaLooser[0])
        #print 'deltaWinner[1]' + str(deltaWinner[1])
        #print 'deltaLooser[1]' + str(deltaLooser[1])

        omegaDeltaWinner = np.dot(omega,deltaWinner)

        omegaDeltaLooser = np.dot(omega, deltaLooser)

        omega[0][0] -= learnrateMetricWeights * (
        facWin * (omegaDeltaWinner[0] * deltaWinner[0] + omegaDeltaWinner[0] * deltaWinner[0]) - facLos * (
        omegaDeltaLooser[0] * deltaLooser[0] + omegaDeltaLooser[0] * deltaLooser[0]))

        omega[0][1] -= learnrateMetricWeights * (
        facWin * (omegaDeltaWinner[1] * deltaWinner[0] + omegaDeltaWinner[0] * deltaWinner[1]) - facLos * (
        omegaDeltaLooser[1] * deltaLooser[0] + omegaDeltaLooser[0] * deltaLooser[1]))

        omega[1][0] -= learnrateMetricWeights * (
        facWin * (omegaDeltaWinner[0] * deltaWinner[1] + omegaDeltaWinner[1] * deltaWinner[0]) - facLos * (
        omegaDeltaLooser[0] * deltaLooser[1] + omegaDeltaLooser[1] * deltaLooser[0]))

        omega[1][1] -= learnrateMetricWeights * (
        facWin * (omegaDeltaWinner[1] * deltaWinner[1] + omegaDeltaWinner[1] * deltaWinner[1]) - facLos * (
        omegaDeltaLooser[1] * deltaLooser[1] + omegaDeltaLooser[1] * deltaLooser[1]))


        normTerm = math.sqrt((omega[0][0] * omega[0][0]) + (omega[0][1] * omega[0][1]) + (omega[1][0] * omega[1][0]) + (
        omega[1][1] * omega[1][1]))
        #snormTerm = (omega[0][0] * omega[0][0]) + (omega[0][1] * omega[0][1]) + (omega[1][0] * omega[1][0]) + (omega[1][1] * omega[1][1])
        '''print 'winnerDist ' + str(winnerDist)
        print 'looserDist ' + str(looserDist)

        print 'deltaWinner' +str(deltaWinner)
        print 'deltaLooser' +str(deltaLooser)
        print 'facWin' +str(facWin)
        print 'facLos' +str(facLos)
        print 'omega before norm\n',omega'''
        omega[0][0] /= normTerm
        omega[0][1] /= normTerm
        omega[1][0] /= normTerm
        omega[1][1] /= normTerm

        metricWeights[0][0] = omega[0][0] * omega[0][0] + omega[0][1] * omega[0][1];
        metricWeights[0][1] = omega[0][0] * omega[1][0] + omega[0][1] * omega[1][1];
        metricWeights[1][0] = omega[1][0] * omega[0][0] + omega[1][1] * omega[0][1];
        metricWeights[1][1] = omega[1][0] * omega[1][0] + omega[1][1] * omega[1][1];

        '''print 'omega after norm\n',omega
        print 'metricWeights \n' + str(metricWeights)'''

    @staticmethod
    def genGMLVQ_LRFALSEValues():
        initValues = LVQSimple.initValues()
        dataClass1 = initValues[0]
        dataClass2 = initValues[1]

        proto1 = initValues[2]
        proto2 = initValues[3]
        startLearnRate = initValues[4]
        startMetricLearnRate = initValues[5]

        metricWeights = np.array([[0.5, 0], [0, 0.5]]);
        omega = np.array([[math.sqrt(1.0 / 2.0), 0], [0, math.sqrt(1.0 / 2)]]);

        maxSamples = len(dataClass1) * 2
        sampleCount = 0
        learnrate = startLearnRate
        metricWeightsLearnrate = startMetricLearnRate
        for sample in range(dataClass1.shape[0]-1):
            # for sample in range(1):
            LVQSimple.trainGMLVQ(proto1, 0, proto2, 1, learnrate, metricWeightsLearnrate, dataClass1[sample + 1, :], 0, metricWeights,
                       omega)
            sampleCount += 1
            learnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startLearnRate)
            metricWeightsLearnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startMetricLearnRate)
            LVQSimple.trainGMLVQ(proto1, 0, proto2, 1, learnrate, metricWeightsLearnrate, dataClass2[sample + 1, :], 1, metricWeights,
                       omega)
            sampleCount += 1
            learnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startLearnRate)

            metricWeightsLearnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startMetricLearnRate)

        prototypes = proto1
        prototypes = numpy.vstack([prototypes, proto2])
        prototypes = numpy.round(prototypes, 3)
        metricWeights = numpy.round(metricWeights, 3)
        return (prototypes, metricWeights)

    @staticmethod
    def genGMLVQValues(LVQFactory):
        initValues = LVQSimple.initValues()
        dataClass1 = initValues[0]
        dataClass2 = initValues[1]
        startLearnRate = initValues[4]
        startMetricLearnRate = initValues[5]
        maxSamples = len(dataClass1) * 2
        gmlvq = LVQFactory.getLVQClassifier(dataClass1.shape[1], learnRateInitial=startLearnRate, learnRateAnnealingSteps=maxSamples, netType='GMLVQ', metricLearnRate=startMetricLearnRate)

        gmlvq.addPrototype(initValues[2], 0)
        gmlvq.addPrototype(initValues[3], 1)
        sampleCount = 0
        for sample in range(dataClass1.shape[0]-1):
            # for sample in range(1):
            gmlvq.train(dataClass1[sample + 1, :], 0)
            sampleCount += 1
            metricWeightsLearnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startMetricLearnRate)
            gmlvq.metricLearnRate = metricWeightsLearnrate
            gmlvq.train(dataClass2[sample + 1, :], 1)
            sampleCount += 1
            metricWeightsLearnrate = LVQSimple.getNewLearnRate(sampleCount, maxSamples, startMetricLearnRate)
            gmlvq.metricLearnRate = metricWeightsLearnrate
        prototypes = gmlvq.prototypes
        prototypes = numpy.round(prototypes, 3)
        metricWeights = numpy.round(gmlvq.metricWeights, 3)

        return (prototypes, metricWeights)

    @staticmethod
    def compareGLVQValues2(LVQFactory):
        values = LVQSimple.genGLVQ_OwnImplLRFALSEValues2(LVQFactory)
        ownValues = LVQSimple.genGLVQ_LRFALSEValues()
        if ownValues.tolist() != values.tolist():
            logging.warning(' ***Prototypes are equal=' + str(ownValues.tolist() == values.tolist()))
            logging.warning('py ' + str(ownValues.tolist()))
            logging.warning('new  ' + str(values.tolist()))
        return ownValues.tolist() == values.tolist()

    @staticmethod
    def compareGLVQValues():
        values = generateNetworkValues.genNetworkValues(libpythoninterface.NETTYPE_GLVQ, False)
        ownValues = LVQSimple.genGLVQ_LRFALSEValues()
        print ' ***Prototypes are equal=' + str(ownValues.tolist() == values[0].tolist());
        print 'py ' + str(ownValues.tolist())
        print 'c  ' + str(values[0].tolist())

    @staticmethod
    def compareGRLVQValues():
        values = generateNetworkValues.genNetworkValues(libpythoninterface.NETTYPE_GRLVQ, False)
        ownValues = LVQSimple.genGRLVQ_LRFALSEValues()
        print ' ***Prototypes are equal=' + str(ownValues[0].tolist() == values[0].tolist());
        print 'py ' + str(ownValues[0].tolist())
        print 'c  ' + str(values[0].tolist())
        print ' ***Lambdas are equal=' + str(ownValues[1].tolist() == values[1].tolist());
        print 'py ' + str(ownValues[1].tolist())
        print 'c  ' + str(values[1].tolist())

    @staticmethod
    def compareGMLVQValues():
        values = generateNetworkValues.genNetworkValues(libpythoninterface.NETTYPE_GMLVQ, False)
        ownValues = LVQSimple.genGMLVQ_LRFALSEValues()

        print ' ***Prototypes are equal=' + str(ownValues[0].tolist() == values[0].tolist())
        print 'py ' + str(ownValues[0].tolist())
        print 'c  ' + str(values[0].tolist())
        print ' ***Lambdas are equal=' + str(ownValues[1].tolist() == values[1].tolist())
        print 'py ' + str(ownValues[1].tolist())
        print 'c  ' + str(values[1].tolist())

    @staticmethod
    def compareGMLVQOwnImplValues(LVQFactory):
        values = LVQSimple.genGMLVQValues(LVQFactory)
        ownValues = LVQSimple.genGMLVQ_LRFALSEValues()
        if ownValues[0].tolist() != values[0].tolist():
            logging.warning(' ***Prototypes are equal=' + str(ownValues[0].tolist() == values[0].tolist()))
            logging.warning('py ' + str(ownValues[0].tolist()))
            logging.warning('new  ' + str(values[0].tolist()))
        if ownValues[1].tolist() != values[1].tolist():
            logging.warning(' ***Lambdas are equal=' + str(ownValues[1].tolist() == values[1].tolist()))
            logging.warning('py ' + str(ownValues[1].tolist()))
            logging.warning('new  ' + str(values[1].tolist()))
        return ownValues[0].tolist() == values[0].tolist() and ownValues[1].tolist() == values[1].tolist()


if __name__ == '__main__':
    LVQSimple.compareGMLVQOwnImplValues()
    LVQSimple.compareGLVQValues2()
    '''compareGLVQValues()
    compareGRLVQValues()
    compareGMLVQValues()'''