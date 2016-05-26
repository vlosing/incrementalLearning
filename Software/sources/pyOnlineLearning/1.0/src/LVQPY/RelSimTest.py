__author__ = 'vlosing'

import numpy
import matplotlib.pyplot as plt

from DataGeneration import toyDataSets
from LVQPY import GLVQPlot
from LVQPY import GLVQ
from Base import Paths
from Classification.StatisticsLogger import StatisticsLogger


if __name__ == '__main__':
    GLVQS = {}
    criterias = ['Center', 'Border']
    criteria = criterias[0]
    statistics = StatisticsLogger(False, False)

    colors = ['black', 'white']
    x1 = -7
    x2 = 7
    y = 0
    samples, labels = toyDataSets.getRelSimSet(x1, y, x2, y, (7, 7))
    statistics.newIteration(None, None, samples, labels)

    glvq = GLVQ.GLVQ(0, 0, 0, 2)
    proto1 = numpy.array([x1, y])
    proto2 = numpy.array([x2, y])
    glvq.addPrototype(proto1, 0, True)
    glvq.addPrototype(proto2, 1, True)
    GLVQLst = []
    GLVQLst.append(glvq)
    GLVQS[criteria] = GLVQLst
    statistics.newCfgIteration(criteria)

    fig = GLVQPlot.plotRelSim(glvq, glvq.getProtos(), glvq.getProtoLabels(), samples, labels, colors, '')
    fig.savefig(Paths.StatisticsClassificationDir() + 'RelSimCenter.png', bbox_inches='tight')

    criteria = criterias[1]
    x1 = -1
    x2 = 1
    glvq = GLVQ.GLVQ(0, 0, 0, 2)
    proto1 = numpy.array([x1, y])
    proto2 = numpy.array([x2, y])
    glvq.addPrototype(proto1, 0, True)
    glvq.addPrototype(proto2, 1, True)
    GLVQLst = []
    GLVQLst.append(glvq)
    GLVQS[criteria] = GLVQLst
    statistics.newCfgIteration(criteria)

    fig = GLVQPlot.plotRelSim(glvq, glvq.getProtos(), glvq.getProtoLabels(), samples, labels, colors, '')
    fig.savefig(Paths.StatisticsClassificationDir() + 'RelSimBorder.png', bbox_inches='tight')
    statistics.finishIteration()
    statistics.initRejectionData(GLVQS)

    statistics.plotRejections(criterias, Paths.StatisticsClassificationDir(), '', yLim=[0.95, 1])

    plt.show()