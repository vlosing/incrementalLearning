__author__ = 'vlosing'

import copy

import numpy
import matplotlib.pyplot as plt

from DataGeneration import toyDataSets
from LVQPY import GLVQPlot
from LVQPY.GLVQ import GLVQ


def demonstrateLearning(glvq, samples, labels, fig, subplot):
    gauss3 = toyDataSets.createNormDistData(3, mean=(7, -8), var=(1, 1))
    for sample in gauss3:
        GLVQPlot.plotSamples(subplot, sample, 2, colors)
        fig.canvas.draw()
        raw_input('pause : press any key ...')
        glvq.train(sample, 2)
        fig.clf()
        samples = numpy.vstack([samples, sample])
        labels = numpy.append(labels, 2)
        fig, subplot = GLVQPlot.plotAll(glvq, glvq.prototypes(), glvq.prototypeLabels(), samples, labels, colors, '',
                                        fig=fig)
        fig.canvas.draw()
        raw_input('pause : press any key ...')


def demonstrateInsertions(glvq, samples, labels, fig):
    gauss3 = toyDataSets.createNormDistData(30, mean=(6, -7), var=(1., 1.))
    gauss4 = toyDataSets.createNormDistData(30, mean=(10, -4), var=(1., 1.))
    gauss = numpy.vstack([gauss3, gauss4])
    glvq.learnRateInitial = 0
    for sample in gauss:
        glvq.train(sample, 2)
        samples = numpy.vstack([samples, sample])
        labels = numpy.append(labels, 2)
    fig.clf()
    fig, subplot = GLVQPlot.plotAll(glvq, glvq.prototypes(), glvq.prototypeLabels(), samples, labels, colors, '', fig=fig)
    fig.canvas.draw()
    raw_input('pause : press any key ...')
    glvqClosest = copy.deepcopy(glvq)

    glvqClosest._insertionStrategyClosest(glvqClosest.incClassMatrix)
    fig.clf()
    fig, subplot = GLVQPlot.plotAll(glvqClosest, glvqClosest.prototypes(), glvqClosest.prototypeLabels(), samples, labels,
                                    colors, '', fig=fig)
    fig.canvas.draw()
    raw_input('pause : press any key ...')

    glvqClustering = copy.deepcopy(glvq)
    glvqClustering._insertionStrategyClustering(glvqClustering.incClassMatrix, 'kMeans')
    fig.clf()
    fig, subplot = GLVQPlot.plotAll(glvqClustering, glvqClustering.prototypes(), glvqClustering.prototypeLabels(),
                                    samples, labels, colors, '', fig=fig)
    fig.canvas.draw()
    raw_input('pause : press any key ...')

    glvqClustering = copy.deepcopy(glvq)
    glvqClustering.insertionStrategyVoronoi(glvqClustering.incClassMatrix)
    fig.clf()
    fig, subplot = GLVQPlot.plotAll(glvqClustering, glvqClustering.prototypes(), glvqClustering.prototypeLabels(),
                                    samples, labels, colors, '', fig=fig)
    fig.canvas.draw()
    raw_input('pause : press any key ...')

    glvqClustering = copy.deepcopy(glvq)
    glvqClustering._insertionStrategySamplingCost(glvqClustering.incClassMatrix, 100)
    fig.clf()
    fig, subplot = GLVQPlot.plotAll(glvqClustering, glvqClustering.prototypes(), glvqClustering.prototypeLabels(),
                                    samples, labels, colors, '', fig=fig)
    fig.canvas.draw()
    raw_input('pause : press any key ...')


if __name__ == '__main__':
    glvq = GLVQ(2, learnRateInitial=30)
    x1 = -7
    x2 = 7
    x3 = 0
    y = 0
    y3 = -7
    proto1 = numpy.array([x1, y])
    proto2 = numpy.array([x2, y])
    proto3 = numpy.array([x3, y3])

    samples, labels = toyDataSets.getGLVQTest(x1, y, x2, y, x3, y3, (1, 1))
    glvq.addPrototype(proto1, 0)
    glvq.addPrototype(proto2, 1)
    glvq.addPrototype(proto3, 2)
    colors = ['red', 'blue', 'green']

    plt.ion()
    fig, subplot = GLVQPlot.plotAll(glvq, glvq.getProtos(), glvq.getProtoLabels(), samples, labels, colors, '')
    fig.canvas.draw()
    raw_input('pause : press any key ...')

    '''sample = numpy.array([7,-8])
    GLVQPlot.plotSample(subplot, sample, 2, colors)
    fig.canvas.draw()
    raw_input('pause : press any key ...')'''
    # demonstrateLearning(glvq, samples, labels, fig, subplot)
    demonstrateInsertions(glvq, samples, labels, fig, subplot)



