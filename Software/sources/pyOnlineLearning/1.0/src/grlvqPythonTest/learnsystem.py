'''
Created on Feb 10, 2012

@author: jqueisse
'''
import numpy
import scipy.misc
import random
import matplotlib
import matplotlib.pyplot as plt


if __name__ == '__main__':
    datasamples = 3000
    colors = ["b", "g", "r", "c"]

    mean = (0, 0)
    cov = [[2., -1.8],
           [-1.8, 2]]
    x1 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    mean = (5, 5)
    cov = [[0.5, 0.3],
           [0.3, 0.1]]
    x2 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    mean = (6, 2)
    cov = [[0.3, 0],
           [0, 0.1]]
    x3 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    mean = (0, 0)
    cov = [[2, 1.8],
           [1.8, 2]]
    x4 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.hold(True)
    ax.plot(x1[:, 0], x1[:, 1], '1' + colors[0], x2[:, 0], x2[:, 1], '2' + colors[0], x3[:, 0], x3[:, 1],
            '3' + colors[2], x4[:, 0], x4[:, 1], '4' + colors[3])
    ax.set_title('Data Input')
    plt.show()