'''
Created on Feb 1, 2012

@author: jqueisse
'''

import numpy
import scipy.misc
import random
import matplotlib
import matplotlib.pyplot as plt

import libpythoninterface


if __name__ == '__main__':


    colors = ["b", "g", "r", "c"]

    datasamples = 100

    mean = (-2, 2)
    cov = [[1., 0],
           [0, 1]]
    x1 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    mean = (2, 2)
    cov = [[1, 0],
           [0, 1]]
    x2 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.hold(True)
    ax.plot(x1[:, 0], x1[:, 1], '1' + colors[0], x2[:, 0], x2[:, 1], '2' + colors[1])
    ax.set_title('Data Input')
    dataRange = (-8, 8)

    # Create network:
    net1 = "net1"
    nettype = libpythoninterface.NETTYPE_LGRLVQ
    learnrate_per_node = False
    num_of_classes = 2
    prototypes_per_class = 1
    trainsteps = 4 * datasamples
    do_node_random_init = True
    threads_per_nodes = 1
    dimensionality = 2
    libpythoninterface.create_network(net1, dimensionality, nettype, learnrate_per_node, num_of_classes,
                                      prototypes_per_class, trainsteps, do_node_random_init, threads_per_nodes)
    print 'before training'
    pos = libpythoninterface.get_weights_network(net1, 1)
    print pos
    pos = libpythoninterface.get_weights_network(net1, 2)
    print pos

    libpythoninterface.set_epsilon_start_network(net1, 1.0)
    libpythoninterface.set_epsilon_lambda_start_network(net1, 0.01)

    for sample in range(datasamples):
        libpythoninterface.train_network(net1, x1[sample, :], [0])
        libpythoninterface.train_network(net1, x2[sample, :], [1])

    print 'after training'
    for c in range(num_of_classes):
        for p in range(prototypes_per_class):
            i = c * prototypes_per_class + p
            pos = libpythoninterface.get_weights_network(net1, i)
            print pos

            lambda_node = libpythoninterface.get_lambdas_network(net1, i)
            ax.plot(pos[0], pos[1], "D" + colors[c])

    plt.xlim([dataRange[0], dataRange[1]]);
    plt.ylim([dataRange[0], dataRange[1]]);

    #elipse1.set_clip_box(ax.bbox)
    #ax.add_artist(elipse1)
    #    lambda1=libpythoninterface.get_eigenvectors_network(net1,0)
    #    lambda2=libpythoninterface.get_eigenvectors_network(net1,1)


    #    print "lambdas: ",  lambda1
    #    print "lambdas: ",  lambda2

    #    if (len(lambda1.shape)==2):
    #        fig = plt.figure()
    #ax = fig.add_subplot(111, aspect='equal')
    #ax.hold(True)
    #ax.plot(x1[:,0]*lambda1[0,0]+x1[:,1]*lambda1[0,1], x1[:,0]*lambda1[1,0]+x1[:,1]*lambda1[1,1], '1'+colors[0],
    #        x2[:,0]*lambda2[0,0]+x1[:,1]*lambda2[0,1], x2[:,0]*lambda2[1,0]+x2[:,1]*lambda2[1,1], '2'+colors[1])
    #ax.set_title('Data transformed')


    #for p in range(prototypes_per_class):
    #    i=0*prototypes_per_class+p
    #    pos = libpythoninterface.get_weights_network(net1, i)
    #    ax.plot(pos[0]*lambda1[0,0]+pos[1]*lambda1[0,1],pos[0]*lambda1[1,0]+pos[1]*lambda1[1,1], "D"+colors[0])

    #    i=1*prototypes_per_class+p
    #    pos = libpythoninterface.get_weights_network(net1, i)
    #    ax.plot(pos[0]*lambda2[0,0]+pos[1]*lambda2[0,1],pos[0]*lambda2[1,0]+pos[1]*lambda2[1,1], "D"+colors[1])

    if (0 == 0):
        imgsize = (300, 300)
        xVals = numpy.linspace(dataRange[0], dataRange[1], imgsize[0]);
        yVals = numpy.linspace(dataRange[0], dataRange[1], imgsize[1]);
        img = numpy.ones((imgsize[1], imgsize[0], 3))
        for col in range(imgsize[0]):
            for row in range(imgsize[1]):
                tdata = numpy.array((xVals[col], yVals[row]))
                [result, index] = libpythoninterface.process_network(net1, tdata)
                img[row, col, :] = matplotlib.colors.colorConverter.to_rgb(colors[result]);

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_title('VoronoiCells')
        ax.imshow(img)
        ax.hold(True)
        #       for c in range(num_of_classes):
        #            for p in range(prototypes_per_class):
        #                i=c*prototypes_per_class+p
        #                pos = libpythoninterface.get_weights_network(net1, i)
        #                ax.plot((pos[0]/imrange[0])*imgsize[0] -offset[0],(pos[1]/imrange[1])*imgsize[1] -offset[1], "D"+colors[c])


        plt.ylim([0, imgsize[0]])
        plt.xlim([0, imgsize[1]])
        ax.set_axis_off()

        plt.show()





