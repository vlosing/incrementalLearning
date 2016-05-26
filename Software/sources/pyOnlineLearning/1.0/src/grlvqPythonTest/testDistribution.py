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

    datasamples = 1000

    mean = (1, 2)
    cov = [[1., -0.8],
           [-0.8, 1]]
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
    cov = [[0.6, -.5],
           [-.5, 0.6]]
    x4 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.hold(True)
    ax.plot(x1[:, 0], x1[:, 1], '1' + colors[0], x2[:, 0], x2[:, 1], '2' + colors[1], x3[:, 0], x3[:, 1],
            '3' + colors[2], x4[:, 0], x4[:, 1], '4' + colors[3])
    ax.set_title('Data Input')
    dataRange = (-3, 8)

    # Create network:
    net1 = "net1"
    nettype = libpythoninterface.NETTYPE_GLVQ
    learnrate_per_node = libpythoninterface.NETWORK_LEARNRATE_PROTOTYPE
    num_of_classes = 4
    prototypes_per_class = 1
    trainsteps = 4 * datasamples
    do_node_random_init = True
    threads_per_nodes = 1
    dimensionality = 2
    libpythoninterface.create_network(net1, dimensionality, nettype, learnrate_per_node, num_of_classes,
                                      prototypes_per_class, trainsteps, do_node_random_init, threads_per_nodes)

    libpythoninterface.set_learnrate_start_network(net1, 1.0)
    libpythoninterface.set_learnrate_metricWeights_start_network(net1, 0.01)

    for sample in range(datasamples):
        libpythoninterface.train_network(net1, x1[sample, :], [0])
        libpythoninterface.train_network(net1, x2[sample, :], [1])
        libpythoninterface.train_network(net1, x3[sample, :], [2])
        libpythoninterface.train_network(net1, x4[sample, :], [3])

    for c in range(num_of_classes):
        for p in range(prototypes_per_class):
            i = c * prototypes_per_class + p
            pos = libpythoninterface.get_weights_network(net1, i)
            print pos

            lambda_node = libpythoninterface.get_metricWeights_network(net1, i)
            ax.plot(pos[0], pos[1], "D" + colors[c])

    plt.xlim([dataRange[0], dataRange[1]]);
    plt.ylim([dataRange[0], dataRange[1]]);
    #elipse1.set_clip_box(ax.bbox)
    #ax.add_artist(elipse1)
    #    lambda1=libpythoninterface.get_eigenvectors_network(net1,0)
    #    lambda2=libpythoninterface.get_eigenvectors_network(net1,1)
    #    lambda3=libpythoninterface.get_eigenvectors_network(net1,2)
    #    lambda4=libpythoninterface.get_eigenvectors_network(net1,3)


    #    print "lambdas: ",  lambda1
    #    print "lambdas: ",  lambda2
    #    print "lambdas: ",  lambda3
    #    print "lambdas: ",  lambda4

    #    if (len(lambda1.shape)==2):
    #        fig = plt.figure()
    #ax = fig.add_subplot(111, aspect='equal')
    #ax.hold(True)
    #ax.plot(x1[:,0]*lambda1[0,0]+x1[:,1]*lambda1[0,1], x1[:,0]*lambda1[1,0]+x1[:,1]*lambda1[1,1], '1'+colors[0],
    #        x2[:,0]*lambda2[0,0]+x1[:,1]*lambda2[0,1], x2[:,0]*lambda2[1,0]+x2[:,1]*lambda2[1,1], '2'+colors[1],
    #        x3[:,0]*lambda3[0,0]+x1[:,1]*lambda3[0,1], x3[:,0]*lambda3[1,0]+x3[:,1]*lambda3[1,1], '3'+colors[2],
    #        x4[:,0]*lambda4[0,0]+x1[:,1]*lambda4[0,1], x4[:,0]*lambda4[1,0]+x4[:,1]*lambda4[1,1], '4'+colors[3])
    #ax.set_title('Data transformed')


    #for p in range(prototypes_per_class):
    #    i=0*prototypes_per_class+p
    #    pos = libpythoninterface.get_weights_network(net1, i)
    #    ax.plot(pos[0]*lambda1[0,0]+pos[1]*lambda1[0,1],pos[0]*lambda1[1,0]+pos[1]*lambda1[1,1], "D"+colors[0])

    #    i=1*prototypes_per_class+p
    #    pos = libpythoninterface.get_weights_network(net1, i)
    #    ax.plot(pos[0]*lambda2[0,0]+pos[1]*lambda2[0,1],pos[0]*lambda2[1,0]+pos[1]*lambda2[1,1], "D"+colors[1])

    #    i=2*prototypes_per_class+p
    #    pos = libpythoninterface.get_weights_network(net1, i)
    #    ax.plot(pos[0]*lambda3[0,0]+pos[1]*lambda3[0,1],pos[0]*lambda3[1,0]+pos[1]*lambda3[1,1], "D"+colors[2])

    #    i=3*prototypes_per_class+p
    #    pos = libpythoninterface.get_weights_network(net1, i)
    #    ax.plot(pos[0]*lambda4[0,0]+pos[1]*lambda4[0,1],pos[0]*lambda4[1,0]+pos[1]*lambda4[1,1], "D"+colors[3])



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



            #         for row in range(yVal):
            #            for col in range(xVal):
            #                tdata = numpy.array(col, row)
            #                [result, index] = libpythoninterface.process_network(net1, tdata)
            #                img[row,col,:]=matplotlib.colors.colorConverter.to_rgb(colors[result]);

            #create voronoi cells:

            #        imrange=(8., 8.)
            #        offset=(0,0)
            #        img=numpy.ones((imgsize[1], imgsize[0],3))
            #        for row in range(imgsize[1]):
            #            for col in range(imgsize[0]):
            #                tdata = numpy.array(((col+offset[0])*imrange[0]/imgsize[0], (row+offset[1])*imrange[1]/imgsize[1]))
            #                [result, index] = libpythoninterface.process_network(net1, tdata)
            #                img[row,col,:]=matplotlib.colors.colorConverter.to_rgb(colors[result]);
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





    #-------INC TEST -------------#

    datasamples = 3000

    mean = (1, 2)
    cov = [[1., -0.8],
           [-0.8, 1]]
    x1 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    mean = (5, 5)
    cov = [[0.5, 0.3],
           [0.3, 0.1]]
    cov = [[9.9, 0.0],
           [0.0, 0.1]]
    x2 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    mean = (6, 2)
    cov = [[0.3, 0],
           [0, 0.1]]
    cov = [[9.9, 0],
           [0, 0.1]]
    x3 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    mean = (0, 0)
    cov = [[0.6, -.5],
           [-.5, 0.6]]
    x4 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    x2 = numpy.vstack((x2, x4))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.hold(True)
    ax.plot(x1[:, 0], x1[:, 1], '1' + colors[0], x2[:, 0], x2[:, 1], '2' + colors[1], x3[:, 0], x3[:, 1],
            '3' + colors[2], zorder=1)
    ax.set_title('Data Input')

    ax.hold(True)






    #Create network:
    net1 = "net2"
    nettype = libpythoninterface.NETTYPE_LGMLVQ
    learnrate_per_node = libpythoninterface.NETWORK_LEARNRATE
    num_of_classes = 3
    prototypes_per_class = 1
    trainsteps = 4 * datasamples
    do_node_random_init = True
    threads_per_nodes = 1
    dimensionality = 2
    libpythoninterface.create_network(net1, dimensionality, nettype, learnrate_per_node, num_of_classes,
                                      prototypes_per_class, trainsteps, do_node_random_init, threads_per_nodes)

    for sample in range(datasamples):
        failinc = True
        g_max = 200
        libpythoninterface.incremental_train_network(net1, x1[sample, :], [0], failinc, g_max,
                                                     libpythoninterface.NETMODE_BOTH, False)
        libpythoninterface.incremental_train_network(net1, x2[sample, :], [1], failinc, g_max,
                                                     libpythoninterface.NETMODE_BOTH, False)
        libpythoninterface.incremental_train_network(net1, x3[sample, :], [2], failinc, g_max,
                                                     libpythoninterface.NETMODE_BOTH, False)
        libpythoninterface.incremental_train_network(net1, x2[datasamples + sample, :], [1], failinc, g_max,
                                                     libpythoninterface.NETMODE_BOTH, False)

    protos = libpythoninterface.get_numofprototypes_network(net1)

    for p in range(protos):
        pos = libpythoninterface.get_weights_network(net1, p)
        labelnr = libpythoninterface.get_prototypelabel_network(net1, p)
        ax.plot(pos[0], pos[1], "D" + colors[labelnr], zorder=10)
        lambdas = libpythoninterface.get_eigenvectors_network(net1, p)
        print "Proto" + str(p) + "_lambda:", lambdas
        ax.arrow(pos[0], pos[1], lambdas[0, 0], lambdas[0, 1], shape='full', lw=3, length_includes_head=True,
                 head_width=.01, zorder=9)
        ax.arrow(pos[0], pos[1], lambdas[1, 0], lambdas[1, 1], shape='full', lw=3, length_includes_head=True,
                 head_width=.01, zorder=9)



    #create voronoi cells:
    imgsize = (300, 300)
    imrange = (9., 9.)
    offset = (-50, -50)
    img = numpy.ones((imgsize[1], imgsize[0], 3))
    for row in range(imgsize[1]):
        for col in range(imgsize[0]):
            tdata = numpy.array(
                ((col + offset[0]) * imrange[0] / imgsize[0], (row + offset[1]) * imrange[1] / imgsize[1]))
            [result, index] = libpythoninterface.process_network(net1, tdata)
            img[row, col, :] = matplotlib.colors.colorConverter.to_rgb(colors[result]);

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(img)
    ax.hold(True)
    for p in range(protos):
        pos = libpythoninterface.get_weights_network(net1, p)
        labelnr = libpythoninterface.get_prototypelabel_network(net1, p)
        ax.plot((pos[0] / imrange[0]) * imgsize[0] - offset[0], (pos[1] / imrange[1]) * imgsize[1] - offset[1],
                "D" + colors[labelnr])

    plt.ylim([0, imgsize[0]])
    plt.xlim([0, imgsize[1]])
    ax.set_axis_off()

    plt.show()

