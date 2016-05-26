'''
Created on Feb 29, 2012

This implements the online-offline classifier system
It uses one of two different artificial data sets.
Furthermore a set of methods for the visualization of the training
are included.


Goto __main__ for more information

@author: jqueisse
'''

import pickle
from copy import deepcopy

import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt
import numpy

# from svmutil import *

import random

import libpythoninterface as libgxlvq
from matplotlib.patches import Rectangle, Ellipse, Circle
from mpl_toolkits.axes_grid.anchored_artists import AnchoredDrawingArea

outfolder = "/hri/localdisk/jqueisse/tmpgen/1/"


def getRotationMatrix(rotation):
    sinrot = numpy.sin(rotation)
    cosrot = numpy.cos(rotation)
    return numpy.array([[cosrot, -sinrot], [sinrot, cosrot]])


def createNormDistData(samples, mean=(0, 0), var=(1, 1), rotation=0):
    cov = numpy.array([[var[0], -0],
                       [-0, var[1]]])
    if not rotation == 0:
        points = numpy.random.multivariate_normal((0, 0), cov, (samples))
        return numpy.dot(getRotationMatrix(rotation), points.T).T + mean
    else:
        return numpy.random.multivariate_normal(mean, cov, (samples))


def showEvalFull():
    num_of_runs = 10

    gdata = []
    gdata2 = []

    for i in range(num_of_runs):
        path = outfolder + str(i + 1) + "/artificialTrain_2900_perf.p"
        ptr = open(path, "rb")
        datastruc = pickle.load(ptr)
        ptr.close()

        plotdata = []
        plotdata2 = []
        for entry in range(0, 2900, 100):
            [perf, protnum, probestimator] = datastruc[entry]
            print probestimator
            if len(plotdata) > 0:
                plotdata = numpy.vstack([plotdata, [int(entry), 1 - perf]])
            else:
                plotdata = [int(entry), 1 - perf]
            if len(plotdata2) > 0:
                plotdata2 = numpy.vstack([plotdata2, [int(entry), protnum]])
            else:
                plotdata2 = [int(entry), protnum]
            print protnum

        if (len(gdata) == 0):
            gdata = numpy.array(plotdata)
        else:
            gdata = numpy.hstack((gdata, plotdata))

        if (len(gdata2) == 0):
            gdata2 = numpy.array(plotdata2)
        else:
            gdata2 = numpy.hstack((gdata2, plotdata2))

    gdata3 = []
    gdata4 = []

    for i in range(num_of_runs):
        path = outfolder + "1_" + str(i + 1) + "/artificialTrain_2900_perf.p"
        ptr = open(path, "rb")
        datastruc = pickle.load(ptr)
        ptr.close()

        plotdata = []
        plotdata2 = []
        for entry in range(0, 2900, 100):
            [perf, protnum, probestimator] = datastruc[entry]
            print probestimator
            if len(plotdata) > 0:
                plotdata = numpy.vstack([plotdata, [int(entry), 1 - perf]])
            else:
                plotdata = [int(entry), 1 - perf]
            if len(plotdata2) > 0:
                plotdata2 = numpy.vstack([plotdata2, [int(entry), protnum]])
            else:
                plotdata2 = [int(entry), protnum]
            print protnum

        if (len(gdata3) == 0):
            gdata3 = numpy.array(plotdata)
        else:
            gdata3 = numpy.hstack((gdata3, plotdata))

        if (len(gdata4) == 0):
            gdata4 = numpy.array(plotdata2)
        else:
            gdata4 = numpy.hstack((gdata4, plotdata2))


    #plt.plot(gdata[:,0], numpy.var(gdata[:,1::2], axis=1))
    #plt.show()


    plotframe = plt.figure()
    ax = plotframe.add_subplot(111)

    lineslist = list()
    ndist = scipy.sqrt(numpy.var(gdata2[:, 1::2], axis=1))
    means = numpy.mean(gdata2[:, 1::2], axis=1)

    print means, ndist

    xs = list(gdata[:, 0])
    ys = list((means + ndist)[:])
    ys2 = list((means - ndist)[:])
    ys2.reverse()
    xs2 = list(xs)
    xs2.reverse()
    coords = zip(xs + xs2, ys + ys2)
    patch1 = plt.Polygon(coords, facecolor=[0.8, 0.8, 0.8, 0.8], edgecolor='none')
    lineslist.append(patch1)

    ys = list((means + 0.5 * ndist)[:])
    ys2 = list((means - 0.5 * ndist)[:])
    ys2.reverse()
    coords = zip(xs + xs2, ys + ys2)
    patch2 = plt.Polygon(coords, facecolor=[0.5, 0.5, 0.5, 0.8], edgecolor='none')
    lineslist.append(patch2)

    ax2 = ax
    ndist = scipy.sqrt(numpy.var(gdata4[:, 1::2], axis=1))
    mmeans = numpy.mean(gdata4[:, 1::2], axis=1)
    xxs = list(gdata[:, 0])
    ys = list((mmeans + ndist)[:])
    ys2 = list((mmeans - ndist)[:])
    ys2.reverse()
    xs2 = list(xxs)
    xs2.reverse()
    coords = zip(xxs + xs2, ys + ys2)
    patch3 = plt.Polygon(coords, facecolor=[0.8, 0.8, 0.8, 0.8], edgecolor='none')
    lineslist.append(patch3)

    ys = list((mmeans + 0.5 * ndist)[:])
    ys2 = list((mmeans - 0.5 * ndist)[:])
    ys2.reverse()
    coords = zip(xxs + xs2, ys + ys2)
    patch4 = plt.Polygon(coords, facecolor=[0.5, 0.5, 0.5, 0.8], edgecolor='none')
    lineslist.append(patch4)

    ax.add_patch(patch1)
    ax.add_patch(patch3)
    ax.add_patch(patch2)
    ax.add_patch(patch4)

    line = ax2.plot(xs, means, '-')
    line = ax2.plot(xxs, mmeans, '-')
    legendtext = ['3D Dataset', '2D Dataset', 'Std Deviation']
    ax.set_xlabel("# of Training Samples", size=18)
    ax.set_ylabel("# of Nodes", size=18)
    #ax.set_title("g_max Optimiazation")

    ax.legend(legendtext, 'upper left')
    ax.set_xlim(0, 2800)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)

    plt.setp(ax.get_legend().get_texts(), fontsize=16)

    ax.grid(True)
    plotframe.show()
    DefaultSize = plotframe.get_size_inches()
    plotframe.set_size_inches((DefaultSize[0] * 1.5, DefaultSize[1]))
    plotframe.savefig('/hri/localdisk/jqueisse/generatedfigs/artificialnumberofnodes.pdf', transparent=True)
    plt.show()


def showEvalFullPerf():
    num_of_runs = 10

    gdata = []
    gdata2 = []

    for i in range(num_of_runs):
        path = outfolder + str(i + 1) + "/artificialTrain_2900_perf.p"
        ptr = open(path, "rb")
        datastruc = pickle.load(ptr)
        ptr.close()

        plotdata = []
        plotdata2 = []
        for entry in range(0, 2900, 100):
            [perf, protnum, probestimator] = datastruc[entry]
            print probestimator
            if len(plotdata) > 0:
                plotdata = numpy.vstack([plotdata, [int(entry), 1 - perf]])
            else:
                plotdata = [int(entry), 1 - perf]
            if len(plotdata2) > 0:
                plotdata2 = numpy.vstack([plotdata2, [int(entry), protnum]])
            else:
                plotdata2 = [int(entry), protnum]
            print protnum

        if (len(gdata) == 0):
            gdata = numpy.array(plotdata)
        else:
            gdata = numpy.hstack((gdata, plotdata))

        if (len(gdata2) == 0):
            gdata2 = numpy.array(plotdata2)
        else:
            gdata2 = numpy.hstack((gdata2, plotdata2))

    gdata3 = []
    gdata4 = []

    for i in range(num_of_runs):
        path = outfolder + "1_" + str(i + 1) + "/artificialTrain_2900_perf.p"
        ptr = open(path, "rb")
        datastruc = pickle.load(ptr)
        ptr.close()

        plotdata = []
        plotdata2 = []
        for entry in range(0, 2900, 100):
            [perf, protnum, probestimator] = datastruc[entry]
            print probestimator
            if len(plotdata) > 0:
                plotdata = numpy.vstack([plotdata, [int(entry), 1 - perf]])
            else:
                plotdata = [int(entry), 1 - perf]
            if len(plotdata2) > 0:
                plotdata2 = numpy.vstack([plotdata2, [int(entry), protnum]])
            else:
                plotdata2 = [int(entry), protnum]
            print protnum

        if (len(gdata3) == 0):
            gdata3 = numpy.array(plotdata)
        else:
            gdata3 = numpy.hstack((gdata3, plotdata))

        if (len(gdata4) == 0):
            gdata4 = numpy.array(plotdata2)
        else:
            gdata4 = numpy.hstack((gdata4, plotdata2))


    #plt.plot(gdata[:,0], numpy.var(gdata[:,1::2], axis=1))
    #plt.show()


    plotframe = plt.figure()
    ax = plotframe.add_subplot(111)

    lineslist = list()
    ndist = scipy.sqrt(numpy.var(gdata[:, 1::2], axis=1))
    means = numpy.mean(gdata[:, 1::2], axis=1)

    print means, ndist

    xs = list(gdata[:, 0])
    ys = list((means + ndist)[:])
    ys2 = list((means - ndist)[:])
    ys2.reverse()
    xs2 = list(xs)
    xs2.reverse()
    coords = zip(xs + xs2, ys + ys2)
    patch1 = plt.Polygon(coords, facecolor=[0.8, 0.8, 0.8, 0.8], edgecolor='none')
    lineslist.append(patch1)

    ys = list((means + 0.5 * ndist)[:])
    ys2 = list((means - 0.5 * ndist)[:])
    ys2.reverse()
    coords = zip(xs + xs2, ys + ys2)
    patch2 = plt.Polygon(coords, facecolor=[0.5, 0.5, 0.5, 0.8], edgecolor='none')
    lineslist.append(patch2)

    ax2 = ax
    ndist = scipy.sqrt(numpy.var(gdata3[:, 1::2], axis=1))
    mmeans = numpy.mean(gdata3[:, 1::2], axis=1)
    xxs = list(gdata[:, 0])
    ys = list((mmeans + ndist)[:])
    ys2 = list((mmeans - ndist)[:])
    ys2.reverse()
    xs2 = list(xxs)
    xs2.reverse()
    coords = zip(xxs + xs2, ys + ys2)
    patch3 = plt.Polygon(coords, facecolor=[0.8, 0.8, 0.8, 0.8], edgecolor='none')
    lineslist.append(patch3)

    ys = list((mmeans + 0.5 * ndist)[:])
    ys2 = list((mmeans - 0.5 * ndist)[:])
    ys2.reverse()
    coords = zip(xxs + xs2, ys + ys2)
    patch4 = plt.Polygon(coords, facecolor=[0.5, 0.5, 0.5, 0.8], edgecolor='none')
    lineslist.append(patch4)

    ax.add_patch(patch1)
    ax.add_patch(patch3)
    ax.add_patch(patch2)
    ax.add_patch(patch4)

    line = ax2.plot(xs, means, '-')
    line = ax2.plot(xxs, mmeans, '-')
    legendtext = ['3D Dataset', '2D Dataset', 'Std Deviation']
    ax.set_xlabel("# of Training Samples", size=18)
    ax.set_ylabel("Error Rate", size=18)
    #ax.set_title("g_max Optimiazation")

    ax.legend(legendtext, 'upper right')
    ax.set_xlim(0, 2800)
    ax.set_ylim(0, 0.7)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)

    plt.setp(ax.get_legend().get_texts(), fontsize=16)

    ax.grid(True)
    plotframe.show()
    DefaultSize = plotframe.get_size_inches()
    plotframe.set_size_inches((DefaultSize[0] * 1.5, DefaultSize[1]))
    plotframe.savefig('/hri/localdisk/jqueisse/generatedfigs/artificialperformance.pdf', transparent=True)
    plt.show()


def showEvalFullPerfII():
    num_of_runs = 10

    gdata = []
    gdata2 = []

    for i in range(num_of_runs):
        path = outfolder + str(i + 1) + "/artificialTrain_2900_perf.p"
        ptr = open(path, "rb")
        datastruc = pickle.load(ptr)
        ptr.close()

        plotdata = []
        plotdata2 = []
        for entry in range(0, 2900, 100):
            [perf, protnum, probestimator] = datastruc[entry]
            print probestimator
            if len(plotdata) > 0:
                plotdata = numpy.vstack([plotdata, [int(entry), 1 - perf]])
            else:
                plotdata = [int(entry), 1 - perf]
            if len(plotdata2) > 0:
                plotdata2 = numpy.vstack([plotdata2, [int(entry), protnum]])
            else:
                plotdata2 = [int(entry), protnum]
            print protnum

        if (len(gdata) == 0):
            gdata = numpy.array(plotdata)
        else:
            gdata = numpy.hstack((gdata, plotdata))

        if (len(gdata2) == 0):
            gdata2 = numpy.array(plotdata2)
        else:
            gdata2 = numpy.hstack((gdata2, plotdata2))

    gdata3 = []
    gdata4 = []

    for i in range(num_of_runs):
        path = outfolder + "2_" + str(i + 1) + "/artificialTrain_2900_perf.p"
        ptr = open(path, "rb")
        datastruc = pickle.load(ptr)
        ptr.close()

        plotdata = []
        plotdata2 = []
        for entry in range(0, 2900, 100):
            [perf, protnum, probestimator] = datastruc[entry]
            print probestimator
            if len(plotdata) > 0:
                plotdata = numpy.vstack([plotdata, [int(entry), 1 - perf]])
            else:
                plotdata = [int(entry), 1 - perf]
            if len(plotdata2) > 0:
                plotdata2 = numpy.vstack([plotdata2, [int(entry), protnum]])
            else:
                plotdata2 = [int(entry), protnum]
            print protnum

        if (len(gdata3) == 0):
            gdata3 = numpy.array(plotdata)
        else:
            gdata3 = numpy.hstack((gdata3, plotdata))

        if (len(gdata4) == 0):
            gdata4 = numpy.array(plotdata2)
        else:
            gdata4 = numpy.hstack((gdata4, plotdata2))


    #plt.plot(gdata[:,0], numpy.var(gdata[:,1::2], axis=1))
    #plt.show()




    lineslist = list()
    ndist = scipy.sqrt(numpy.var(gdata3[:, 1::2], axis=1))
    means = numpy.mean(gdata3[:, 1::2], axis=1)

    print means, ndist

    xs = list(gdata[:, 0])
    ys = list((means + ndist)[:])
    ys2 = list((means - ndist)[:])
    ys2.reverse()
    xs2 = list(xs)
    xs2.reverse()
    coords = zip(xs + xs2, ys + ys2)
    patch1 = plt.Polygon(coords, facecolor=[0.8, 0.8, 0.8, 0.8], edgecolor='none')
    lineslist.append(patch1)

    ys = list((means + 0.5 * ndist)[:])
    ys2 = list((means - 0.5 * ndist)[:])
    ys2.reverse()
    coords = zip(xs + xs2, ys + ys2)
    patch2 = plt.Polygon(coords, facecolor=[0.5, 0.5, 0.5, 0.8], edgecolor='none')
    lineslist.append(patch2)

    ndist = scipy.sqrt(numpy.var(gdata4[:, 1::2], axis=1))
    mmeans = numpy.mean(gdata4[:, 1::2], axis=1)
    xxs = list(gdata[:, 0])
    ys = list((mmeans + ndist)[:])
    ys2 = list((mmeans - ndist)[:])
    ys2.reverse()
    xs2 = list(xxs)
    xs2.reverse()
    coords = zip(xxs + xs2, ys + ys2)
    patch3 = plt.Polygon(coords, facecolor=[0.8, 0.8, 0.8, 0.8], edgecolor='none')
    lineslist.append(patch3)

    ys = list((mmeans + 0.5 * ndist)[:])
    ys2 = list((mmeans - 0.5 * ndist)[:])
    ys2.reverse()
    coords = zip(xxs + xs2, ys + ys2)
    patch4 = plt.Polygon(coords, facecolor=[0.5, 0.5, 0.5, 0.8], edgecolor='none')
    lineslist.append(patch4)

    plotframe = plt.figure()
    ax = plotframe.add_subplot(111)
    ax2 = ax  #.twinx()
    ax.add_patch(patch1)
    #ax2.add_patch(patch3)
    ax.add_patch(patch2)
    #ax2.add_patch(patch4)


    line = ax.plot(xs, means, 'g-')
    #line = ax2.plot(xxs, mmeans, 'b-',zorder=10)
    legendtext = ['Error Rate', 'Std Deviation']
    ax.set_xlabel("# of Training Samples", size=18)
    ax.set_ylabel("Error Rate", size=18)
    #ax.set_title("g_max Optimiazation")

    ax.legend(legendtext, 'upper right')
    ax.set_xlim(0, 2800)
    ax.set_ylim(0, 0.7)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)

    plt.setp(ax.get_legend().get_texts(), fontsize=16)

    ax.grid(True)
    plotframe.show()
    DefaultSize = plotframe.get_size_inches()
    plotframe.set_size_inches((DefaultSize[0] * 1.5, DefaultSize[1]))
    plotframe.savefig('/hri/localdisk/jqueisse/generatedfigs/artificialperformanceII.pdf', transparent=True)
    plt.show()

    plotframe = plt.figure()
    ax = plotframe.add_subplot(111)
    ax2 = ax  #.twinx()
    #ax.add_patch(patch1)
    ax2.add_patch(patch3)
    #ax.add_patch(patch2)
    ax2.add_patch(patch4)


    #line = ax.plot(xs, means, 'g-')
    line = ax2.plot(xxs, mmeans, 'b-', zorder=10)
    legendtext = ['# of Nodes', 'Std Deviation']
    ax.set_xlabel("# of Training Samples", size=18)
    ax.set_ylabel("# of Nodes", size=18)
    #ax.set_title("g_max Optimiazation")

    ax.legend(legendtext, 'upper left')
    ax.set_xlim(0, 2800)
    #ax.set_ylim(0,0.7)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)

    plt.setp(ax.get_legend().get_texts(), fontsize=16)

    ax.grid(True)
    plotframe.show()
    DefaultSize = plotframe.get_size_inches()
    plotframe.set_size_inches((DefaultSize[0] * 1.5, DefaultSize[1]))
    plotframe.savefig('/hri/localdisk/jqueisse/generatedfigs/artificialnodesII.pdf', transparent=True)
    plt.show()


def showEval():
    file = outfolder + "artificialTrain_2900_perf.p"

    ptr = open(file, "rb")
    datastruc = pickle.load(ptr)
    ptr.close()

    plotdata = []
    plotdata2 = []

    for entry in range(0, 2900, 100):
        [perf, protnum, probestimator] = datastruc[entry]
        print probestimator
        if len(plotdata) > 0:
            plotdata = numpy.vstack([plotdata, [int(entry), 1 - perf]])
        else:
            plotdata = [int(entry), 1 - perf]
        if len(plotdata2) > 0:
            plotdata2 = numpy.vstack([plotdata2, [int(entry), protnum]])
        else:
            plotdata2 = [int(entry), protnum]
        print protnum

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    lns1 = ax1.plot(plotdata[:, 0], plotdata[:, 1], "b-", label="Error Rate")
    ax1.set_xlabel('# of Training Samples')
    ax1.set_ylabel("Error Rate")

    ax2 = ax1.twinx()
    ax2.set_ylim(0, 15)
    lns2 = ax2.plot(plotdata2[:, 0], plotdata2[:, 1], "g-", label="# of Nodes")
    ax2.set_ylabel("# of Nodes")

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)


    #plt.xlabel("# of Training samples")
    #plt.ylabel("Error Rate")


    #plt.plot(plotdata2[:,0], plotdata2[:,1])

    #plt.savefig(outfolder+'artificialTrainperfdev.pdf', transparent=True)
    #plt.savefig(outfolder+'artificialNodecountdev.pdf', transparent=True)
    plt.show()


def singleperf():
    perfarr = numpy.array(
        [0.5, 0.275, 0.275, 0.29, 0.225, 0.24, 0.21, 0.21, 0.16, 0.155, 0.15, 0.14, 0.14, 0.15, 0.145, 0.145, 0.3, 0.1,
         0.11, 0.11, 0.112, 0.115, 0.054, 0.062, 0.058, 0.058, 0.06, 0.048, 0.048])
    perfaxis = range(0, 2900, 100)

    print len(perfarr), len(perfaxis)
    plotframe = plt.figure()
    ax = plotframe.add_subplot(111)
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 0.6)
    ax.plot(perfaxis, perfarr[0:len(perfaxis)])

    ax.plot(1600, 0.3, 'o', markersize=20, markerfacecolor='none', markeredgewidth=2, markeredgecolor='red')


    #ellipse1 = Circle((200,0.200), 0.50, edgecolor='k', fill=False)
    #ax.add_patch(ellipse1)

    ax.set_xlabel("# of Training Samples", size=18)
    ax.set_ylabel("# of Nodes", size=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)

    ax.grid(True)
    plotframe.show()
    DefaultSize = plotframe.get_size_inches()
    plotframe.set_size_inches((DefaultSize[0] * 1.5, DefaultSize[1]))
    plotframe.savefig('/hri/localdisk/jqueisse/generatedfigs/artificialTrainperfdev.pdf', transparent=True)

    plt.show()


'''
Estimate the performance of a giiven network
'''


def estimatePerf(datatestfull, resulttestfull, v, net, probestimator):
    correctcount = 0
    allcount = 0
    for i in range(datatestfull.shape[0]):
        sampledata = datatestfull[i, :]
        sampleresult = resulttestfull[i]

        percent = float(i) / datatestfull.shape[0]

        (svmresult, (accuracy, mean_squared, correlation_coefficient), probablility) = svm_predict([0], [
            sampledata.tolist()], v)
        svmresult = svmresult[0]
        [netres, index] = libgxlvq.process_network(net, sampledata)
        netres = netres[0]
        situation = index[0]

        #Get approximated performance for one input sample
        #Or use initial perf settings is no node perf approx is available
        if not probestimator.has_key(situation):
            probestimator[situation] = [1.0 / float(num_of_classes), 1.0 / float(num_of_classes)]
        pobabilities = probestimator[situation]

        #Estimate performance for each class
        resultval = [0, 0, 0]
        for i in range(3):
            if svmresult == i:
                resultval[i] += pobabilities[0]
            else:
                resultval[i] += (1 - pobabilities[0]) / 2.0

            if netres == i:
                resultval[i] += pobabilities[1]
            else:
                resultval[i] += (1 - pobabilities[1]) / 2.0

        resultval = numpy.array(resultval)
        ressum = numpy.sum(resultval)
        resultval /= ressum

        #Determine winner class
        winindex = numpy.argmax(resultval)

        #Determine used classifier (by probability approximation)
        if pobabilities[0] >= pobabilities[1]:
            selectedClassifier = 0
        else:
            selectedClassifier = 1

        #Track the number of correct classified samples
        if winindex == sampleresult:
            correctcount += 1
        allcount += 1

    print "Testresult: ", float(correctcount) / float(allcount)

    return [float(correctcount) / float(allcount), libgxlvq.get_numofprototypes_network(net), deepcopy(probestimator)]


if __name__ == '__main__':

    #Used for evaluation plotting
    #outfolder="/hri/localdisk/jqueisse/tmpgen/2_10/"
    #singleperf()
    #1/0

    for i in range(10):

        outfolder = "/hri/localdisk/jqueisse/tmpgen/2_" + str(i + 1) + "/"

        #1-Simple dataset
        #2-complex dataset
        datasettype = 2
        #Use third dimension for single dataset
        use3d = False

        '''
        Initialize the randomized dataset
        '''
        if datasettype == 1:
            a1 = createNormDistData(1000, mean=(0, 0), var=(30, 1), rotation=numpy.pi / 2) * 0.1
            b1 = createNormDistData(1000, mean=(10, 10), var=(10, 1), rotation=numpy.pi / 2) * 0.1
            a2 = createNormDistData(1000, mean=(20, 10), var=(10, 1), rotation=numpy.pi / 2) * 0.1
            b2 = createNormDistData(1000, mean=(30, 0), var=(30, 1), rotation=numpy.pi / 2) * 0.1

            classc = createNormDistData(2000, mean=(15, -7), var=(90, 2), rotation=0) * 0.1
            if use3d:
                classc_Z = numpy.sin((numpy.pi / 30) * classc[:, 0] - numpy.pi / 2) * 1
            else:
                classc_Z = 0 * classc[:, 0]
            print classc_Z.shape, classc.shape
            classc = numpy.vstack([classc.T, classc_Z]).T
            print classc.shape

            classa = numpy.vstack([a1, a2])
            classb = numpy.vstack([b1, b2])

            result = numpy.zeros([classa.shape[0]])
            data = numpy.vstack([classa, classb])
            result = numpy.hstack([result, numpy.ones([classb.shape[0]])])
            resultfull = numpy.hstack([result, 2 * numpy.ones([classc.shape[0]])])

            print data.shape, numpy.zeros((data.shape[0], 1)).shape
            data3d = numpy.hstack([data, numpy.zeros((data.shape[0], 1))])
            datafull = numpy.vstack([data3d, classc])

            datatrain = data[0:data.shape[0]:2, :]
            datatest = data[1:data.shape[0]:2, :]
            resultrain = result[0:result.shape[0]:2]
            resulttest = result[1:result.shape[0]:2]

            datatrainfull = datafull[0:datafull.shape[0]:2, :]
            datatestfull = datafull[1:datafull.shape[0]:2, :]
            resultrainfull = resultfull[0:resultfull.shape[0]:2]
            resulttestfull = resultfull[1:resultfull.shape[0]:2]

        else:
            dist = 5.0
            parts = 10

            classa = None
            classb = None

            classa2 = None
            classb2 = None

            for p in range(parts):
                rot = p * (2 * numpy.pi / parts)
                center = (numpy.sin(rot) * dist - 4, numpy.cos(rot) * dist)
                cdata = createNormDistData(300, mean=center, var=(0.1, 1), rotation=-rot)

                classtype = p % 2

                if classtype == 0:
                    if classa == None:
                        classa = cdata
                    else:
                        classa = numpy.vstack([classa, cdata])
                else:
                    if classb == None:
                        classb = cdata
                    else:
                        classb = numpy.vstack([classb, cdata])

            for p in range(parts):
                rot = p * (2 * numpy.pi / parts)
                center = (numpy.sin(rot) * dist + 4, numpy.cos(rot) * dist)
                cdata = createNormDistData(300, mean=center, var=(0.1, 1), rotation=-rot)

                classtype = p % 2

                if classtype == 0:
                    if classa2 == None:
                        classa2 = numpy.vstack([classa, cdata])
                    else:
                        classa2 = numpy.vstack([classa2, cdata])
                else:
                    if classb2 == None:
                        classb2 = numpy.vstack([classb, cdata])
                    else:
                        classb2 = numpy.vstack([classb2, cdata])

            classc = createNormDistData(500, mean=(-4, 0), var=(0.5, 0.5), rotation=0)
            classc_Z = numpy.sin((numpy.pi / 30) * classc[:, 0] - numpy.pi / 2)
            #classc = numpy.vstack([classc.T, classc_Z]).T

            classc2 = createNormDistData(500, mean=(4, 0), var=(0.5, 0.5), rotation=0)
            classc_Z = numpy.sin((numpy.pi / 30) * classc2[:, 0] - numpy.pi / 2)
            #classc2 = numpy.vstack([classc2.T, classc_Z]).T
            classc = numpy.vstack([classc, classc2])

            result = numpy.zeros([classa.shape[0]])
            data = numpy.vstack([classa, classb])
            result = numpy.hstack([result, numpy.ones([classb.shape[0]])])
            result2 = numpy.zeros([classa2.shape[0]])
            data2 = numpy.vstack([classa2, classb2])
            result2 = numpy.hstack([result2, numpy.ones([classb2.shape[0]])])

            resultfull = numpy.hstack([result2, 2 * numpy.ones([classc2.shape[0]])])

            data3d = data  #numpy.hstack([data, numpy.zeros((data.shape[0],1))])
            data3d2 = data2  #numpy.hstack([data2, numpy.zeros((data2.shape[0],1))])
            datafull = numpy.vstack([data3d2, classc2])

            datatrain = data[0:data.shape[0]:2, :]
            datatest = data[1:data.shape[0]:2, :]
            resultrain = result[0:result.shape[0]:2]
            resulttest = result[1:result.shape[0]:2]

            datatrainfull = datafull[0:datafull.shape[0]:2, :]
            datatestfull = datafull[1:datafull.shape[0]:2, :]
            resultrainfull = resultfull[0:resultfull.shape[0]:2]
            resulttestfull = resultfull[1:resultfull.shape[0]:2]

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.hold(True)
        ax.plot(classa[:, 0], classa[:, 1], '1y', classb[:, 0], classb[:, 1], '2m', classc[:, 0], classc[:, 1], '3g')
        #ax.set_title('Data Input')

        if datasettype == 1:
            plt.ylim([-2, 2.5])
            plt.xlim([-2, 5])
        else:
            plt.ylim([-10, 10])
            plt.xlim([-15, 15])
        plt.savefig(outfolder + 'artificialDatasetII.pdf', transparent=True)





        #train svm:
        px = svm.svm_problem(resultrain, datatrain.tolist())
        # the kernel function is RBF (radial basis function)
        pm = svm.svm_parameter()
        pm.kernel_type = svm.RBF
        pm.probability = False

        v = svm_train(px, pm)

        (result, (accuracy, mean_squared, correlation_coefficient), probablility) = svm_predict(resulttest,
                                                                                                datatest.tolist(), v)
        print accuracy, mean_squared, correlation_coefficient
        print  resulttestfull.shape, datatestfull.shape, datafull.shape
        (result, (accuracy, mean_squared, correlation_coefficient), probablility) = svm_predict(resulttestfull,
                                                                                                datatestfull[:,
                                                                                                0:2].tolist(), v)
        print accuracy, mean_squared, correlation_coefficient



        #Create Online learner!
        net = "net" + str(i)
        if datasettype == 1:
            dimensionality = 3
        else:
            dimensionality = 2
        nettype = libgxlvq.NETTYPE_GMLVQ
        learnrate_per_node = libgxlvq.NETWORK_LEARNRATE
        num_of_classes = 3
        prototypes_per_class = 1
        trainsteps = 3000
        evalstepwidth = 100
        do_node_random_init = True
        threads_per_nodes = 1
        initialdimensions = 0
        g_max = 20
        libgxlvq.create_network(net, dimensionality, nettype, learnrate_per_node, num_of_classes, prototypes_per_class,
                                trainsteps, do_node_random_init, threads_per_nodes, initialdimensions)

        colors = ["y", "m", "b"]

        #Used for per node probability approximation nodenr->probability
        probestimator = dict()

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.hold(True)

        trainres = dict()

        if datasettype == 1:
            traineddata = {0: numpy.zeros((1, 3)), 1: numpy.zeros((1, 3)), 2: numpy.zeros((1, 3))}
        else:
            traineddata = {0: numpy.zeros((1, 2)), 1: numpy.zeros((1, 2)), 2: numpy.zeros((1, 2))}

        #train with random samples
        for i in range(trainsteps):
            samplenr = random.randrange(0, datatrainfull.shape[0])
            sampledata = datatrainfull[samplenr, :]
            sampleresult = resultrainfull[samplenr]

            percent = float(i) / trainsteps

            print "Percent", percent
            failinc = True
            netmode = libgxlvq.NETMODE_WEIGHTS
            if percent >= 0:
                if percent < 0.5:
                    #Enable node insertion for the first half of the trainsamples
                    failinc = True
                else:
                    failinc = True
                if percent > 0.40:
                    #Enable net adaption and lambda adoption for the last 60 percent
                    netmode = libgxlvq.NETMODE_BOTH
                else:
                    #Only adopt network for the first 40 percent
                    netmode = libgxlvq.NETMODE_WEIGHTS


            #predict the closest node for the given input sample
            (svmresult, (accuracy, mean_squared, correlation_coefficient), probablility) = svm_predict([0], [
                sampledata.tolist()], v)
            svmresult = svmresult[0]
            [netres, index] = libgxlvq.process_network(net, sampledata)
            netres = netres[0]
            #situation holds the node id
            situation = index[0]

            #Request probability approximation
            if not probestimator.has_key(situation):
                #If no approximation is available use std initialization
                probestimator[situation] = [0.95, 1.0 / float(num_of_classes)]
            pobabilities = probestimator[situation]

            #update the probability approximation:
            #old = old*(1-fac) (+ fac if prediction is correct)
            probhistory = 50
            fac = 1.0 / float(probhistory)
            pobabilities[0] *= (1 - fac)
            pobabilities[1] *= (1 - fac)

            print svmresult, sampleresult, netres
            if svmresult == sampleresult:
                pobabilities[0] += fac
                print "svm ok"
            else:
                print "svm fail!"

            if netres == sampleresult:
                pobabilities[1] += fac
                print "net ok"
            else:
                print "net fail!", situation

            probestimator[situation] = pobabilities
            print "probs: ", probestimator


            #determine probability for all classes
            #since probability is only given for the -winner- class
            #other classes are set to (1-probability)/(# of other classes)
            resultval = [0, 0, 0]
            for ij in range(3):
                if svmresult == ij:
                    resultval[ij] += pobabilities[0]
                else:
                    resultval[ij] += (1 - pobabilities[0]) / 2.0

                if netres == ij:
                    resultval[ij] += pobabilities[1]
                else:
                    resultval[ij] += (1 - pobabilities[1]) / 2.0

            resultval = numpy.array(resultval)
            ressum = numpy.sum(resultval)
            resultval /= ressum

            winindex = numpy.argmax(resultval)

            if pobabilities[0] >= pobabilities[1]:
                selectedClassifier = 0
            else:
                selectedClassifier = 1




            #Present new traindata to classifier
            #if the online classifier is activ for the current region
            #There was an classification error
            #A given probability was not reached
            print "traindata", sampledata, sampleresult
            if selectedClassifier > 0 or not winindex == sampleresult or pobabilities[selectedClassifier] < 0.9:
                print "Train!", sampleresult, svmresult, netres, pobabilities, resultval

                [delta, perf] = libgxlvq.incremental_train_network(net, sampledata, [sampleresult], failinc, g_max,
                                                                   netmode, False)
                if len(traineddata[sampleresult]) > 0:
                    traineddata[sampleresult] = numpy.vstack((traineddata[sampleresult], sampledata))
                else:
                    traineddata[sampleresult] = sampledata
                print "GXLVQ", perf

                vis = True
                if vis == True:

                    ax.clear()

                    cla = traineddata[0]
                    clb = traineddata[1]
                    clc = traineddata[2]

                    ax.plot(cla[:, 0], cla[:, 1], '1' + colors[0], clb[:, 0], clb[:, 1], '2' + colors[1], clc[:, 0],
                            clc[:, 1], '3' + colors[2])

                    protos = libgxlvq.get_numofprototypes_network(net)
                    for p in range(protos):
                        pos = libgxlvq.get_weights_network(net, p)
                        labelnr = libgxlvq.get_prototypelabel_network(net, p)
                        ax.plot(pos[0], pos[1], "D" + colors[labelnr])

                    if datasettype == 1:
                        plt.ylim([-2, 2.5])
                        plt.xlim([-2, 5])
                    else:
                        plt.ylim([-10, 10])
                        plt.xlim([-15, 15])

                    fig.show()
                    plt.draw()


            #Estimate performance of current net configuration
            if i % evalstepwidth == 0:
                perfres = estimatePerf(datatestfull, resulttestfull, v, net, probestimator)
                trainres[i] = perfres
                fig.savefig(outfolder + 'artificialTrain_' + str(i) + '.pdf', transparent=True)

                ptr = open(outfolder + 'artificialTrain_' + str(i) + '_perf.p', "wb")
                pickle.dump(trainres, ptr)
                ptr.close()

        protos = libgxlvq.get_numofprototypes_network(net)
        for p in range(protos):
            pos = libgxlvq.get_weights_network(net, p)
            labelnr = libgxlvq.get_prototypelabel_network(net, p)
            if probestimator.has_key(p):
                print "Proto ", p, " Pos", pos, " label ", labelnr, " prob", probestimator[p]
            else:
                print "Proto ", p, " Pos", pos, " label ", labelnr, " prob unknown!"



        #Estimate performance (same scheme as before)
        correctcount = 0
        allcount = 0
        for i in range(datatestfull.shape[0]):
            sampledata = datatestfull[i, :]
            sampleresult = resulttestfull[i]

            percent = float(i) / datatestfull.shape[0]

            (svmresult, (accuracy, mean_squared, correlation_coefficient), probablility) = svm_predict([0], [
                sampledata.tolist()], v)
            svmresult = svmresult[0]
            [netres, index] = libgxlvq.process_network(net, sampledata)
            netres = netres[0]
            situation = index[0]

            if not probestimator.has_key(situation):
                probestimator[situation] = [1.0 / float(num_of_classes), 1.0 / float(num_of_classes)]
            pobabilities = probestimator[situation]

            resultval = [0, 0, 0]
            for i in range(3):
                if svmresult == i:
                    resultval[i] += pobabilities[0]
                else:
                    resultval[i] += (1 - pobabilities[0]) / 2.0

                if netres == i:
                    resultval[i] += pobabilities[1]
                else:
                    resultval[i] += (1 - pobabilities[1]) / 2.0

            resultval = numpy.array(resultval)
            ressum = numpy.sum(resultval)
            resultval /= ressum

            winindex = numpy.argmax(resultval)

            if pobabilities[0] >= pobabilities[1]:
                selectedClassifier = 0
            else:
                selectedClassifier = 1

            if winindex == sampleresult:
                correctcount += 1
            allcount += 1

        print "Testresult: ", float(correctcount) / float(allcount)


