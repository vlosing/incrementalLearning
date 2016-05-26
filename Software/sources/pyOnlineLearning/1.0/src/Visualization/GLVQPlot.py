import matplotlib.pyplot as plt
import numpy as np


def getDefColors():
    return np.array(['#0000FF', '#FF0000', '#00CC01', '#2F2F2F', '#8900CC', '#0099CC',
                     '#ACE600', '#D9007E', '#FFCCCC', '#5E6600', '#FFFF00', '#999999',
                     '#FF6000', '#00FF00', '#FF00FF', '#00FFFF', '#FFFF0F', '#F0CC01',
                     '#9BC6ED', '#915200',
                     '#0000FF', '#FF0000', '#00CC01', '#2F2F2F', '#8900CC', '#0099CC',
                     '#ACE600', '#D9007E', '#FFCCCC', '#5E6600', '#FFFF00', '#999999',
                     '#FF6000', '#00FF00', '#FF00FF', '#00FFFF', '#FFFF0F', '#F0CC01',
                     '#9BC6ED', '#915200',

                     ])


def getDefXRange():
    return [-19.5, 18.5]


def getDefYRange():
    return [-10.5, 18.5]


def getDefXRange2():
    return [-22, 22]


def getDefYRange2():
    return [-22, 22]


def getDefXRange3():
    return [-3, 3]


def getDefYRange3():
    return [-3, 3]


def getDefXRange4():
    return [-10, 15]


def getDefYRange4():
    return [-10, 5]


def PlotWrongSamples():
    return False

def MarkWrongClassifications():
    return False


def PlotNewPrototypes():
    return False


def plotVisualization(glvq, prototypes, protoLabels, samples, samplesLabels, colors, title, fig, subplot,
                      XRange=getDefXRange2(), YRange=getDefYRange2(), doPlotStats=False):
    plotIntern(glvq, prototypes, protoLabels, samples, samplesLabels, colors, title, fig, subplot, XRange=XRange,
               YRange=YRange, plotBoundary=False, emphasizeLastPrototype=True, emphasizeLastSample=True)
    if doPlotStats:
        plotStats(glvq, fig, subplot)


def plotStats(glvq, fig, subplot):
    classRate, costValue = glvq.getShortTermMemoryClassRateOnInternalDistanceMatrix()
    fig.subplots_adjust(left=0.01, right=0.65)

    xTextPos = 18.5
    yTextPos = 17
    spacingY = 1.5
    fontSize = 15
    subplot.text(xTextPos, yTextPos, '#trainSamples: ' + str(glvq.trainStepCount), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, '#wSamples: ' + str(len(glvq.windowSamples)), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, '#protos: ' + str(len(glvq.prototypes)), fontsize=fontSize)
    yTextPos -= spacingY
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, 'window-rate: ' + "{0:.3f}".format(classRate), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, 'window-cost: ' + "{0:.3f}".format(costValue), fontsize=fontSize)
    yTextPos -= spacingY
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, '#insertedProtos: ' + str(glvq.insertedProtoCount), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, '#deletedProtos: ' + str(glvq.deletedProtoCount), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, '#triedInsertions: ' + str(glvq.triedInsertions), fontsize=fontSize)
    yTextPos -= spacingY
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, 'avgInsDelta: ' + "{0:.3f}".format(glvq.avgInsertionWindowDeltaCost), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, 'lastInsDelta: ' + "{0:.3f}".format(glvq.lastInsertionWindowDeltaCost), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, 'avgTInsDelta: ' + "{0:.3f}".format(glvq.avgTriedInsertionWindowDeltaCost), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, 'lastTInsDelta: ' + "{0:.3f}".format(glvq.lastTriedInsertionWindowDeltaCost), fontsize=fontSize)
    yTextPos -= spacingY

    subplot.text(xTextPos, yTextPos, 'avgInsProtos: ' + "{0:.1f}".format(glvq.avgInsertionWindowPrototypeCount), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, 'lastInsProtos: ' + str(glvq.lastInsertionWindowPrototypeCount), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, 'avgTInsProtos: ' + "{0:.1f}".format(glvq.avgTriedInsertionWindowPrototypeCount), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, 'lastTInsProtos: ' + str(glvq.lastTriedInsertionWindowPrototypeCount), fontsize=fontSize)
    yTextPos -= spacingY
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, '#avgInsProtoDens: ' + "{0:.3f}".format(glvq.avgInsertionWindowPrototypeDensity), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, '#lastInsProtoDens: ' + "{0:.3f}".format(glvq.lastInsertionWindowPrototypeDensity), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, '#avgTInsProtoDens: ' + "{0:.3f}".format(glvq.avgTriedInsertionWindowPrototypeDensity), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, '#lastTInsProtoDens: ' + "{0:.3f}".format(glvq.lastTriedInsertionWindowPrototypeDensity), fontsize=fontSize)
    yTextPos -= spacingY


    '''subplot.text(xTextPos, yTextPos,
                 'avgDelDelta: ' + "{0:.3f}".format(glvq.protoDeletionDeltaCost / max(glvq.deletedProtoCount, 1)),
                 fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, 'lastDelDelta: ' + "{0:.3f}".format(glvq.lastDeletionDeltaCost), fontsize=fontSize)
    yTextPos -= spacingY
    yTextPos -= spacingY

    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos,
                 'avgDeltaCost: ' + "{0:.3f}".format(glvq.totalDeltaCost / max(glvq.triedInsertions, 1)), fontsize=fontSize)
    yTextPos -= spacingY
    subplot.text(xTextPos, yTextPos, 'deltaCost: ' + "{0:.3f}".format(glvq.deltaCost), fontsize=15)'''


def plotIntern(glvq, prototypes, protoLabels, samples, samplesLabels, colors, title, fig, subplot,
               XRange=getDefXRange(), YRange=getDefYRange(), emphasizeLastSample=False, emphasizeLastPrototype=False,
               plotBoundary=False):
    fig.hold(True)
    # subplot = fig.add_subplot(111, aspect='equal')
    if len(samplesLabels) > 0:
        plotSamples(subplot, samples, samplesLabels, colors)
        if emphasizeLastSample:
            plotSamples(subplot, np.array([samples[::-1][0]]), np.array([samplesLabels[::-1][0]]), colors, size=60)

    if len(protoLabels) > 0:
        plotPrototypes(subplot, prototypes, protoLabels, colors)
        if emphasizeLastPrototype:
            plotPrototypes(subplot, np.array([prototypes[::-1][0]]), np.array([protoLabels[::-1][0]]), colors, size=400)
        if plotBoundary:
            plotDeciscionBoundary(glvq, protoLabels, subplot, XRange, YRange)
    subplot.set_title(title, fontsize=20)

    subplot.get_axes().xaxis.set_ticks([])
    subplot.set_xlim([XRange[0], XRange[1]])
    subplot.set_ylim([YRange[0], YRange[1]])
    subplot.get_axes().xaxis.set_ticks([])
    subplot.get_axes().yaxis.set_ticks([])


def plotAll(glvq, prototypes, protoLabels, samples, samplesLabels, colors, title, fig=None, subplot=None,
            XRange=getDefXRange(), YRange=getDefYRange(), plotBoundary=True):
    plt.close()
    if fig is None:
        fig = plt.figure(figsize=(5, 5))
    if subplot is None:
        subplot = fig.add_subplot(111, aspect='equal')

    '''if MarkWrongClassifications():
        relSimValues = glvq.getCostFunctionValues(samples, samplesLabels)
        for i in numpy.unique(samplesLabels).astype(int):
            classIdx = numpy.where((samplesLabels==i) & (relSimValues>0))[0]
            subplot.scatter(samples[classIdx,0], samples[classIdx,1], s=20, c=colors[i], edgecolor=colors[i])
            classIdx = numpy.where((samplesLabels==i) & (relSimValues<=0))[0]
            subplot.scatter(samples[classIdx,0], samples[classIdx,1], s=20, c=colors[i], edgecolor='red')
    else:
        for i in numpy.unique(samplesLabels).astype(int):
            classIdx = numpy.where((samplesLabels==i))[0]
            subplot.scatter(samples[classIdx,0], samples[classIdx,1], s=20, c=colors[i],
            edgecolor=colors[samplesLabels[classIdx]])'''

    plotIntern(glvq, prototypes, protoLabels, samples, samplesLabels, colors, title, fig, subplot, XRange=XRange,
               YRange=YRange, plotBoundary=plotBoundary)

    #plt.show()
    return fig, subplot


def plotPrototypes(subplot, prototypes, protosLabels, colors, size=200):
    subplot.scatter(prototypes[:, 0], prototypes[:, 1], s=size, c=colors[protosLabels.astype(int)], marker="^", alpha=0.7, edgecolor='black', linewidths=2)


def highlightPrototypes(subplot, prototypes, protosLabels, colors, size=200):
    subplot.scatter(prototypes[:, 0], prototypes[:, 1], s=size, c=colors[protosLabels.astype(int)], marker="^",
                    edgecolor='black', linewidths=5)


def plotPrototype(subplot, proto, protoLabel, colors):
    subplot.scatter(proto[0], proto[1], s=400, c=colors[int(protoLabel)], marker="^")


def plotSamples(subplot, samples, samplesLabels, colors, size=5):
    subplot.scatter(samples[:, 0], samples[:, 1], s=size, c=colors[samplesLabels.astype(int)],
                    edgecolor=colors[samplesLabels.astype(int)])


def plotDeciscionBoundary(glvq, protoLabels, subplot, XRange, YRange):
    preciscion = 200
    u = np.linspace(XRange[0], XRange[1], preciscion)
    v = np.linspace(YRange[0], YRange[1], preciscion)

    z = np.zeros((len(u), len(v)), dtype=np.uint8)
    for i in range(len(u)):
        for j in range(len(v)):
            tdata = np.array([u[i], v[j]])
            result = glvq.predict(tdata)
            z[i, j] = result

    z = z.transpose()
    for i in np.unique(protoLabels):
        idx = np.where(protoLabels == i)[0][0]
        subplot.contour(u, v, z, [idx, idx + 1], linewidths=[1], colors='black', antialiased=True, alpha=0.5)

def plotContourRelSim(glvq, subplot, XRange, YRange):
    u = np.linspace(XRange[0], XRange[1], 200)
    v = np.linspace(YRange[0], YRange[1], 200)

    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            tdata = np.array([u[i], v[j]])
            dummy, result = glvq.predict(tdata, True)
            z[i, j] = result
    z = z.transpose()
    subplot.contour(u, v, z)

def plotRelSim(glvq, prototypes, protoLabels, data, labels, colors, title,
               XRange=np.array([-15, 15]), YRange=np.array([-10, 10])):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.hold(True)
    relSimValues = glvq.getCostFunctionValues(data, labels)
    cax = ax.scatter(data[:, 0], data[:, 1], s=16, c=(np.abs(relSimValues)), linewidth=0.1)
    ticks_at = [min(np.abs(relSimValues)), 0.5, max(np.abs(relSimValues))]

    cbar = plt.colorbar(cax, ticks=ticks_at, format='%1.2g', fraction=0.03)
    cbar.ax.set_yticklabels(['0', '0.5', '1'])

    plotPrototypes(prototypes, protoLabels, ax, colors)
    if PlotDeciscionBoundary() and len(prototypes) > 0:
        plotDeciscionBoundary(glvq, protoLabels, ax, XRange, YRange)

    # plotContourRelSim(glvq, ax, XRange, YRange)
    ax.set_title(title)

    plt.xlim([XRange[0], XRange[1]])
    plt.ylim([YRange[0], YRange[1]])
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    return fig



def plotWrong(glvq, colors, title, samples, samplesLabels, XRange=getDefXRange(), YRange=getDefYRange()):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.hold(True)
    relSimValues = glvq.getCostFunctionValues(samples, samplesLabels)

    ax.scatter(samples[:, 0], samples[:, 1], s=16, c=(np.abs(relSimValues) * 100 / 1).astype(int))
    ax.set_title(title)
    plotPrototypes(glvq.prototypes(), glvq.prototypeLabels(), ax, colors)
    plt.xlim([XRange[0], XRange[1]])
    plt.ylim([YRange[0], YRange[1]])

    '''for i in range(0, numpy.max(self.samplesLabels) + 1):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        classIdx = numpy.where(samplesLabels == i)
        ax.scatter(samples[classIdx[0],0], samples[classIdx[0],1], s=16,
            c=(numpy.abs(relSimValues[classIdx])*100/1).astype(int), edgecolor=numpy.abs(relSimValues[classIdx]))
        self.plotPrototypes(ax, colors)
        ax.set_title('class ' + str(i))
        plt.ylim(-10,10)
        plt.xlim(-15,15)'''


def plotProtoHist(histList, title):
    # hist, bin_edges = numpy.histogram(numpy.arange(len(histList)), bins=numpy.arange(len(histList)), density=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.bar(np.arange(len(histList)), histList, width=1)
    # plt.xlim(min(bin_edges), max(bin_edges))
    # plt.show()

    #tmp = plt.hist(histList)
    #print tmp
    ax.set_xlabel('Class')
    if title == 'Closest' or title == 'SamplingAcc' or title == 'Voronoi':
        ax.set_ylabel('# Prototypes')
    plt.yticks(np.arange(0, 21, 5.0))
    ax.set_title(title)
    #ax.set_xlim([1,50])
    #ax.set_ylim([0,0.1])

    return fig


