__author__ = 'vlosing'
import copy

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

from Visualization import GLVQPlot
from StatisticsLogger import StatisticsLogger


class StatisticsPlot:
    def __init__(self, TrainStepCountAll, TrainStepCountTestAccAll, TrainStepCountTrainAccAll,
                 TrainStepCountInternSampleAccAll, TrainStepCountDevelopmentAccAll,
                 TrainStepCountTestCostAll, TrainStepCountTrainCostAll, TrainStepCountInternSampleCostAll,

                 ComplexitySizeTrainStepCountAll, ComplexitySizeAll, portionRejectionAll, classRateRejectionAll,
                 recordIntervall, accuracyYLim=[0., 1], title=''):
        self.TrainStepCountAll = TrainStepCountAll
        self.TrainStepCountTestAccAll = TrainStepCountTestAccAll
        self.TrainStepCountTrainAccAll = TrainStepCountTrainAccAll
        self.TrainStepCountInternSampleAccAll = TrainStepCountInternSampleAccAll
        self.TrainStepCountDevelopmentAccAll = TrainStepCountDevelopmentAccAll
        self.TrainStepCountTestCostAll = TrainStepCountTestCostAll
        self.TrainStepCountTrainCostAll = TrainStepCountTrainCostAll
        self.TrainStepCountInternSampleCostAll = TrainStepCountInternSampleCostAll
        self.ComplexitySizeTrainStepCountAll = ComplexitySizeTrainStepCountAll
        self.ComplexitySizeAll = ComplexitySizeAll
        self.portionRejectionAll = portionRejectionAll
        self.classRateRejectionAll = classRateRejectionAll
        self.recordIntervall = recordIntervall
        self.accuracyYLim = accuracyYLim
        self.title = title

    def plotAll(self, cfgNames, dstDir, experimentPrefix, marker='', plotVariance=False):
        plt.rcParams.update({'font.size': 10})
        maxTrainSteps = np.max(self.TrainStepCountAll[cfgNames[0]][0][0])
        TrainStepsXLim = [self.recordIntervall, maxTrainSteps]
        fig, axarr = plt.subplots(2, 2)
        plt.suptitle(self.title)
        self.plotTrainStepsTestAccuracy(cfgNames, plot=axarr[0, 0], dstDir=dstDir, experimentPrefix=experimentPrefix,
                                        marker=marker, xLim=TrainStepsXLim, yLim=self.accuracyYLim,
                                        plotVariance=plotVariance)
        self.plotTrainStepsTrainAccuracy(cfgNames, plot=axarr[1, 0], dstDir=dstDir, experimentPrefix=experimentPrefix,
                                         marker=marker, xLim=TrainStepsXLim, yLim=self.accuracyYLim,
                                         plotVariance=plotVariance)
        self.plotTrainStepsInternTrainAccuracy(cfgNames, plot=axarr[0, 1], dstDir=dstDir,
                                               experimentPrefix=experimentPrefix, marker=marker, xLim=TrainStepsXLim,
                                               yLim=self.accuracyYLim, plotVariance=plotVariance)
        self.plotTrainStepsDevelopmentAcc(cfgNames, plot=axarr[1, 1], dstDir=dstDir, experimentPrefix=experimentPrefix,
                                          marker=marker, xLim=TrainStepsXLim, yLim=self.accuracyYLim,
                                          plotVariance=plotVariance)

        fig, subplot = plt.subplots(1, 1)
        self.plotTrainStepsTestAccuracy(cfgNames, plot=subplot, title='Gaussian, 6 prototypes', dstDir=dstDir, experimentPrefix=experimentPrefix,
                                        marker=marker, xLim=TrainStepsXLim, yLim=self.accuracyYLim,
                                        plotVariance=plotVariance, savePlot=True, figure=fig)
        plt.show()

    '''
    def plotNaive(cls, protoSteps, titlePrefix, marker='', plotVariance=False):
        ax = plt.subplot()
        fontP = FontProperties()
        fontP.set_size('x-small')

        accuracyYLim = [0., 1]

        protosAll = self.ComplexitySizeAll['naive']
        protoMin = sys.maxint
        protoMax = -1
        for protos in protosAll:
            protoMin = min(np.min(protos), protoMin)
            protoMax = max(np.max(protos), protoMax)

        xticks = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, protoMax]

        # XXVL iwas stimmt hier net self.plotIntern(criterias, ax, self.ProtoCountsAll,
        self.ProtoTestRatesAll, protoSteps, marker, '-', titlePrefix + ' - Testdata',
        yLim= accuracyYLim, xLim=[0, protoMax], plotVariance=plotVariance)
        ax.legend(loc=1, prop=fontP)
        ax.set_xlabel('# Prototypes')
        ax.set_ylabel('Accuracy')
        ax.xaxis.set_ticks(xticks)'''

    def plotTrainStepsTestAccuracy(self, cfgNames, plot, title='', dstDir=None, experimentPrefix=None, marker='', xLim=None,
                                   yLim=None, plotVariance=False, savePlot=False, figure=None):
        # print experimentPrefix
        #print marker
        #fig, ax = plt.subplots(1, 1)
        #fig, ax = plt.subplots(1, 1)
        #plot = ax
        StatisticsPlot.plotIntern(cfgNames, plot, self.TrainStepCountAll, self.TrainStepCountTestAccAll,
                                  self.recordIntervall, marker, '-', title, yLim=yLim,
                                  xLim=xLim, plotVariance=plotVariance)
        plot.set_xlabel('# Training Samples')
        plot.set_ylabel('Accuracy')
        fontP = FontProperties()
        fontP.set_size('large')
        plot.legend(loc=0, prop=fontP)
        if savePlot and dstDir != None:
            figure.savefig(dstDir + experimentPrefix + '_accuracy.pdf', bbox_inches='tight')

    def plotTrainStepsTestCost(self, cfgNames, plot, dstDir=None, experimentPrefix=None, marker='', xLim=None,
                               yLim=None, plotVariance=False):
        # print experimentPrefix
        #print marker
        #fig, ax = plt.subplots(1, 1)
        #fig, ax = plt.subplots(1, 1)
        #plot = ax
        StatisticsPlot.plotIntern(cfgNames, plot, self.TrainStepCountAll, self.TrainStepCountTestCostAll,
                                  self.recordIntervall, marker, '-', 'Test-Cost',
                                  yLim=yLim, xLim=xLim, plotVariance=plotVariance)
        plot.set_xlabel('# Training Samples')
        plot.set_ylabel('Accuracy')
        fontP = FontProperties()
        fontP.set_size('large')
        plot.legend(loc=0, prop=fontP)
        #if dstDir != None:
        #    fig.savefig(dstDir + experimentPrefix + '_accuracy.pdf', bbox_inches='tight')

    def plotTrainStepsDevelopmentAcc(self, cfgNames, plot, dstDir=None, experimentPrefix=None, marker='', xLim=None,
                                     yLim=None, plotVariance=False):
        # print experimentPrefix
        #print marker
        #fig, ax = plt.subplots(1, 1)
        #fig, ax = plt.subplots(1, 1)
        #plot = ax
        StatisticsPlot.plotIntern(cfgNames, plot, self.TrainStepCountAll, self.TrainStepCountDevelopmentAccAll,
            self.recordIntervall, marker, '-', 'Dev-Acc', yLim=yLim, xLim=xLim, plotVariance=plotVariance)
        plot.set_xlabel('# Training Samples')
        plot.set_ylabel('Accuracy')
        fontP = FontProperties()
        fontP.set_size('large')
        plot.legend(loc=0, prop=fontP)
        #if dstDir != None:
        #    fig.savefig(dstDir + experimentPrefix + '_accuracy.pdf', bbox_inches='tight')

    def plotRejections(self, cfgNames, plot, dstDir=None, experimentPrefix=None, marker='', yLim=None,
                       plotVariance=False):
        print 'rejections'
        # fig, ax = plt.subplots(1, 1)
        #plot = ax
        filteredCfgNames = copy.deepcopy(cfgNames)
        filteredCfgNames.remove('SVM')
        StatisticsPlot.plotIntern(filteredCfgNames, plot, self.portionRejectionAll, self.classRateRejectionAll,
                                  0.02, marker, '-', '', xLim=[0, 1], yLim=yLim, plotVariance=plotVariance)
        plot.set_xlabel('$|X_\Theta|/|X|$')
        plot.invert_xaxis()
        #plot.set_ylabel('Accuracy')
        fontP = FontProperties()
        fontP.set_size('medium')
        #plot.legend(loc=3, prop = fontP)
        #if dstDir != None:
        #    fig.savefig(dstDir + experimentPrefix + '_rejection.pdf',  bbox_inches='tight')

    def plotTrainStepsTrainAccuracy(self, cfgNames, plot, dstDir=None, experimentPrefix=None, marker='', xLim=None,
                                    yLim=None, plotVariance=False):
        # fig, ax = plt.subplots(1, 1)
        #plot = ax
        StatisticsPlot.plotIntern(cfgNames, plot, self.TrainStepCountAll, self.TrainStepCountTrainAccAll,
                                  self.recordIntervall, marker, '-', 'Train-Ac', yLim=yLim, xLim=xLim,
                                  plotVariance=plotVariance)
        plot.set_xlabel('# Training Samples')
        #if dstDir != None:
        #    fig.savefig(dstDir + experimentPrefix + '_accuracyTrain.pdf',  bbox_inches='tight')

    def plotTrainStepsTrainCost(self, cfgNames, plot, dstDir=None, experimentPrefix=None, marker='', xLim=None,
                                yLim=None, plotVariance=False):
        # fig, ax = plt.subplots(1, 1)
        #plot = ax
        StatisticsPlot.plotIntern(cfgNames, plot, self.TrainStepCountAll, self.TrainStepCountTrainCostAll,
                                  self.recordIntervall, marker, '-', 'Train-Cost', yLim=yLim,
                                  xLim=xLim, plotVariance=plotVariance)
        plot.set_xlabel('# Training Samples')
        #if dstDir != None:
        #    fig.savefig(dstDir + experimentPrefix + '_accuracyTrain.pdf',  bbox_inches='tight')

    def plotTrainStepsInternTrainAccuracy(self, cfgNames, plot, dstDir=None, experimentPrefix=None, marker='',
                                          xLim=None, yLim=None, plotVariance=False):
        # fig, ax = plt.subplots(1, 1)
        #plot = ax
        StatisticsPlot.plotIntern(cfgNames, plot, self.TrainStepCountAll, self.TrainStepCountInternSampleAccAll,
                                    self.recordIntervall, marker, '-', 'InternSample-Ac', yLim=yLim, xLim=xLim,
                                    plotVariance=plotVariance)
        plot.set_xlabel('# Training Samples')
        #if dstDir != None:
        #    fig.savefig(dstDir + experimentPrefix + '_accuracyTrain.pdf',  bbox_inches='tight')

    def plotTrainStepsInternTrainCost(self, cfgNames, plot, dstDir=None, experimentPrefix=None, marker='', xLim=None,
                                      yLim=None, plotVariance=False):
        # fig, ax = plt.subplots(1, 1)
        #plot = ax
        StatisticsPlot.plotIntern(cfgNames, plot, self.TrainStepCountAll, self.TrainStepCountInternSampleAccAll,
                        self.recordIntervall, marker, '-', 'InternSample-Cost', yLim=yLim, xLim=xLim,
                        plotVariance=plotVariance)
        plot.set_xlabel('# Training Samples')
        #if dstDir != None:
        #    fig.savefig(dstDir + experimentPrefix + '_accuracyTrain.pdf',  bbox_inches='tight')

    def plotTrainStepsPrototypesAmount(self, cfgNames, plot, dstDir=None, experimentPrefix=None, marker='', xLim=None,
                                       plotVariance=False):
        # fig, ax = plt.subplots(1, 1)
        #plot = ax
        StatisticsPlot.plotIntern(cfgNames, plot, self.ComplexitySizeTrainStepCountAll, self.ComplexitySizeAll,
                        self.recordIntervall * 5, marker, '-', 'Prototypes', xLim=xLim, plotVariance=plotVariance)
        plot.set_ylabel('# Prototypes')
        plot.set_xlabel('# Training Samples')
        plot.set_xlim(xLim)
        fontP = FontProperties()
        fontP.set_size('medium')
        plot.legend(loc=0, prop=fontP)
        #if dstDir != None:
        #    fig.savefig(dstDir + experimentPrefix + '_growing.pdf',  bbox_inches='tight')

    @staticmethod
    def plotIntern(cfgNames, ax, XAll, YAll, XStepSize, marker, lineStyle, title, xLim=None, yLim=None,
                   plotVariance=False):
        cIdx = 0
        for cfgName in cfgNames:
            X, Y, YVar = StatisticsLogger.getMeanAndVarianceNested(XAll[cfgName], YAll[cfgName], XStepSize)
            ax.plot(X, Y, label=cfgName, color=GLVQPlot.getDefColors()[cIdx], marker=marker, linestyle=lineStyle)
            if plotVariance:
                ax.fill_between(X, Y + YVar, Y - YVar, facecolor=GLVQPlot.getDefColors()[cIdx], alpha=0.5)

            cIdx += 1
        ax.set_title(title)
        if not (xLim is None):
            ax.set_xlim(xLim)
        if not (yLim is None):
            ax.set_ylim(yLim)
        return ax
