__author__ = 'vlosing'
import csv

import numpy as np

from StatisticsLogger import StatisticsLogger


class StatisticsWriter:
    def __init__(self, TrainStepCountAll, TrainStepCountTestAccAll, TrainStepCountTrainAccAll,
                 TrainStepCountDevelopmentAccAll, ComplexitySizeTrainStepCountAll, ComplexitySizeAll,
                 TrainStepCountComplexitySizeAll, evalIntervall):
        self.TrainStepCountAll = TrainStepCountAll
        self.TrainStepCountTestAccAll = TrainStepCountTestAccAll
        self.TrainStepCountTrainAccAll = TrainStepCountTrainAccAll
        self.TrainStepCountDevelopmentAccAll = TrainStepCountDevelopmentAccAll
        self.ComplexitySizeTrainStepCountAll = ComplexitySizeTrainStepCountAll
        self.ComplexitySizeAll = ComplexitySizeAll
        self.TrainStepCountComplexitySizeAll = TrainStepCountComplexitySizeAll
        self.evalIntervall = evalIntervall

    @staticmethod
    def getFinalStats(cfgNames, XAll, YAll, XStepSize):
        statList = []
        for cfgName in cfgNames:
            X, Y, YVar = StatisticsLogger.getMeanAndVarianceNested(XAll[cfgName], YAll[cfgName], XStepSize)
            statList.append([cfgName, Y[-1], YVar[-1]])
        return statList

    def getStatsToTrainStep(self, step, cfgNames, YAll, XStepSize):
        statList = []
        for cfgName in cfgNames:
            X, Y, YVar = StatisticsLogger.getMeanAndVarianceNested(self.TrainStepCountAll[cfgName], YAll[cfgName], XStepSize)
            idx = np.where(X == step)[0]
            statList.append([cfgName, Y[idx][0], YVar[idx][0]])
        return statList

    def saveFinalStatsToCSV(self, cfgNames, dstDir, experimentPrefix):
        statList1 = StatisticsWriter.getFinalStats(cfgNames, self.TrainStepCountAll, self.TrainStepCountTestAccAll,
                                       self.evalIntervall)
        statList2 = StatisticsWriter.getFinalStats(cfgNames, self.TrainStepCountAll, self.TrainStepCountTrainAccAll,
                                       self.evalIntervall)
        statList3 = StatisticsWriter.getFinalStats(cfgNames, self.TrainStepCountAll, self.TrainStepCountDevelopmentAccAll,
                                       self.evalIntervall)
        statList4 = StatisticsWriter.getFinalStats(cfgNames, self.ComplexitySizeTrainStepCountAll, self.ComplexitySizeAll,
                                       self.evalIntervall)

        csvfile = open(dstDir + experimentPrefix + '_Stats.csv', 'wb')
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(['', 'TestAcc', 'TestAcc-Var', 'TrainAcc', 'TrainAcc-Var', 'Size', 'Size-Var'])
        writer.writerow(['', 'Test-Accuracy', 'Train-Accuracy', 'Dev-Accuracy', 'Nodes'])
        for i in range(len(statList1)):
            writer.writerow(
                [statList1[i][0], "{0:.2f}".format(statList1[i][1] * 100), "{0:.2f}".format(statList2[i][1] * 100),
                 "{0:.2f}".format(statList3[i][1] * 100), "{0:.2f}".format(statList4[i][1])])
        return statList1, statList2, statList3, statList4

    def saveStatsInTrainStepIntervallsToCSV(self, startStep, intervall, cfgNames, dstDir, experimentPrefix):
        currentStep = startStep
        maxTrainStep = np.max(self.TrainStepCountAll[cfgNames[0]])
        csvfile = open(dstDir + experimentPrefix + '_StatsIntervall.csv', 'wb')
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Algorithm', 'Step', 'Test-Accuracy', 'Train-Accuracy', 'Nodes'])
        while currentStep < maxTrainStep:
            statList1 = self.getStatsToTrainStep(currentStep, cfgNames, self.TrainStepCountTestAccAll,
                                                 self.evalIntervall)
            statList2 = self.getStatsToTrainStep(currentStep, cfgNames, self.TrainStepCountComplexitySizeAll,
                                                 self.evalIntervall)
            for i in range(len(statList1)):
                writer.writerow([statList1[i][0], currentStep, statList1[i][1] * 100, 0, statList2[i][1]])
            currentStep += intervall
            currentStep = min(currentStep, maxTrainStep)
        statList1 = self.getStatsToTrainStep(currentStep, cfgNames, self.TrainStepCountTestAccAll, self.evalIntervall)
        statList2 = self.getStatsToTrainStep(currentStep, cfgNames, self.TrainStepCountComplexitySizeAll,
                                             self.evalIntervall)
        for i in range(len(statList1)):
            writer.writerow([statList1[i][0], currentStep, statList1[i][1] * 100, 0, statList2[i][1]])
