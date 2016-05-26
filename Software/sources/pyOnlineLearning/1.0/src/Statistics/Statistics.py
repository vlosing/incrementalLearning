__author__ = 'vlosing'
import numpy as np

from StatisticsLogger import StatisticsLogger
from StatisticsPlot import StatisticsPlot
from StatisticsWriter import StatisticsWriter
from Experiment.LVQExperiment import LVQExperiment
from Base import Paths


class StatisticList:
    def __init__(self, cfgNames, recordInterval, srcDir, dstDir):
        self.cfgNames = cfgNames
        self.recordInterval = recordInterval
        self.srcDir = srcDir
        self.dstDir = dstDir
        self.expData = []

    def addExpStat(self, trainingSetCfg, accuracyYLim=[0., 1]):
        statsLogger = StatisticsLogger(generateRejectionData=False)
        experimentPrefix = LVQExperiment.getExpPrefix(None, trainingSetCfg)
        statsLogger.deSerialize(cfgNames, experimentPrefix, path=self.srcDir)

        self.expData.append([statsLogger, trainingSetCfg, accuracyYLim])

    def plot(self):
        for datum in self.expData:
            statsPlot = StatisticsPlot(datum[0].trainStepCounts.itemAll, datum[0].trainStepCountTestAcc.itemAll,
                                       datum[0].trainStepCountTrainAcc.itemAll,
                                       datum[0].trainStepCountInternSampleAcc.itemAll,
                                       datum[0].trainStepCountDevelopmentAcc.itemAll,
                                       datum[0].trainStepCountTestCost.itemAll,
                                       datum[0].trainStepCountTrainCost.itemAll,
                                       datum[0].trainStepCountInternSampleCost.itemAll,
                                       datum[0].complexitySizeTrainStepCount.itemAll,
                                       datum[0].complexitySize.itemAll, datum[0].portionRejectionAll,
                                       datum[0].classRateRejectionAll,
                                       self.recordInterval,
                                       accuracyYLim=datum[2],
                                       title=datum[1]['dsName'])
            experimentPrefix = LVQExperiment.getExpPrefix(None, datum[1])
            statsPlot.plotAll(cfgNames, self.dstDir, experimentPrefix, plotVariance=False)

    def writeStats(self):
        allStats = {}
        for datum in self.expData:
            statsLogger = datum[0]
            experimentPrefix = LVQExperiment.getExpPrefix(None, datum[1])
            statsWriter = StatisticsWriter(statsLogger.trainStepCounts.itemAll,
                                           statsLogger.trainStepCountTestAcc.itemAll,
                                           statsLogger.trainStepCountTrainAcc.itemAll,
                                           statsLogger.trainStepCountDevelopmentAcc.itemAll,
                                           statsLogger.complexitySizeTrainStepCount.itemAll,
                                           statsLogger.complexitySize.itemAll,
                                           statsLogger.trainStepCountComplexitySize.itemAll,
                                           self.recordInterval)

            statList1, statList2, statList3, statList4 = statsWriter.saveFinalStatsToCSV(cfgNames, self.dstDir,
                                                                                         experimentPrefix)
            for i in range(len(statList1)):
                cfgName = statList1[i][0]
                entry = np.array([statList1[i][1] * 100, statList2[i][1] * 100, statList3[i][1] * 100, statList4[i][1]])
                if cfgName in allStats.keys():
                    allStats[cfgName] = np.vstack([allStats[cfgName], entry])
                else:
                    allStats[cfgName] = entry
        np.set_printoptions(precision=1)
        for key in allStats.keys():
            print key, np.average(np.atleast_2d(allStats[key]), axis=0)

if __name__ == '__main__':
    #cfgNames = ['no updates', 'standard', 'WU-1', 'WU-5', 'WU-10', 'WU-20']
    cfgNames = ['no updates', 'standard', 'window-100', 'window-500', 'window-1000', 'window-2000', 'window-4000', 'window-10000']
    #cfgNames = ['no-training', 'retraining-0', 'retraining-5']
    #path = Paths.StatisticsClassificationDir()
    #path = Paths.StatisticsClassificationDir() + 'Retraining/Gaussian-random-insertions/'
    #path = Paths.StatisticsClassificationDir() + 'Retraining/Gaussian-random/'
    #path = Paths.StatisticsClassificationDir() + 'Retraining/Gaussian-Sequential/'
    path = Paths.StatisticsClassificationDir() + 'Retraining/Gaussian-Sequential-WindowSizes/'

    #path = Paths.StatisticsClassificationDir() + 'Retraining/USPS-sequential/'
    #path = Paths.StatisticsClassificationDir() + 'Retraining/USPS-sequential-windows/'




    #path = Paths.StatisticsClassificationDir() + 'Retraining/USPS-random-insertions/'
    #path = Paths.StatisticsClassificationDir() + 'Retraining/USPS-random/'
    statsList = StatisticList(cfgNames, 50, path, path)

    #statsList.addExpStat({'dsName': 'Gaussian', 'trainOrder': 'random', 'trainSetSize': 0.7, 'expName': 'Gaussian_chunks50'}, accuracyYLim = [0.1, 1.0])

    #statsList.addExpStat({'dsName': 'Gaussian', 'trainOrder': 'random', 'trainSetSize': 0.7, 'expName': 'Gaussian_random_6Protos'}, accuracyYLim = [0.7, 1.0])
    #statsList.addExpStat({'dsName': 'Gaussian', 'trainOrder': 'random', 'trainSetSize': 0.7, 'expName': 'Gaussian_random_12Protos'}, accuracyYLim = [0.7, 1.0])
    #statsList.addExpStat({'dsName': 'Gaussian', 'trainOrder': 'random', 'trainSetSize': 0.7, 'expName': 'Gaussian_random_24Protos'}, accuracyYLim = [0.7, 1.0])

    #statsList.addExpStat({'dsName': 'Gaussian', 'trainOrder': 'random', 'trainSetSize': 0.7, 'expName': 'Gaussian_orderedByLabel_6Protos'}, accuracyYLim = [0.1, 1.0])
    #statsList.addExpStat({'dsName': 'Gaussian', 'trainOrder': 'random', 'trainSetSize': 0.7, 'expName': 'Gaussian_orderedByLabel_12Protos'}, accuracyYLim = [0.1, 1.0])
    #statsList.addExpStat({'dsName': 'Gaussian', 'trainOrder': 'random', 'trainSetSize': 0.7, 'expName': 'Gaussian_orderedByLabel_24Protos'}, accuracyYLim = [0.1, 1.0])

    statsList.addExpStat({'dsName': 'Gaussian', 'trainOrder': 'orderedByLabel', 'trainSetSize': 0.7, 'expName': 'Gaussian_orderedByLabel_6Protos_windows'}, accuracyYLim = [0.1, 1.0])

    #statsList.addExpStat({'dsName': 'USPS', 'trainOrder': 'orderedByLabel', 'trainSetSize': 0.7, 'expName': 'USPS_random_10Protos'}, accuracyYLim = [0.55, 1.0])
    #statsList.addExpStat({'dsName': 'USPS', 'trainOrder': 'orderedByLabel', 'trainSetSize': 0.7, 'expName': 'USPS_random_30Protos'}, accuracyYLim = [0.6, 1.0])
    #statsList.addExpStat({'dsName': 'USPS', 'trainOrder': 'orderedByLabel', 'trainSetSize': 0.7, 'expName': 'USPS_random_60Protos'}, accuracyYLim = [0.6, 1.0])

    #statsList.addExpStat({'dsName': 'USPS', 'trainOrder': 'orderedByLabel', 'trainSetSize': 0.7, 'expName': 'USPS_orderedByLabel_10Protos'}, accuracyYLim = [0.1, 1.0])
    #statsList.addExpStat({'dsName': 'USPS', 'trainOrder': 'orderedByLabel', 'trainSetSize': 0.7, 'expName': 'USPS_orderedByLabel_30Protos'}, accuracyYLim = [0.1, 1.0])
    #statsList.addExpStat({'dsName': 'USPS', 'trainOrder': 'orderedByLabel', 'trainSetSize': 0.7, 'expName': 'USPS_orderedByLabel_60Protos'}, accuracyYLim = [0.1, 1.0])

    #statsList.addExpStat({'dsName': 'USPS', 'trainOrder': 'orderedByLabel', 'trainSetSize': 0.7, 'expName': 'USPS_orderedByLabel_10Protos_windows'}, accuracyYLim = [0.1, 1.0])



    #statsList.addExpStat({'dsName': 'USPS', 'trainOrder': 'orderedByLabel', 'trainSetSize': 0.7, 'expName': 'USPS_chunksRandom_07'}, accuracyYLim = [0.6, 1.0])


    #statsList.addExpStat({'name': 'COIL', 'trainOrder': 'cv', 'trainPortionValue': 18, 'expName': 'COIL_insert5'}, accuracyYLim=[0.6, 1.0])
    #statsList.addExpStat({'name': 'Border', 'trainOrder': 'random', 'trainPortionValue': 0.7},
    #accuracyYLim = [0.7, 1.0])
    '''statsList.addExpStat({'name': 'Overlap', 'trainOrder': 'random', 'trainPortionValue': 0.7},
    accuracyYLim = [0.3, 0.9])
    statsList.addExpStat({'name': 'Noise', 'trainOrder': 'random', 'trainPortionValue': 0.7},
    accuracyYLim = [0.7, 1.0])
    statsList.addExpStat({'name': 'Chess', 'trainOrder': 'random', 'trainPortionValue': 0.7},
    accuracyYLim = [0.7, 1.0])'''

    '''statsList.addExpStat({'name': 'USPSOrg', 'trainOrder': 'shuffle', 'trainPortionValue': 0.7},
                         accuracyYLim=[0.7, 1.0])
    statsList.addExpStat({'name': 'DNAOrg', 'trainOrder': 'shuffle', 'trainPortionValue': 0.7}, accuracyYLim=[0.7, 1.0])
    statsList.addExpStat({'name': 'LetterOrg', 'trainOrder': 'shuffle', 'trainPortionValue': 0.7},
                         accuracyYLim=[0.7, 1.0])
    statsList.addExpStat({'name': 'PenDigitsOrg', 'trainOrder': 'shuffle', 'trainPortionValue': 0.7},
                         accuracyYLim=[0.7, 1.0])
    statsList.addExpStat({'name': 'OptDigits', 'trainOrder': 'random', 'trainPortionValue': 0.2136},
                         accuracyYLim=[0.7, 1.0])

    statsList.addExpStat({'name': 'OutdoorEasy', 'trainOrder': 'random', 'trainPortionValue': 0.7},
                         accuracyYLim=[0.7, 1.0])
    statsList.addExpStat(
        {'name': 'OutdoorEasy', 'trainOrder': 'chunksRandomEqualCountPerLabel', 'trainPortionValue': 0.7},
        accuracyYLim=[0.7, 1.0])'''
    #statsList.addExpStat({'name': 'COIL', 'trainOrder': 'randomRegular', 'trainPortionValue': 18},
    #                     accuracyYLim=[0.7, 1.0])
    statsList.plot()
    statsList.writeStats()