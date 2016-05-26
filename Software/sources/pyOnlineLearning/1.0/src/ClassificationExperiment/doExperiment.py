__author__ = 'vlosing'
from ExperimentListener import DummyExpListener
from Statistics.StatisticsLogger import StatisticsLogger
from Experiment import Experiment


def doExperiment(trainingSetCfg, expCfg, cfgs):
    statsLogger = None
    statisticsRecordIntervall = expCfg['statisticsRecordIntervall']
    if expCfg['statisticsLevel'] == 0:
        statsLogger = DummyExpListener()
    elif expCfg['statisticsLevel'] == 1:
        statsLogger = StatisticsLogger(doTrainStepStatistics=False, doPrototypeStatistics=False, recordIntervall=statisticsRecordIntervall)
    elif expCfg['statisticsLevel'] == 2:
        statsLogger = StatisticsLogger(doTrainStepStatistics=True, doPrototypeStatistics=False, recordIntervall=statisticsRecordIntervall)
    elif expCfg['statisticsLevel'] == 3:
        statsLogger = StatisticsLogger(doTrainStepStatistics=True, doPrototypeStatistics=True, recordIntervall=statisticsRecordIntervall)
    experiment = Experiment(trainingSetCfg, expCfg, cfgs, statsLogger)
    experiment.execute()
    return experiment.classifiers
