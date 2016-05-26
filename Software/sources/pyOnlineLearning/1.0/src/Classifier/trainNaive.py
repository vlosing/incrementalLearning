import numpy

from NaiveNNClassifier import NaiveNNClassifier
from StatisticsLogger import StatisticsLogger


def coilExperiment(iterations, doStatistics, trainPortionStart, stepSize):
    statistics = StatisticsLogger(doStatistics)

    for iteration in range(iterations):

        statistics.newIteration(None, None, None, None)

        statistics.newCfgIteration('Naive')
        trainPortion = trainPortionStart
        # while trainPortion < 72:
        while trainPortion == trainPortionStart:
            config = dataPrep.DataSetConfig('COILRGB', 'randomRegular', trainPortion, iteration)
            classifier = NaiveNNClassifier()

            for i in range(len(config.TrainLabels)):
                classifier.learn(config.TrainSamples[i, :], config.TrainLabels[i], False)
            classifier.initTree()
            if len(config.TestLabels) == 0:
                break
            resultLabels = []
            for i in range(len(config.TestLabels)):
                resultLabels.append(classifier.classify(config.TestSamples[i, :]))
            correctCount = len(numpy.where(resultLabels == config.TestLabels)[0])
            testRate = correctCount / float(len(config.TestLabels))
            print str(testRate) + " correct on test"

            statistics.newTrainStep2(len(config.TrainLabels), testRate)
            trainPortion += stepSize
        statistics.finishCfgIteration2()
        statistics.finishIteration()

    statistics.serialize('coil_' + 'randomRegular')


def experiment(dataSet, dataSetOrder, iterations, doStatistics, trainPortionStart, stepSize):
    statistics = StatisticsLogger(doStatistics)

    for iteration in range(iterations):
        experimentPrefix = (dataSet + '_' + dataSetOrder + '_' + str(trainPortionStart)).replace('.', '')
        config = dataPrep.DataSetConfig(dataSet, dataSetOrder, 1, experimentPrefix)
        statistics.newIteration(None, None, None, None)

        statistics.newCfgIteration('naive')
        trainPortion = trainPortionStart
        while trainPortion < 1:
            classifier = NaiveNNClassifier()
            length = round(len(config.TrainLabels) * trainPortion)
            TrainLabels = numpy.empty(shape=(0, 1))
            TrainSamples = numpy.empty(shape=(0, config.dataDimensions))
            TestLabels = numpy.empty(shape=(0, 1))
            TestSamples = numpy.empty(shape=(0, config.dataDimensions))

            TrainSamples = numpy.vstack([TrainSamples, config.TrainSamples[0:length, :]])
            TrainLabels = numpy.append(TrainLabels, config.TrainLabels[0:length])
            TestSamples = numpy.vstack([TestSamples, config.TrainSamples[length:, :]])
            TestLabels = numpy.append(TestLabels, config.TrainLabels[length:])
            if len(TestLabels) == 0:
                break
            for i in range(len(TrainLabels)):
                classifier.learn(TrainSamples[i, :], TrainLabels[i], False)
            classifier.initTree()

            resultLabels = []
            for i in range(len(TestLabels)):
                resultLabels.append(classifier.classify(TestSamples[i, :]))
            correctCount = len(numpy.where(resultLabels == TestLabels)[0])
            testRate = correctCount / float(len(TestLabels))
            print str(testRate) + " correct on test"

            statistics.newTrainStep2(len(TrainLabels), testRate)
            trainPortion += stepSize
        statistics.finishCfgIteration2()

    statistics.serialize(dataSet + '_' + dataSetOrder)


if __name__ == '__main__':
    iterations = 1
    doStatistics = False
    # experiment('OutdoorEasy', 'random', iterations, doStatistics, 0.7, 1)
    #experiment('OutdoorEasy', 'chunksRandomEqualCountPerLabel', iterations, doStatistics, 0.7, 1)
    coilExperiment(iterations, doStatistics, 18, 1)

    #experiment('toy1', 'random', iterations, doStatistics, 0.01, 0.01)
    #experiment('toy2', 'random', iterations, doStatistics, 0.01, 0.01)
    #experiment('toy3', 'random', iterations, doStatistics, 0.01, 0.01)
    #experiment('toyComplete', 'random', iterations, doStatistics, 0.01, 0.01)


