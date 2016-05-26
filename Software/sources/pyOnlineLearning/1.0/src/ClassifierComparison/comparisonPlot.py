from auxiliaryFunctions import getTrainSetCfg
from DataGeneration.DataSetFactory import isStationary
import matplotlib.pyplot as plt
from Visualization import GLVQPlot
from Base import Serialization
from ClassificationExperiment.Experiment import Experiment
import numpy as np
from Base import Paths
import csv
import unicodedata


def getColorIdxToClassifier(classifierName):
    return ['ISVM', 'ILVQ', 'ORF', 'LPP', 'LPPNSE', 'IELM', 'SGD', 'GNB', 'LVGB'].index(unicodedata.normalize('NFKD', classifierName).encode('ascii','ignore'))
    #return 0

def getLineStyleToClassifier(classifierName):
    '''if unicodedata.normalize('NFKD', classifierName).encode('ascii','ignore') in ['iSVM', 'iLVQ']:
        return '-'
    else:
        return '--'
        '''
    return '-'

def getMarkerToClassifier(classifierName):
    return ''
    '''if unicodedata.normalize('NFKD', classifierName).encode('ascii','ignore') in ['iSVM']:
        return ''
    elif unicodedata.normalize('NFKD', classifierName).encode('ascii','ignore') in ['iLVQ']:
        return '^'
    elif unicodedata.normalize('NFKD', classifierName).encode('ascii','ignore') in ['iELM']:
        return 'o'
    elif unicodedata.normalize('NFKD', classifierName).encode('ascii','ignore') in ['LPP']:
        return 's'''''

def plotSingleAccuracies(dataSetName, classifierEvaluations, criterionName, chunkSize, numTrainSamples, streamSetting, evalFilePrefix, dstDirectory, accuracyName, classifierNames=None):
    subPlots = []
    figures = []
    for j in range(len(classifierEvaluations['values'][classifierEvaluations['values'].keys()[0]].keys())):
        fig, subplot = plt.subplots(1, 1)
        subPlots.append(subplot)
        figures.append(fig)
    for i in range(len(classifierEvaluations['values'].keys())):
        classifierName = classifierEvaluations['values'].keys()[i]
        if not classifierNames or classifierName in classifierNames:
            for j in range(len(classifierEvaluations['values'][classifierName].keys())):
                criterion = classifierEvaluations['values'][classifierName].keys()[j]
                if classifierEvaluations['values'][classifierName][criterion].has_key(accuracyName):
                    accuracies = []
                    for iteration in range(len(classifierEvaluations['values'][classifierName][criterion][accuracyName])):
                        accuracies.append(classifierEvaluations['values'][classifierName][criterion][accuracyName][iteration])
                    accuracies = np.mean(accuracies, axis=0)
                    if criterionName == 'chunkSize':
                        chunkSize = int(criterion)
                    if streamSetting:
                        xValues = np.arange(chunkSize, numTrainSamples, chunkSize)
                    else:
                        xValues = np.arange(chunkSize, numTrainSamples + 1, chunkSize)
                        if xValues[-1] != numTrainSamples:
                            xValues = np.append(xValues, numTrainSamples)
                    subPlots[j].plot(xValues, accuracies, label=classifierName, color=GLVQPlot.getDefColors()[getColorIdxToClassifier(classifierName)])
                    #print np.mean(accuracies)
                    #print accuracies[-1]
                    subPlots[j].set_title(dataSetName + ' - ' + ('test' if accuracyName == 'testAccuracies' else 'prediction') + ' accuracy ' + criterion)
                    subPlots[j].set_xlabel('#Samples')
                    subPlots[j].set_ylabel('Accuracy')
                    subPlots[j].legend()

    for i in range(len(figures)):
        figures[i].savefig(dstDirectory + evalFilePrefix + '_' + accuracyName + str(i) + '.pdf', bbox_inches='tight')

def plotSingleComplexities(dataSetName, classifierEvaluations, criterionName, chunkSize, numTrainSamples, evalFilePrefix, dstDirectory, classifierNames=None):
    subPlots = []
    figures = []
    for j in range(len(classifierEvaluations['values'][classifierEvaluations['values'].keys()[0]].keys())):
        fig, subplot = plt.subplots(1, 1)
        subPlots.append(subplot)
        figures.append(fig)

    for i in range(len(classifierEvaluations['values'].keys())):
        classifierName = classifierEvaluations['values'].keys()[i]
        if not classifierNames or classifierName in classifierNames:
            for j in range(len(classifierEvaluations['values'][classifierName].keys())):
                criterion = classifierEvaluations['values'][classifierName].keys()[j]
                if classifierEvaluations['values'][classifierName][criterion].has_key('complexitiesNumParamMetric'):
                    complexities = []
                    for iteration in range(len(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'])):
                        complexities.append(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'][iteration])
                    complexities = np.mean(complexities, axis=0)
                    if criterionName == 'chunkSize':
                        chunkSize = int(criterion)
                    xValues = np.arange(chunkSize, numTrainSamples + 1, chunkSize)
                    if xValues[-1] != numTrainSamples:
                        xValues = np.append(xValues, numTrainSamples)
                    subPlots[j].plot(xValues, complexities, label=classifierName, color=GLVQPlot.getDefColors()[getColorIdxToClassifier(classifierName)])
                    subPlots[j].set_title(dataSetName + ' - Model complexity ' + criterion)

                    subPlots[j].set_xlabel('#Samples')
                    subPlots[j].set_ylabel('#Parameter')
                    subPlots[j].legend()

    for i in range(len(figures)):
        figures[i].savefig(dstDirectory + evalFilePrefix + '_complexitiesNumParamMetric' + str(i) + '.pdf', bbox_inches='tight')

def plotChunkSizeAccuracies(dataSetName, classifierEvaluations, streamSetting, evalFilePrefix, dstDirectory, accuracyName, criteria, sortedIndices, classifierNames=None):
    fig, axTestAccuraciesChunkSizes = plt.subplots(1, 1)
    for i in range(len(classifierEvaluations['values'].keys())):
        classifierName = classifierEvaluations['values'].keys()[i]
        if not classifierNames or classifierName in classifierNames:
            criterionAccuracies = []
            for criterion in classifierEvaluations['values'][classifierName].keys():
                if classifierEvaluations['values'][classifierName][criterion].has_key(accuracyName):
                    accuracies = []
                    for iteration in range(len(classifierEvaluations['values'][classifierName][criterion][accuracyName])):
                        if streamSetting:
                            accuracies.append(np.mean(classifierEvaluations['values'][classifierName][criterion][accuracyName][iteration]))
                        else:
                            accuracies.append(classifierEvaluations['values'][classifierName][criterion][accuracyName][iteration][-1])
                    criterionAccuracies.append(np.mean(accuracies))
            axTestAccuraciesChunkSizes.plot(criteria, np.array(criterionAccuracies)[sortedIndices], label=classifierName, color=GLVQPlot.getDefColors()[getColorIdxToClassifier(classifierName)], linestyle=getLineStyleToClassifier(classifierName), marker=getMarkerToClassifier(classifierName), lw=2, ms=7)
            #axTestAccuraciesChunkSizes.set_title(dataSetName + ' - Chunk sizes vs. ' + ('test' if accuracyName == 'testAccuracies' else 'prediction') + ' accuracy')
            #axTestAccuraciesChunkSizes.set_title('Border')
            axTestAccuraciesChunkSizes.set_xlabel('window-/chunk size')
            axTestAccuraciesChunkSizes.set_ylabel('Accuracy')
            axTestAccuraciesChunkSizes.legend(loc=0)
    fig.savefig(dstDirectory + evalFilePrefix + '_' + accuracyName + 'ChunkSizes.jpg', bbox_inches='tight')

def plotChunkSizeComplexities(dataSetName, classifierEvaluations, evalFilePrefix, dstDirectory, criteria, sortedIndices, classifierNames=None):
    fig, axComplexitiesChunkSizes = plt.subplots(1, 1)
    for i in range(len(classifierEvaluations['values'].keys())):
        classifierName = classifierEvaluations['values'].keys()[i]
        if not classifierNames or classifierName in classifierNames:
            criterionComplexities = []
            for criterion in classifierEvaluations['values'][classifierName].keys():
                if classifierEvaluations['values'][classifierName][criterion].has_key('complexitiesNumParamMetric'):
                    complexities = []
                    for iteration in range(len(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'])):
                        complexities.append(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'][iteration][-1])
                    criterionComplexities.append(np.mean(complexities))
            axComplexitiesChunkSizes.semilogy(criteria, np.array(criterionComplexities)[sortedIndices], label=classifierName, color=GLVQPlot.getDefColors()[getColorIdxToClassifier(classifierName)], linestyle=getLineStyleToClassifier(classifierName), marker=getMarkerToClassifier(classifierName), lw=2, ms=7)
            #axComplexitiesChunkSizes.set_title(dataSetName + ' - Chunk sizes vs. Model complexity')
            axComplexitiesChunkSizes.set_xlabel('window-/chunk size')
            axComplexitiesChunkSizes.set_ylabel('#Parameter')
            #axComplexitiesChunkSizes.set_ylim([1000, 6000])
            #axComplexitiesChunkSizes.legend()
    fig.savefig(dstDirectory + evalFilePrefix + '_complexityNumParamMetricChunkSizes.jpg', bbox_inches='tight')

def plotComplexityAccuracies(dataSetName, classifierEvaluations, streamSetting, evalFilePrefix, dstDirectory, accuracyName, classifierNames=None):
    fig, axTestAccuraciesChunkSizes = plt.subplots(1, 1)
    for i in range(len(classifierEvaluations['values'].keys())):
        classifierName = classifierEvaluations['values'].keys()[i]
        if not classifierNames or classifierName in classifierNames:
            criterionAccuracies = []
            criterionComplexities = []
            for criterion in classifierEvaluations['values'][classifierName].keys():
                if classifierEvaluations['values'][classifierName][criterion].has_key(accuracyName):
                    accuracies = []
                    for iteration in range(len(classifierEvaluations['values'][classifierName][criterion][accuracyName])):
                        if streamSetting:
                            accuracies.append(np.mean(classifierEvaluations['values'][classifierName][criterion][accuracyName][iteration]))
                        else:
                            accuracies.append(classifierEvaluations['values'][classifierName][criterion][accuracyName][iteration][-1])
                    criterionAccuracies.append(np.mean(accuracies))

                if classifierEvaluations['values'][classifierName][criterion].has_key('complexitiesNumParamMetric'):
                    complexities = []
                    for iteration in range(len(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'])):
                        complexities.append(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'][iteration][-1])
                    criterionComplexities.append(np.mean(complexities))
            axTestAccuraciesChunkSizes.plot(criterionComplexities, criterionAccuracies, label=classifierName, c=GLVQPlot.getDefColors()[i])
            axTestAccuraciesChunkSizes.set_title(dataSetName + ' - ' + 'Model complexity vs. Accuracy')
            axTestAccuraciesChunkSizes.set_xlabel('#Parameter')
            axTestAccuraciesChunkSizes.set_ylabel('Accuracy')
            axTestAccuraciesChunkSizes.legend()
    fig.savefig(dstDirectory + evalFilePrefix + '_' + accuracyName + 'Complexity.pdf', bbox_inches='tight')

def comparisonPlots(dataSetName, classifierEvaluations, criterionName, chunkSize, numTrainSamples, streamSetting, evalFilePrefix, dstDirectory, classifierNames=None):
    if streamSetting:
        accuracyName = 'trainPredictionAccuracies'
    else:
        accuracyName = 'testAccuracies'
    criteria = np.array(classifierEvaluations['values'][classifierEvaluations['values'].keys()[0]].keys()).astype(int)
    sortedIndices = np.argsort(criteria)
    criteria = criteria[sortedIndices]

    plotSingleAccuracies(dataSetName, classifierEvaluations, criterionName, chunkSize, numTrainSamples, streamSetting, evalFilePrefix, dstDirectory, accuracyName, classifierNames=classifierNames)
    #plotSingleComplexities(dataSetName, classifierEvaluations, criterionName, chunkSize, numTrainSamples, evalFilePrefix, dstDirectory, classifierNames=classifierNames)
    if len(criteria) > 1:
        if criterionName == 'chunkSize':
            plotChunkSizeAccuracies(dataSetName, classifierEvaluations, streamSetting, evalFilePrefix, dstDirectory, accuracyName, criteria, sortedIndices, classifierNames=classifierNames)
            plotChunkSizeComplexities(dataSetName, classifierEvaluations, evalFilePrefix, dstDirectory, criteria, sortedIndices, classifierNames=classifierNames)
        elif criterionName == 'complexity':
            plotComplexityAccuracies(dataSetName, classifierEvaluations, streamSetting, evalFilePrefix, dstDirectory, accuracyName, classifierNames=classifierNames)

def doComparisonPlot(dataSetName, criterionName, classifierNames, chunkSize = 100):
    trainSetCfg = getTrainSetCfg(dataSetName)
    evalFilePrefix = Experiment.getExpPrefix2(None, trainSetCfg)
    evalDstDirectory = Paths.FeatureTmpsDirPrefix() + 'Evaluations/'

    classifierEvaluations = Serialization.loadJsonFile(evalDstDirectory + evalFilePrefix + 'evaluations.json')
    numTrainSamples = classifierEvaluations['meta']['numTrainSamples']
    comparisonPlots(dataSetName, classifierEvaluations, criterionName, chunkSize, numTrainSamples, not isStationary(dataSetName), evalFilePrefix, Paths.FeatureTmpsDirPrefix() + 'Plots/', classifierNames=classifierNames)
    dataToCSV(dataSetName)
    plt.show()

def getBestChunkSize(classifierEvaluations, classifierName, streamSetting):
    maxValue = 0
    bestChunkSize = 0
    if streamSetting:
        accuracyName = 'trainPredictionAccuracies'
    else:
        accuracyName = 'testAccuracies'
    for j in range(len(classifierEvaluations['values'][classifierName].keys())):
        criterion = classifierEvaluations['values'][classifierName].keys()[j]
        if classifierEvaluations['values'][classifierName][criterion].has_key(accuracyName):
            accuracies = []
            for iteration in range(len(classifierEvaluations['values'][classifierName][criterion][accuracyName])):
                accuracies.append(classifierEvaluations['values'][classifierName][criterion][accuracyName][iteration])
            if streamSetting:
                accuracy = np.mean(accuracies)
            else:
                accuracy = np.mean(accuracies, axis=0)[-1]
            if accuracy > maxValue:
                maxValue = accuracy
                bestChunkSize = int(criterion)
    return bestChunkSize, maxValue


def getSingleRunValues(classifierEvaluations, classifierName, bestChunkSize):
    accuracies = []
    meanAccs = []
    complexities = []
    for iteration in range(len(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['complexitiesNumParamMetric'])):
        complexities.append(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['complexitiesNumParamMetric'][iteration])
    complexities = np.array(complexities)
    meanEndComplexityNumParamMetric = np.mean(complexities, axis=0)[-1]
    stdEndComplexityNumParamMetric = np.std(complexities, axis=0)[-1]

    complexities = []
    for iteration in range(len(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['complexities'])):
        complexities.append(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['complexities'][iteration])
    complexities = np.array(complexities)
    meanEndComplexity = np.mean(complexities, axis=0)[-1]
    stdEndComplexity = np.std(complexities, axis=0)[-1]
    numTrainSamples = classifierEvaluations['meta']['numTrainSamples']
    if classifierEvaluations['values'][classifierName][str(bestChunkSize)].has_key('testAccuracies'):
        for iteration in range(len(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['testAccuracies'])):
            accuracies.append(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['testAccuracies'][iteration])
        xValues = np.arange(bestChunkSize, numTrainSamples + 1, bestChunkSize)
        if xValues[-1] != numTrainSamples:
            xValues = np.append(xValues, numTrainSamples)
    elif classifierEvaluations['values'][classifierName][str(bestChunkSize)].has_key('trainPredictionAccuracies'):
        for iteration in range(len(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['trainPredictionAccuracies'])):
            accuracies.append(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['trainPredictionAccuracies'][iteration])
            meanAccs.append(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['meanAcc'][iteration])
        xValues = np.arange(bestChunkSize, numTrainSamples, bestChunkSize)
    accuracies = np.array(accuracies)
    meanFirstTenthAcc = np.interp(0.1 * numTrainSamples, xValues, np.mean(accuracies, axis=0))
    stdFirstTenthAcc = np.interp(0.1 * numTrainSamples, xValues, np.std(accuracies, axis=0))
    meanFirstQuarterAcc = np.interp(0.25 * numTrainSamples, xValues, np.mean(accuracies, axis=0))
    stdFirstQuarterAcc = np.interp(0.25 * numTrainSamples, xValues, np.std(accuracies, axis=0))
    meanSecondQuarterAcc = np.interp(0.5 * numTrainSamples, xValues, np.mean(accuracies, axis=0))
    stdSecondQuarterAcc = np.interp(0.5 * numTrainSamples, xValues, np.std(accuracies, axis=0))
    meanThirdQuarterAcc = np.interp(0.75 * numTrainSamples, xValues, np.mean(accuracies, axis=0))
    stdThirdQuarterAcc = np.interp(0.75 * numTrainSamples, xValues, np.std(accuracies, axis=0))
    meanEndAccuracy = np.mean(accuracies, axis=0)[-1]
    stdEndAccuracy = np.std(accuracies, axis=0)[-1]
    if classifierEvaluations['values'][classifierName][str(bestChunkSize)].has_key('trainPredictionAccuracies'):
        meanAccuracy = np.mean(meanAccs)
        stdAccuracy = np.std(meanAccs)
    else:
        meanAccuracy = np.mean(accuracies)
        stdAccuracy = np.std(accuracies)

    return [meanFirstTenthAcc, stdFirstTenthAcc, meanFirstQuarterAcc, stdFirstQuarterAcc, meanSecondQuarterAcc, stdSecondQuarterAcc, meanThirdQuarterAcc, stdThirdQuarterAcc,
            meanEndAccuracy, stdEndAccuracy, meanAccuracy, stdAccuracy, meanEndComplexity, stdEndComplexity, meanEndComplexityNumParamMetric, stdEndComplexityNumParamMetric]


def getChunkValues(classifierEvaluations, classifierName):
    endAccuracies = []
    allAccuracies = []
    complexities = []
    complexitiesNumParamMetric = []
    for criterion in classifierEvaluations['values'][classifierName].keys():
        for iteration in range(len(classifierEvaluations['values'][classifierName][criterion]['complexities'])):
            if classifierEvaluations['values'][classifierName][criterion].has_key('testAccuracies'):
                allAccuracies = allAccuracies + classifierEvaluations['values'][classifierName][criterion]['testAccuracies'][iteration]
                endAccuracies.append(classifierEvaluations['values'][classifierName][criterion]['testAccuracies'][iteration][-1])
            else:
                allAccuracies = allAccuracies + classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'][iteration]
                endAccuracies.append(classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'][iteration][-1])
            complexities.append(classifierEvaluations['values'][classifierName][criterion]['complexities'][iteration][-1])
            complexitiesNumParamMetric.append(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'][iteration][-1])
    meanAllAcuracy = np.mean(allAccuracies)
    stdAllAcuracy = np.std(allAccuracies)
    meanChunkAccuracy = np.mean(endAccuracies)
    stdChunkAccuracy =  np.std(endAccuracies)
    meanChunkComplexity = np.mean(complexities)
    stdChunkComplexity = np.std(complexities)
    meanChunkComplexityNumParamMetric = np.mean(complexitiesNumParamMetric)
    stdChunkComplexityNumParamMetric = np.std(complexitiesNumParamMetric)
    return [meanChunkAccuracy, stdChunkAccuracy, meanChunkComplexity, stdChunkComplexity, meanChunkComplexityNumParamMetric, stdChunkComplexityNumParamMetric, meanAllAcuracy, stdAllAcuracy]


def dataToCSV(dataSetName):
    trainSetCfg = getTrainSetCfg(dataSetName)
    evalFilePrefix = Experiment.getExpPrefix2(None, trainSetCfg)
    evalDstDirectory = Paths.FeatureTmpsDirPrefix() + 'Evaluations/'
    classifierEvaluations = Serialization.loadJsonFile(evalDstDirectory + evalFilePrefix + 'evaluations.json')
    values = []
    accs = []
    complexities = []
    for i in range(len(classifierEvaluations['values'].keys())):
        classifierName = classifierEvaluations['values'].keys()[i]
        bestChunkSize, dummy = getBestChunkSize(classifierEvaluations, classifierName, not isStationary(dataSetName))
        singleValues = getSingleRunValues(classifierEvaluations, classifierName, bestChunkSize)
        chunkValues = getChunkValues(classifierEvaluations, classifierName)
        values.append([classifierName, bestChunkSize] + singleValues + chunkValues)
        accs.append(singleValues[8])
        complexities.append(singleValues[14])
    tmp = np.argsort(accs)[::-1]
    accRanks = np.empty(len(accs), int)
    accRanks[tmp] = np.arange(len(accs))
    tmp = np.argsort(complexities)
    complRanks = np.empty(len(complexities), int)
    complRanks[tmp] = np.arange(len(complexities))


    csvfile = open(evalDstDirectory + evalFilePrefix + 'single.csv', 'wb')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['classifier', 'BestChunkSize', '10% acc', '25% acc', '50% acc', '75% acc', '100% acc', 'mean acc', 'acc rank', 'complexity',
                     'complexityNumParamMetric', 'complexity rank', 'meanChunkAccuracy', 'meanChunkComplexity',
                     'meanChunkComplexityNumParamMetric', 'meanAllAccuracy'])

    for i in range(len(values)):
        writer.writerow(
            [values[i][0], "%d" % (values[i][1]),
             "%.1f(%.1f)" % (values[i][2] * 100, values[i][3] * 100),
             "%.1f(%.1f)" % (values[i][4] * 100, values[i][5] * 100),
             "%.1f(%.1f)" % (values[i][6] * 100, values[i][7] * 100),
             "%.1f(%.1f)" % (values[i][8] * 100, values[i][9] * 100),
             "%.1f(%.1f)" % (values[i][10] * 100, values[i][11] * 100),
             "%.1f(%.1f)" % (values[i][12] * 100, values[i][13] * 100),
             "%.1f" % (accRanks[i]),
             "%.1f(%.1f)" % (values[i][14], values[i][15]),
             "%.1f(%.1f)" % (values[i][16], values[i][17]),
             "%.1f" % (complRanks[i]),
             "%.1f(%.1f)" % (values[i][18] * 100, values[i][19] * 100),
             "%.1f(%.1f)" % (values[i][20], values[i][21]),
             "%.1f(%.1f)" % (values[i][22], values[i][23]),
             "%.1f(%.1f)" % (values[i][24] * 100, values[i][25] * 100)])

if __name__ == '__main__':
    criterionName = 'chunkSize'
    classifierNames = None
    classifierNames = ['ILVQ', 'ORF', 'LPP', 'LPPNSE', 'SGD', 'IELM', 'GNB', 'ISVM', 'LVGB']
    #classifierNames = ['iLVQ', 'iSVM', 'iELM', 'LPP']

    #doComparisonPlot('border', criterionName, classifierNames)
    #doComparisonPlot('overlap', criterionName, classifierNames)
    #doComparisonPlot('coil', criterionName, classifierNames)
    #doComparisonPlot('outdoor', criterionName, classifierNames)
    #doComparisonPlot('USPS', criterionName, classifierNames)
    #doComparisonPlot('DNA', criterionName, classifierNames)
    #doComparisonPlot('isolet', criterionName, classifierNames)
    #doComparisonPlot('letter', criterionName, classifierNames)
    #doComparisonPlot('satImage', criterionName, classifierNames)
    #doComparisonPlot('penDigits', criterionName, classifierNames)
    #doComparisonPlot('HAR', criterionName, classifierNames)
    #doComparisonPlot('gisette', criterionName, classifierNames)
    #doComparisonPlot('mnist', criterionName, classifierNames)
    #doComparisonPlot('news20', criterionName, classifierNames)

    #doComparisonPlot('elec', criterionName, classifierNames)
    #doComparisonPlot('covType', criterionName, classifierNames)
    doComparisonPlot('weather', criterionName, classifierNames)
    #doComparisonPlot('keystroke', criterionName, classifierNames)


    #doComparisonPlot('sea', criterionName, classifierNames)
    #doComparisonPlot('souza2CHT', criterionName, classifierNames)
    #doComparisonPlot('souza4CREV1', criterionName, classifierNames)
    #doComparisonPlot('souzaGears2C2D', criterionName, classifierNames)






