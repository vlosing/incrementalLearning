__author__ = 'vlosing'
import numpy as np

TTC_PRED_IDX = 12

def getTrainSetName(vehicleType, streams, seqLength, maxTTCPredPos, maxTTCPredNeg, negSamplesRatio):
    return vehicleType + '_' + "%03d" % streams[0] + '-' + "%03d" % streams[-1] + '_' + str(seqLength) + '_' + str(maxTTCPredPos) + '_' + str(maxTTCPredNeg) + '_' + str(negSamplesRatio)

def getProperties2TrainSetName(trainSetName):

    splitted = trainSetName.split('-')

    firstStream = int(splitted[0][-3:])
    lastStream = int(splitted[1][:3])
    streams = np.arange(firstStream, lastStream + 1, 1)
    vehicleType = splitted[0][:5]

    splitted = splitted[1][3:].split('_')

    seqLength = splitted[1]
    maxTTCPredPos = splitted[2]
    maxTTCPredNeg = splitted[3]
    negSamplesRatio = splitted[4]
    return vehicleType,  streams, seqLength, maxTTCPredPos, maxTTCPredNeg, negSamplesRatio

def getTrainSetStreamName(vehicleType, stream, seqLength, maxTTCPredPos, maxTTCPredNeg, negSamplesRatio):
    return vehicleType + '_' + "%03d" % stream + '_' + str(seqLength) + '_' + str(maxTTCPredPos) + '_' + str(maxTTCPredNeg) + '_' + str(negSamplesRatio)