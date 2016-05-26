__author__ = 'vlosing'
import json
from Base.Serialization import SimpleFileIterator
import os
import time
import logging
import cProfile

class CutInHystereseDataGenerator(object):
    def __init__(self, prefixDirectory, dstPathPrefix, streamNames, ppPrefix='PhysicalPrediction_'):
        self.prefixDirectory = prefixDirectory
        self.dstPathPrefix = dstPathPrefix
        self.streamNames = streamNames
        self.ppPrefix = ppPrefix

    def loadJsonFile(self, fileNamePath):
        try:
            f = open(fileNamePath, 'r')
            try:
                data = json.load(f)
            finally:
                f.close()
            return data
        except ValueError:
            print 'valError ' + fileNamePath
            return None

    def getPPFileName(self, CBPFileName):
        fileNameSplit = CBPFileName.split('_')
        return self.ppPrefix + fileNameSplit[1]

    def getSamplesFromFileName(self, srcDirectoryCBP, fileName, srcDirectoryPP):
        featureFile = self.loadJsonFile(srcDirectoryCBP + fileName)
        features = []
        vehicleIds = []
        allVehicleIds = []
        ppFilename = srcDirectoryPP + self.getPPFileName(fileName)
        ppData = None
        if os.path.exists(ppFilename):
            ppData = self.loadJsonFile(ppFilename)
            if featureFile is not None and ppData is not None:
                for vehicle in featureFile['Vehicles']:
                        #[perception_confidence_pred', 'ttc_lpred', 't_on_lane_pred', 'relV_pred', 'ttc_lsuc',
                        #'perception_confidence_v', 'g_size', 'abs_a', 't_last_lane_change', 'diff_ttc_ttg',
                        #'diff_ttc_ttca', 'ttg', 'ttc_pred']
                        allVehicleIds.append(vehicle['Id'])
                        features = features + vehicle['IndicatorVariables'].values()
                        vehicleIds.append(vehicle['Id'])
        else:
            print 'PhysicalPrediction file %s does not exist to CBP file %s' % (ppFilename, srcDirectoryCBP + fileName)
        return features, vehicleIds, allVehicleIds, ppData, ppFilename.split('/Results_c/')[1]

    def getTimeStampFromFileName(self, fileName):
        fileNameSplit = fileName.split('_')
        timeStamp = int(fileNameSplit[1].split('.')[0])
        return timeStamp

    def generateAndSave(self):
        tic = time.time()
        for streamName in self.streamNames:
            streamData = []
            logging.info(streamName)
            prefixDirectory = os.path.normpath(self.prefixDirectory) + os.sep + streamName
            srcDirectoryCBP = prefixDirectory + '/CBP/data/'
            srcDirectoryPP = prefixDirectory + '/PhysicalPrediction/data/'
            groundTruthPath = prefixDirectory + '/Evaluation/groundtruth.json'

            fileIterator = SimpleFileIterator(srcDirectoryCBP, 'json')
            while fileIterator.getNextFileName() is not None:
                samples, vehicleIds, allVehicleIds, ppData, ppFilename = self.getSamplesFromFileName(srcDirectoryCBP, fileIterator.getCurrentFileName(), srcDirectoryPP)
                if len(vehicleIds) > 0:
                    timeStamp = self.getTimeStampFromFileName(fileIterator.getCurrentFileName())
                    streamData.append([timeStamp, samples, vehicleIds, allVehicleIds, ppData, ppFilename])
            groundTruths = self.loadJsonFile(groundTruthPath)
            f = open(self.dstPathPrefix + streamName + '.json', 'w')
            json.dump([streamData, groundTruths], f)
            f.close()
        logging.info(str(time.time() - tic) + " seconds")




if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    dataGen = CutInHystereseDataGenerator('/hri/localdisk/vlosing/ACC_Data/2X_EU/Results_c/', '/hri/storage/user/vlosing/HystereseCutIn/2X_EU_',
                                          ['Stream073', 'Stream074', 'Stream075', 'Stream076', 'Stream077', 'Stream078', 'Stream079',
                                           'Stream080', 'Stream081', 'Stream082', 'Stream083', 'Stream084', 'Stream085', 'Stream086', 'Stream087', 'Stream088', 'Stream089'])
    dataGen.generateAndSave()
                                           #['Stream001', 'Stream002', 'Stream003', 'Stream004', 'Stream005', 'Stream006', 'Stream007', 'Stream008', 'Stream009',
                                           # 'Stream010', 'Stream011', 'Stream012', 'Stream013', 'Stream014', 'Stream015', 'Stream016', 'Stream017', 'Stream018',
                                           # 'Stream019', 'Stream020', 'Stream021', 'Stream022', 'Stream023', 'Stream024', 'Stream025', 'Stream026', 'Stream027'])




