import os
import shutil
import glob

from Base import Serialization as Serial
from Base import Constants as Const
import numpy as np
from Base import CVFcts
from Base import Paths

class ImageFilesPreparer(object):
    def __init__(self):
        pass

    @staticmethod
    def copyAndRenameBySeqFile(imageDirectory, dstRoot, sequenceFilePath, fileSuffix):
        seqFile = open(sequenceFilePath, 'r')
        SeqCounter = {}
        if not os.path.exists(dstRoot):
            os.makedirs(dstRoot)
        for line in seqFile:
            splitList = line.split()
            if splitList[2] in SeqCounter.keys():
                SeqCounter[splitList[2]] += 1
            else:
                i = 1
                dstPath = dstRoot + splitList[2] + '_' + str(i)
                while len(glob.glob(dstPath + '*')) > 0:
                    i += 1
                    dstPath = dstRoot + splitList[2] + '_' + str(i)
                SeqCounter[splitList[2]] = i
            count = 1
            for i in range(int(splitList[0]), int(splitList[1]) + 1):
                srcFileName = str(i) + fileSuffix
                dstFileName = str(count) + fileSuffix
                srcPath = imageDirectory + srcFileName
                if os.path.exists(srcPath):
                    dstPath = dstRoot + splitList[2] + '_' + str(SeqCounter[splitList[2]]) + '_' + dstFileName
                    shutil.copy(srcPath, dstPath)
                    count += 1
                else:
                    print 'path ' + srcPath + ' doesnt exist'

    @staticmethod
    def mergeDirectories(srcDirectory1, srcDirectory2, dstDirectory, fileSuffix):
        fileIterator = Serial.SimpleFileIterator(srcDirectory2, fileSuffix)

        if not os.path.exists(dstDirectory):
            os.makedirs(dstDirectory)
        for filename in glob.glob(os.path.join(srcDirectory1, '*' + fileSuffix)):
            shutil.copy(filename, dstDirectory)
        LabelNameList = []
        lastSequenceNumber = -1
        dstPath = ''
        while True:
            imageFileName = fileIterator.getNextFileName()
            if imageFileName is None:
                break
            splitList = imageFileName.split('_')
            labelName = splitList[0]
            sequenceNumber = int(splitList[1])

            if not (labelName in LabelNameList):
                LabelNameList.append(labelName)
                lastSequenceNumber = -1

            if lastSequenceNumber != sequenceNumber:
                i = 1
                dstPath = dstDirectory + splitList[0] + '_' + str(i)
                while len(glob.glob(dstPath + '*')) > 0:
                    i += 1
                    dstPath = dstDirectory + splitList[0] + '_' + str(i)
                lastSequenceNumber = sequenceNumber
            srcPath = srcDirectory2 + imageFileName
            shutil.copy(srcPath, dstPath + '_' + splitList[2])


    @staticmethod
    def removeDuplicates(srcDirectory1, fileSuffix):
        fileIterator = Serial.SimpleFileIterator(srcDirectory1, fileSuffix)
        lastImg = None
        while True:
            imageFileName = fileIterator.getNextFileName()
            print imageFileName
            if imageFileName is None:
                break
            filePath = srcDirectory1 + imageFileName

            img = Serial.loadImage2(filePath)
            if lastImg is not None and np.all(img == lastImg):
                os.remove(filePath)
            lastImg = img.copy()

    @staticmethod
    def loadMasks(maskDirectory):
        fileIterator = Serial.SimpleFileIterator(maskDirectory, 'png')
        masks = []
        imageFileName = fileIterator.getNextFileName()
        while imageFileName is not None:
            if imageFileName is None:
                break
            filePath = maskDirectory + imageFileName
            img = Serial.loadImage2(filePath)
            masks.append(img)
            imageFileName = fileIterator.getNextFileName()

        return masks

    @staticmethod
    def generateFeaturesToImageFiles(imageFileDirectory, maskDirectory):
        masks = ImageFilesPreparer.loadMasks(maskDirectory)
        fileIterator = Serial.SimpleFileIterator(imageFileDirectory, 'png', revertOrder=True)
        allFeatures = np.empty(shape=(0, Const.featureDim()))
        allLabels = []
        while True:
            imageFileName = fileIterator.getNextFileName()

            if imageFileName is None:
                break
            filePath = imageFileDirectory + imageFileName
            print imageFileName
            for i in range(len(masks)):
                img = Serial.loadImage2(filePath)
                features = CVFcts.computeObjectFeatures(img, masks[i], reshape=False)
                allFeatures = np.vstack([allFeatures, features])
                allLabels = np.append(allLabels, i)
                #maskedImage = CVFcts.getMaskedImage(img, masks[i])
                #Serial.saveImage(maskedImage, imageFileDirectory + 'test.png')

        np.savetxt(Paths.rialtoTrainSamplesPath(), allFeatures, comments='', fmt='%.6g')
        np.savetxt(Paths.rialtoTrainLabelsPath(), allLabels, comments='', fmt='%i')
        return allFeatures, allLabels


if __name__ == '__main__':
    '''ImageFilesPreparer.copyAndRenameBySeqFile(
        '/hri/localdisk/vlosing/OnlineLearning/Images/Recordings/Demo/RecordedImages_Border/Images/',
        '/hri/localdisk/vlosing/OnlineLearning/Images/Demo/Border/',
        '/hri/localdisk/vlosing/OnlineLearning/Images/Recordings/Demo/RecordedImages_Border/annotations/objects.sec',
        Const.imageFilesSuffix())'''

    # ImageFilesPreparer.mergeDirectories(Paths.OutdoorSunnyImagesPath(),
    # Paths.OutdoorCloudedImagesPath(), Paths.OutdoorImagesDir(), Const.imageFilesSuffix())
    ImageFilesPreparer.removeDuplicates('/media/viktor/6EF25B34F25B002F/Dokumente und Einstellungen/Viktor/rialto/', 'png')
    #ImageFilesPreparer.generateFeaturesToImageFiles('/media/viktor/6EF25B34F25B002F/Dokumente und Einstellungen/Viktor/rialto/images/all/', '/home/viktor/rialto/masks/')