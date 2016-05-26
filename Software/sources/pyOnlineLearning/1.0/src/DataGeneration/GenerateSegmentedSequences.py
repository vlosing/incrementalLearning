import time
import os
import logging

import numpy
import AlmClib

from Base import Constants as Const
from Base import Paths
from Base import Serialization as BaseSerial
from Base.GrassSegmentation import GrassSegmentation
from Base import CVFcts


# # setup image buffers
def initBuffers(width, height, width_segment, height_segment):
    image = numpy.zeros((height, width * 3), dtype='uint8')
    rgbImage = numpy.zeros((height, width * 3), dtype='uint8')
    grayImage = numpy.zeros((height, width), dtype='uint8')
    rgbSegmentImage = numpy.zeros((height_segment, width_segment * 3), dtype='uint8')
    graySegmentImage = numpy.zeros((height_segment, width_segment), dtype='uint8')
    return image, rgbImage, grayImage, rgbSegmentImage, graySegmentImage


# # need to turn? where? turnDir==0: nor turn: -1:right, 1: left
def computeEvasionDir(AlmClib, mask, masks, w, h):
    tic_overlapComp = time.time()
    leftOverlap = AlmClib.maskOverlap(mask, masks['left'], w, h)
    rightOverlap = AlmClib.maskOverlap(mask, masks['right'], w, h)
    turnNecessary = leftOverlap + rightOverlap > 0
    if turnNecessary:
        if leftOverlap > rightOverlap:
            turnDir = "right"
        else:
            turnDir = "left"
    else:
        turnDir = "no turn"
    logging.debug("Left overlap: %d, right overlap: %d, dt: %.2fms" % (
        leftOverlap, rightOverlap, 1000 * (time.time() - tic_overlapComp)))
    logging.debug("Turning: %d, direction: %s" % (turnNecessary, turnDir))
    return turnNecessary, turnDir


if __name__ == '__main__':
    global lastAction
    lastAction = None

    h_segment = Const.imageHeight()
    w_segment = Const.imageWidth()

    ## verbosity of command line outputs XXVL loglevel einstellen
    logging.basicConfig(level=logging.INFO)
    LogLevel = 1

    ############################################################
    image, rgbImage, grayImage, rgbSegmentImage, graySegmentImage = initBuffers(Const.imageWidth(), Const.imageHeight(),
                                                                                w_segment, h_segment)

    imagePath = Paths.demoPreTrainImagesPath()
    dstImagePath = Paths.demoAppDataMissClassifiedImagesDir()

    fileIterator = BaseSerial.SimpleFileIterator(imagePath, Const.imageFilesSuffix())

    mask = GrassSegmentation.getInst().createMask()

    ## agnostic ring buffer, takes list of objects, will try to reuse buffers (if list elems are numpy arrays)
    ringBuf = BaseSerial.RingBuffer(100)

    ## running image number from camera
    imageIndex = 0
    ## running set index stored from buffered images
    setIndex = 1

    AlmClib.SetInputMode()  # XXVL?!
    masks = BaseSerial.loadMasks(w_segment, h_segment)

    masks = CVFcts.setIgnoreInMasks(masks)
    dropOutLimit = 10
    limitedSequenceLength = 10

    LabelNameList = []
    newSequence = True
    sequenceDone = False
    ObjSeqNumber = 0
    lastSequence = -2
    while True:
        try:
            imageFileName = fileIterator.getNextFileName()
            if imageFileName is None:
                break
            rgbImage = BaseSerial.loadImage(Const.imageWidth(), Const.imageHeight(), imagePath + imageFileName)
            tmp = imageFileName.split('_')
            labelName = tmp[0]
            newSequence = tmp[1] != lastSequence
            lastSequence = tmp[1]
            if not (labelName in LabelNameList):
                LabelNameList.append(labelName)
                newSequence = True
                ObjSeqNumber = 1
            if newSequence:
                ringBuf.clear()
                sequenceDone = False

            if not sequenceDone:
                rgbImage.shape = [Const.imageHeight(), Const.imageWidth(),
                                  3]  # XXVL hack for green channel color correction
                #rgbImage[:,:,2] = scipy.clip(rgbImage[:,:,2], 0, 127)
                #rgbImage[:,:,2] = rgbImage[:,:,2] * 2
                rgbImage.shape = [Const.imageHeight(), Const.imageWidth() * 3]

                # call grass segmentation on rgbSegmentImage, store results in mask. Specs item 4-5
                GrassSegmentation.getInst().processImage(rgbImage, mask)
                numObjects, objectMaskOutlined, maskProcessed = CVFcts.singlePronouncedObstacle(mask)  # Specs item 6

                '''cv2.imshow('maskProcessed', maskProcessed)
                tmp=rgbImage.copy();
                tmp.shape=[h_segment,w_segment,3]
                tmp2=tmp
                cv2.cvtColor(tmp, CV_RGB2BGR, tmp2)
                fileNameSplitted =  imageFileName.split('_')
                cv2.imshow('image', tmp2)
                key = cv2.waitKey(100)'''

                mask[:, :] = maskProcessed
                maskVec = maskProcessed.ravel().astype('bool')
                if numObjects == 1 and maskVec.sum() > Const.minObjectSize():
                    ringBuf.add([rgbImage])
                else:
                    ringBuf.clear()
                if numObjects > 0:
                    turnNecessary, turnDir = computeEvasionDir(AlmClib, mask, masks, w_segment, h_segment)
                    if turnNecessary and ringBuf.validEntries > dropOutLimit and not sequenceDone:
                        entries = ringBuf.getHistory()
                        if not os.path.exists(dstImagePath):
                            os.makedirs(dstImagePath)
                        resultPath = dstImagePath + labelName + '_' + str(ObjSeqNumber) + '_'
                        for i in range(limitedSequenceLength):
                            BaseSerial.saveImage(entries[len(entries) - limitedSequenceLength + i][0],
                                                 resultPath + str(i + 1) + Const.imageFilesSuffix(), True, h_segment,
                                                 w_segment)

                        sequenceDone = True
                        ObjSeqNumber += 1
                        ringBuf.clear()
        except KeyboardInterrupt:
            break
    GrassSegmentation.getInst().clear()
