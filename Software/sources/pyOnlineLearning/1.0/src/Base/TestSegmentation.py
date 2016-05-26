# # @addtogroup Alm
# # @{
import time
import logging

import cv2
from cv2.cv import CV_RGB2BGR
import numpy

import AlmClib
from Base import Constants as Const
from Base.GrassSegmentation import GrassSegmentation
from Base import CVFcts
from Base import Paths
from ALM_v2.ImageGenerator import CameraImageGenerator, FileImageGenerator


## setup image buffers
def initBuffers(width, height, width_segment, height_segment):
    image = numpy.zeros((height, width * 3), dtype='uint8')
    rgbImage = numpy.zeros((height, width * 3), dtype='uint8')
    grayImage = numpy.zeros((height, width), dtype='uint8')
    rgbSegmentImage = numpy.zeros((height_segment, width_segment * 3), dtype='uint8')
    graySegmentImage = numpy.zeros((height_segment, width_segment), dtype='uint8')
    return image, rgbImage, grayImage, rgbSegmentImage, graySegmentImage


def getImageGenerator(useCamera):
    if useCamera:
        return CameraImageGenerator()
    else:
        return FileImageGenerator(Paths.OutdoorTestImageDir())


if __name__ == '__main__':
    global lastAction;
    lastAction = None

    ## verbosity of command line outputs XXVL loglevel einstellen
    logging.basicConfig(level=logging.INFO)
    LogLevel = 1
    ############################################################
    useCamera = True
    # MAIN START
    ## help text

    imageGenerator = getImageGenerator(useCamera)
    mask = GrassSegmentation.getInst().createMask()
    AlmClib.SetInputMode();  #XXVL?!

    while True:
        try:
            logging.debug("")
            ## current time for timing stats
            roundTic = time.time()
            image, grayImage = imageGenerator.getImage(False)
            if image == None:
                break

            image.shape = [Const.imageHeight(), Const.imageWidth(), 3]
            image.shape = [Const.imageHeight(), Const.imageWidth() * 3]

            GrassSegmentation.getInst().processImage(image,
                                                     mask)  # call grass segmentation on rgbSegmentImage, store results in mask. Specs item 4-5
            numObjects, objectMaskOutlined, maskProcessed = CVFcts.singlePronouncedObstacle(mask)  # Specs item 6
            cv2.imshow('mask', mask)
            cv2.imshow('maskProcessed', maskProcessed)
            #cv2.imshow('maskOutlined', objectMaskOutlined)
            CVFcts.paintOutlineMask(image, objectMaskOutlined)  # Specs item 6.1.1

            tmp = image.copy();
            tmp.shape = [Const.imageHeight(), Const.imageWidth(), 3]
            tmp2 = tmp
            cv2.cvtColor(tmp, CV_RGB2BGR, tmp2)
            cv2.imshow('image', tmp2)
            imageGenerator.adapt(maskProcessed)
            key = cv2.waitKey(2)
            '''if numObjects == 1:
                features = CVFcts.computeObjectFeatures(rgbImage, maskProcessed)
                print features
                hist.paintFeaturHist(features)'''

        except KeyboardInterrupt:
            break
    print "shutting down, please wait 3s"
    GrassSegmentation.getInst().clear()
    time.sleep(3)

## @}
