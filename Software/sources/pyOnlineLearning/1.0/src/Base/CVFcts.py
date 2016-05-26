import logging

import scipy
import numpy
import cv2

import hist  # color histograms
import Constants as Const


# # compute border outline visualization (mask area minus eroded mask area)
def visualizeOutline(mask):
    tmp1 = cv2.erode(mask, numpy.ones(shape=(5, 5)), iterations=1)
    diff = mask - tmp1
    return diff


# # paint red outline in rgbImage
def paintOutlineMask(rgbImage, mask):
    s = rgbImage.shape
    tmp = rgbImage
    tmp.shape = [s[0], s[1] / 3, 3]
    tmp = tmp[:, :, 0]
    tmp[mask.astype("bool")] = 255
    rgbImage.shape = s


# # set all mask areas within ignore area to zero
def setIgnoreInMasks(masks):
    ignoreMask = masks['ignore']
    s = ignoreMask.shape
    for maskName in masks.keys():
        if maskName == "ignore":
            pass
        m = masks[maskName].ravel()
        m[ignoreMask.ravel()] = 0
        m.shape = s
        masks[maskName] = m
    return masks


## is there a single pronounced obstacle in the color segmentation output "mask"?
##
## ignores upper part of image and very small obstacles
## Only one obstacle-> True
## Multiple obstacles -> Is the largest one [thresh] times larger than second largest?
## @returns (bool, mask of single pronounced obstacle)
def singlePronouncedObstacle(inMask, thresh=2., useNLowerLines=int(Const.imageHeight() * 3.0 / 4.0)):
    mask = inMask.copy()

    mask[:-useNLowerLines, :] = 0  # ignore upper part
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, numpy.ones(shape=(5, 5)))
    #enumerate distinct blobs
    labeled, num = scipy.ndimage.measurements.label(mask, structure=scipy.ones([3, 3], "uint8"))
    if num == 1:
        if len(numpy.where(mask == 255)[0]) > 5:  # XXVL
            logging.debug("Only one obstacle")
            return 1, visualizeOutline(mask), mask
        else:
            return 0, visualizeOutline(mask), mask
    if num > 1:
        numAr = scipy.zeros(num + 1)
        for i in range(num + 1):
            numAr[i] = scipy.sum(labeled == i)
        values = numAr.copy()
        values.sort()
        v1 = values[-1]
        v2 = values[-2]
        idx = scipy.argmax(numAr[1:]) + 1
        logging.debug("%d obstacles found" % num)
        logging.debug("Biggest obstacle is %.2f times larger than second one" % (1. * v1 / v2))
        logging.debug("idx:" + str(idx))
        logging.debug("numAr:" + str(numAr))
        if v2 * thresh < v1:
            logging.debug("idx %d is much bigger" % idx)
            mask[labeled != idx] = 0
            return 1, visualizeOutline(mask), mask
        logging.debug("Biggest obstacle is not much bigger than second one")

    return num, visualizeOutline(mask), mask


## compute color histogram of that part in imageIn where mask is > 0
def computeObjectFeatures(imageIn, mask, numberOfBins=Const.histogramBins(), colorSpace="rgb", reshape = True):
    image = imageIn.copy()
    if reshape: image.shape = [image.shape[0], image.shape[1] / 3, 3]
    if colorSpace == "rgb":
        pass
    elif colorSpace == "hsv":
        raise Exception("not yet implemented")
    else:
        raise Exception("Unknown colorspace %s" % colorSpace)
    #features = hist.histogramRGchromo(image, nbins=numberOfBins, mask=mask)
    features = hist.histogramRGB(image, nbins=numberOfBins, mask=mask)
    return features


def getMaskedImage(image, mask):
    tmp = image.copy()
    #tmp.shape = [Const.imageHeight(), Const.imageWidth(), 3]
    resultImage = numpy.zeros(shape=tmp.shape)
    indices = numpy.where(mask == 255)
    for i in range(3):
        resultImage[indices[0], indices[1], i] = tmp[indices[0], indices[1], i]
    return resultImage


def getMaskedData(mask, data):
    assert (len(data.shape) == 3)
    numberOfDimensions = data.shape[2]
    maskVec = mask.ravel().astype('bool')
    maskedData = numpy.zeros([maskVec.sum(), numberOfDimensions], numpy.float32)
    for i in range(numberOfDimensions):
        maskedData[:, i] = data[:, :, i].ravel()[maskVec]
    return maskedData


