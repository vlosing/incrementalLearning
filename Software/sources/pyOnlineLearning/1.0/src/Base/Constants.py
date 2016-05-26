__author__ = 'pioneer'


def coilImageFilesSuffix():
    return ".png"


def imageFilesSuffix():
    return ".jpg"


def histogramBins():
    return 3
    #return 4


def featureDim():
    #return int(0.5*(histogramBins() * histogramBins() + histogramBins())) #rg-chromaticity
    return histogramBins() ** 3  # rgb


def imageWidth():
    return 320


def imageHeight():
    return 240


def minObjectSize():
    return 500


def DemoMinObjectSize():
    return 2500