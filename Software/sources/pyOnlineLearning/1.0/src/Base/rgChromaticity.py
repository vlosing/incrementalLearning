__author__ = 'vlosing'
import numpy
import matplotlib.pyplot as plt

import Constants as Const
import Serialization as Serial


def getRGChromacitySpaceFig(withBinning=False):
    pxColor = numpy.empty(shape=(0, 3))
    red = numpy.empty(shape=(0, 1))
    green = numpy.empty(shape=(0, 1))
    for r in numpy.arange(0, 1, 0.003):
        for g in numpy.arange(0, 1, 0.003):
            g = min(1 - r, g)
            b = numpy.clip(1 - r - g, 0, 1)
            red = numpy.append(red, r)
            green = numpy.append(green, g)
            color = [r, g, b]
            color = numpy.asarray(color)
            factor = 1 / float(max(r, g, b))
            color *= factor
            pxColor = numpy.vstack([pxColor, color])
    fig = plt.figure()
    plt.hold(True)

    plt.scatter(red, green, s=2, color=pxColor)

    if withBinning:
        nBin = 5
        for r in range(nBin):
            for c in range(nBin - r):
                plt.plot([c * 1 / float(nBin), c * 1 / float(nBin)], [r * 1 / float(nBin), (r + 1) * 1 / float(nBin)],
                         color='black')
                plt.plot([c * 1 / float(nBin), (c + 1) * 1 / float(nBin)], [r * 1 / float(nBin), r * 1 / float(nBin)],
                         color='black')
                plt.plot([c * 1 / float(nBin), (c + 1) * 1 / float(nBin)],
                         [(r + 1) * 1 / float(nBin), (r + 1) * 1 / float(nBin)], color='black')
                plt.plot([(c + 1) * 1 / float(nBin), (c + 1) * 1 / float(nBin)],
                         [r * 1 / float(nBin), (r + 1) * 1 / float(nBin)], color='black')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel('r')
    plt.ylabel('g')
    plt.rcParams.update({'font.size': 20})
    plt.show()
    return fig


def getSeveralObjectsRGChromacity(objectPixels1, objectPixels2, label1, label2):
    fig = plt.figure()
    plt.hold(True)
    r1, g1, pxColor = getObjectPixelsRGChromacity(objectPixels1)
    sc1 = plt.scatter(r1, g1, s=4, color='black', alpha=0.5, label=label1)
    r2, g2, pxColor = getObjectPixelsRGChromacity(objectPixels2)
    sc2 = plt.scatter(r2, g2, s=4, color='blue', alpha=0.5, label=label2)

    r3 = []
    g3 = []
    delta = 0.01
    for i in range(len(r1)):
        for j in range(len(r2)):
            if abs(r1[i] - r2[j]) <= delta:
                if abs(g1[i] - g2[j]) <= delta:
                    r3.append(r1[i])
                    g3.append(g1[i])
    sc3 = plt.scatter(r3, g3, s=4, color='red', label='Both')

    l1 = plt.legend([sc1, sc2, sc3], loc=1)
    legend = plt.legend(frameon=1)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('red')
    plt.ylabel('green')
    return fig


def getObjectPixelsRGChromacity(pixels, ignoreBlack=False):
    assert (len(pixels.shape) == 2)
    assert (pixels.shape[1] == 3)

    RGBSum = numpy.zeros(shape=(pixels.shape[0]), dtype=numpy.int32)
    RGBSum += numpy.sum([pixels[:, 0], pixels[:, 1], pixels[:, 2]], axis=0, dtype=numpy.float)
    RGBSum = RGBSum.astype(numpy.float)

    if not ignoreBlack:
        zeroIndices = numpy.argwhere(RGBSum == 0)
        nonZeroIndices = numpy.argwhere(RGBSum != 0)
        if len(zeroIndices) > 0:
            RGBSum[zeroIndices] = 3.0
            pixels[zeroIndices, :] = 1.0
        r = pixels[:, 0] / RGBSum
        g = pixels[:, 1] / RGBSum
        b = numpy.clip(numpy.ones(len(r)) - r - g, 0, 1)
        pxColor = numpy.empty(shape=(len(r), 3))
        pxColor[:, 0] = r
        pxColor[:, 1] = g
        pxColor[:, 2] = b

        for i in range(len(pxColor)):
            factor = 1 / float(numpy.max(pxColor[i, :]))
            pxColor[i, :] *= factor
    else:
        nonZeroIndices = numpy.argwhere(RGBSum != 0)
        r = pixels[:, 0]
        g = pixels[:, 1]
        r[nonZeroIndices] /= RGBSum[nonZeroIndices]
        g[nonZeroIndices] /= RGBSum[nonZeroIndices]
        b = numpy.zeros(len(r))
        b[nonZeroIndices] = 1 - r[nonZeroIndices] - g[nonZeroIndices]

        pxColor = numpy.empty(shape=(len(RGBSum), 3))
        pxColor[:, 0] = r
        pxColor[:, 1] = g
        pxColor[:, 2] = b
        for i in range(len(pxColor)):
            if numpy.max(pxColor[i, :]) != 0:
                factor = 1 / float(numpy.max(pxColor[i, :]))
                pxColor[i, :] *= factor
    return r, g, pxColor


def getRgChromaticityHistColors(nbins=Const.histogramBins()):
    colors = numpy.empty(shape=(0, 3))
    for r in range(nbins):
        for c in range(nbins - r):
            green = (r + 1) / float(nbins) - 0.5 / float(nbins)
            red = (c + 1) / float(nbins) - 0.5 / float(nbins)
            blue = 1 - red - green
            colors = numpy.vstack([colors, numpy.array([red, green, blue])])
    for i in range(len(colors)):
        factor = 1 / float(numpy.max(colors[i, :]))
        colors[i, :] *= factor
    return colors


def getObjectPixelsRGChromacityFig(pixels, title, withBinning=False):
    r, g, pxColor = getObjectPixelsRGChromacity(pixels)

    fig = plt.figure()
    plt.hold(True)

    plt.scatter(r, g, s=1, color=pxColor)

    if withBinning:
        nBin = 6
        for r in range(nBin):
            for c in range(nBin - r):
                plt.plot([c * 1 / float(nBin), c * 1 / float(nBin)], [r * 1 / float(nBin), (r + 1) * 1 / float(nBin)],
                         color='black')
                plt.plot([c * 1 / float(nBin), (c + 1) * 1 / float(nBin)], [r * 1 / float(nBin), r * 1 / float(nBin)],
                         color='black')
                plt.plot([c * 1 / float(nBin), (c + 1) * 1 / float(nBin)],
                         [(r + 1) * 1 / float(nBin), (r + 1) * 1 / float(nBin)], color='black')
                plt.plot([(c + 1) * 1 / float(nBin), (c + 1) * 1 / float(nBin)],
                         [r * 1 / float(nBin), (r + 1) * 1 / float(nBin)], color='black')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('red')
    plt.ylabel('green')
    plt.title(title)
    return fig


def saveRGB2RGChromacityImage(srcFileName, dstFileName):
    image = Serial.loadImage(Const.imageWidth(), Const.imageHeight(), srcFileName)

    image = image.reshape(Const.imageHeight() * Const.imageWidth(), 3)
    r, g, pxColor = getObjectPixelsRGChromacity(image)
    pxColor = pxColor.reshape((Const.imageHeight(), Const.imageWidth(), 3))
    # pxColor.shape = [Const.imageHeight(),Const.imageWidth(), 3]
    pxColor *= 255
    pxColor += 0.5
    Serial.saveImage(pxColor, dstFileName)


if __name__ == "__main__":
    getRGChromacitySpaceFig(withBinning=True)
    # saveMaskedRGChromaticityImageForFile('/hri/localdisk/vlosing/OnlineLearning/Images/Outdoor/Sunny+Clouded/RedBall_7_5.jpg','test.jpg')