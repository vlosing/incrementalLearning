import numpy
import matplotlib.pyplot as plt
import cv2

from Base import Constants as Const
import rgChromaticity



# XXVL refactor!
def histogramRGchromo(dataIn, mask=None, nbins=Const.histogramBins()):
    """compute nd-histogram of data
       data shape is T*d or w*h*d
       mask is optional 2d array """
    assert dataIn.dtype == "uint8"
    slen = len(dataIn.shape)
    assert (slen == 3)
    data = dataIn.astype(numpy.float32)
    data.shape = [dataIn.shape[0] * dataIn.shape[1], dataIn.shape[2]]

    RGBSum = data[:, 0] + data[:, 1] + data[:, 2]
    RGBSum = RGBSum.astype(numpy.float)

    rg = numpy.ndarray(shape=(data.shape[0], 2))
    zeroIndices = numpy.argwhere(RGBSum == 0)
    RGBSum[zeroIndices] = 3.0
    data[zeroIndices, :] = 1.0
    rg[:, 0] = data[:, 0] / RGBSum
    rg[:, 1] = data[:, 1] / RGBSum

    ndims = 2
    data = rg
    normalizationDivisor = 1

    if mask is not None:
        maskVec = mask.ravel().astype('bool')
        normalizationDivisor = maskVec.sum()
        maskedData = numpy.zeros([maskVec.sum(), ndims], numpy.float32)
        for i in range(ndims):
            tmp = data[:, i]
            maskedData[:, i] = tmp[maskVec]
        data = maskedData

    data[:, :] = numpy.floor(data[:, :] * nbins)
    data[:, :] = numpy.clip(data[:, :], 0, nbins - 1)

    # comment missing small gauss for nbins
    #i=index in Table, c = column of table, r = row of table, n=number of bins
    #i = n*c - 0.5((c-1)^2+c-1)+r
    tmp = numpy.ndarray(shape=(data.shape[0]))
    tmp[:] = data[:, 1] * nbins - 0.5 * ((data[:, 1] - 1) ** 2 + data[:, 1] - 1) + data[:, 0]
    #print str(numpy.max(tmp)) + ' ' + str(numpy.min(tmp))

    hist = numpy.histogram(tmp, bins=0.5 * (nbins ** 2 + nbins), range=(0, 0.5 * (nbins ** 2 + nbins) - 1))
    return hist[0] / float(normalizationDivisor)


def getRGBHistColors(nbins=Const.histogramBins()):
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


'''def histogramRGB(dataIn, mask=None, nbins=Const.histogramBins()):
    """compute nd-histogram of data
       data shape is T*d or w*h*d
       mask is optional 2d array """
    assert dataIn.dtype == "uint8"
    slen = len(dataIn.shape)
    assert (slen==2 or slen==3)
    ndims = dataIn.shape[-1]
    data = dataIn.astype('uint32')
    if slen==3: data.shape = [dataIn.shape[0]*dataIn.shape[1], dataIn.shape[2]]
    normalizationDivisor = 1
    if mask is not None:
        maskVec = mask.ravel().astype('bool')
        maskedData = numpy.zeros([maskVec.sum(), ndims], 'uint32')
        normalizationDivisor = maskVec.sum()
        for i in range(ndims):
            tmp = data[:,i]
            maskedData[:,i] = tmp[maskVec]
        data = maskedData

    factor = float(nbins)/255
    for d in range(ndims):
        data[:,d] *= factor # binning
        data[:,d] = numpy.clip(data[:,d],0, nbins-1)
        data[:,d] *= nbins**d # integration
    data = numpy.sum(data, axis=1)
    #valueRange = 0
    hist = numpy.histogram(data, bins=nbins**ndims, range=(0, nbins**ndims))
    return hist[0]/ float(normalizationDivisor)'''


def histogramRGB(dataIn, mask=None, nbins=Const.histogramBins()):
    hist = cv2.calcHist([dataIn], [0, 1, 2], mask, [nbins, nbins, nbins], [0, 256, 0, 256, 0, 256])
    maskVec = mask.ravel().astype('bool')
    hist = hist.ravel() / float(max(maskVec.sum(), 1))
    return hist


def getFeatureHistFig(feature, title, nbins=Const.histogramBins(), totalBinCount=Const.featureDim()):
    histColors = rgChromaticity.getRgChromaticityHistColors(nbins)
    fig = plt.figure()
    plt.hold(True)
    '''indices = numpy.argsort(feature)
    indices = indices[::-1]
    sortedFeature = feature[indices,:]
    sortedColors = numpy.array(histColors)[indices]'''
    avgColor = numpy.array([0., 0., 0.])
    histColors = numpy.array(histColors)

    for i in range(totalBinCount):
        p1 = plt.bar(i, feature[i], color=histColors[i])
        avgColor += histColors[i, :] * feature[i]
    plt.xlim(0, totalBinCount)
    plt.ylim(0, 1)
    # plt.xlabel('bins')
    #plt.ylabel('value')
    plt.title(title)

    '''avgColor /= numpy.sum(avgColor)
    p1 = plt.scatter(-1,-1, color = avgColor)
    l1 = plt.legend([p1], ["avgColor"], loc=1)
    legend = plt.legend(frameon = 1)'''
    return fig


def getFeatureHistFig(feature, title, nbins=Const.histogramBins(), totalBinCount=Const.featureDim()):
    histColors = rgChromaticity.getRgChromaticityHistColors(nbins)
    fig = plt.figure()
    plt.hold(True)
    '''indices = numpy.argsort(feature)
    indices = indices[::-1]
    sortedFeature = feature[indices,:]
    sortedColors = numpy.array(histColors)[indices]'''
    avgColor = numpy.array([0., 0., 0.])
    histColors = numpy.array(histColors)

    for i in range(totalBinCount):
        p1 = plt.bar(i, feature[i], color=histColors[i])
        avgColor += histColors[i, :] * feature[i]
    plt.xlim(0, totalBinCount)
    plt.ylim(0, 1)
    # plt.xlabel('bins')
    #plt.ylabel('value')
    plt.title(title)

    '''avgColor /= numpy.sum(avgColor)
    p1 = plt.scatter(-1,-1, color = avgColor)
    l1 = plt.legend([p1], ["avgColor"], loc=1)
    legend = plt.legend(frameon = 1)'''
    return fig


if __name__ == "__main__":
    # genObjectsFigures('../../../Images/Outdoor/SunnyCloudedMedium2/', 'Forceps_7_10.jpg', '../Statistics/', '')
    #genObjectsFigures('../../../Images/Outdoor/SunnyCloudedMedium2/', 'Stump_3_10.jpg', '../Statistics/', '')
    #genObjectsFigures('../../../Images/Outdoor/SunnyCloudedMedium2/', 'SmallDog_4_10.jpg', '../Statistics/', '')

    imgData = numpy.zeros(shape=(10, 10, 3), dtype=numpy.uint8)
    imgData[0:10, 0, 2] = 25
    imgData[0:10, 0, 1] = 0
    imgData[0:10, 0, 0] = 0
    mask = numpy.ones(shape=(10, 10), dtype=numpy.uint8)
    hist = histogramRGB(imgData, mask=mask, nbins=6)
    print hist
    #print hist2
    #plt.plot((hist))
    #plt.show()