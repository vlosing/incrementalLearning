import logging
import json

import numpy

from Base import Serialization as serial
from Base import CVFcts
from Base import Constants as Const
from Base import Paths


if __name__ == '__main__':

    imagesPath, imagesMaskPath, dstFeaturesPath, dstLabelsPath, dstFeatureFileNamesPath = Paths.getCoilPathsRGB()
    fileIterator = serial.SimpleFileIterator(imagesPath, Const.coilImageFilesSuffix())

    AllLabels = numpy.empty(shape=(0, 1))
    AllFeatures = numpy.empty(shape=(0, Const.featureDim()))
    FeatureFileNames = []
    while True:
        imageFileName = fileIterator.getNextFileName()
        if imageFileName is None:
            break
        parts = imageFileName.split('_')
        objectNumber = int(parts[0].rsplit('obj')[1]) - 1
        viewAngle = int(parts[2].rsplit(Const.coilImageFilesSuffix())[0])

        label = imageFileName[0:imageFileName.find('_')]

        if objectNumber < 100:
            rgbImage = serial.loadImage(Const.imageWidth(), Const.imageHeight(), imagesPath + imageFileName)
            mask = serial.loadImage(Const.imageWidth(), Const.imageHeight(), imagesMaskPath + imageFileName)
            rgbImage.shape = [Const.imageHeight(), Const.imageWidth(), 3]
            rgbImage.shape = [Const.imageHeight(), Const.imageWidth() * 3]
            numObjects, objectMaskOutlined, maskProcessed = CVFcts.singlePronouncedObstacle(mask)  # Specs item 6
            if numObjects == 1:
                logging.debug("There is one object")
                features = CVFcts.computeObjectFeatures(rgbImage, maskProcessed,
                                                        Const.histogramBins())  # Specs item 6.1.5
                features = features.astype(numpy.float64)
                AllFeatures = numpy.vstack([AllFeatures, features])
                AllLabels = numpy.append(AllLabels, label)
                FeatureFileNames.append(imageFileName)
            else:
                print imageFileName + " not trained"
    f = open(dstFeaturesPath, 'w')
    json.dump(AllFeatures.tolist(), f)
    f.close()

    f = open(dstLabelsPath, 'w')
    json.dump(AllLabels.tolist(), f)
    f.close()

    f = open(dstFeatureFileNamesPath, 'w')
    json.dump(FeatureFileNames, f)
    f.close()




