import json

import numpy

from Base import Constants as Const
from Base import Paths
from Base.GrassSegmentation import GrassSegmentation
from Base import Serialization as Serial
from Base import CVFcts


if __name__ == '__main__':

    imagePath, dstFeaturesPath, dstLabelsPath, dstFeatureFileNamesPath = Paths.getDemoPreTrainPaths()
    fileIterator = Serial.SimpleFileIterator(imagePath, Const.imageFilesSuffix())
    mask = GrassSegmentation.getInst().createMask()
    AllLabels = numpy.empty(shape=(0, 1))
    AllFeatures = numpy.empty(shape=(0, Const.featureDim()))
    AllCount = 0
    singlePronouncedCount = 0
    ImagesCounter = {}
    FeatureFileNames = []
    while True:
        imageFileName = fileIterator.getNextFileName()
        if imageFileName is None:
            break
        image = Serial.loadImage(Const.imageWidth(), Const.imageHeight(), imagePath + imageFileName)
        label = imageFileName[0:imageFileName.find('_')]

        image.shape = [Const.imageHeight(), Const.imageWidth(), 3]
        image.shape = [Const.imageHeight(), Const.imageWidth() * 3]

        GrassSegmentation.getInst().processImage(image, mask)
        numObjects, objectMaskOutlined, maskProcessed = CVFcts.singlePronouncedObstacle(mask)  # Specs item 6

        '''cv2.imshow('maskProcessed', maskProcessed)
        tmp=image.copy();
        CVFcts.paintOutlineMask(tmp, objectMaskOutlined) # Specs item 6.1.1
        tmp.shape=[Const.imageHeight(), Const.imageWidth(), 3]
        tmp2=tmp
        cv2.cvtColor(tmp, CV_RGB2BGR, tmp2)
        cv2.imshow('image', tmp2)
        key = cv2.waitKey(150)'''

        AllCount += 1
        if numObjects > 0:
            singlePronouncedCount += 1
            maskVec = maskProcessed.ravel().astype('bool')
            if maskVec.sum() > Const.minObjectSize():
                features = CVFcts.computeObjectFeatures(image, maskProcessed)# Specs item 6.1.5
                AllFeatures = numpy.vstack([AllFeatures, features])
                AllLabels = numpy.append(AllLabels, label)
                FeatureFileNames.append(imageFileName)
                if label in ImagesCounter.keys():
                    ImagesCounter[label] += 1
                else:
                    ImagesCounter[label] = 1
            else:
                print 'failed ' + imageFileName
        else:
            print 'numObjects failed ' + str(numObjects) + ' ' + imageFileName
    print numpy.unique(AllLabels)
    print str(AllCount) + ' images'
    print str(singlePronouncedCount) + ' images with one pronounced object'
    print str(len(AllLabels)) + ' selected images'

    for label in numpy.unique(AllLabels):
        if label in ImagesCounter.keys():
            print label + ' ' + str(ImagesCounter[label])
        else:
            print label + ' 0'

    f = open(dstFeaturesPath, 'w')
    json.dump(AllFeatures.tolist(), f)
    f.close()

    f = open(dstLabelsPath, 'w')
    json.dump(AllLabels.tolist(), f)
    f.close()

    f = open(dstFeatureFileNamesPath, 'w')
    json.dump(FeatureFileNames, f)
    f.close()