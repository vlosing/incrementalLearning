import logging

import numpy

from Base import Serialization as serial
from Base import CVFcts as cvFcts
from Base import Constants as Const
from Classification import NaiveNNClassifier
import libpythoninterface


if __name__ == '__main__':
    # # empty online classifier
    classifier = NaiveNNClassifier()

    net1 = "net1"
    num_of_classes = 100
    prototypes_per_class = 1
    trainsteps = num_of_classes * 72
    do_node_random_init = False
    threads_per_nodes = 1
    dimensionality = Const.featureDim()
    libpythoninterface.create_network(net1, dimensionality, libpythoninterface.NETTYPE_GLVQ, False, num_of_classes,
                                      prototypes_per_class, trainsteps, do_node_random_init, threads_per_nodes)
    libpythoninterface.set_learnrate_start_network(net1, 1.0)
    libpythoninterface.set_learnrate_metricWeights_start_network(net1, 0.01)

    fileIterator = serial.SimpleFileIterator(Const.coilSegmentedImagesPath(), Const.imageFilesSuffix())
    while True:
        imageFileName = fileIterator.getNextFileName()
        if imageFileName == None:
            break
        parts = imageFileName.split('_')
        objectNumber = int(parts[0].rsplit('obj')[1])
        viewAngle = int(parts[2].rsplit(Const.imageFilesSuffix())[0])

        if objectNumber <= 100 and viewAngle == 0:
            rgbImage = serial.loadImage(Const.imageWidth(), Const.imageHeight(),
                                        Const.coilSegmentedImagesPath() + imageFileName)
            mask = serial.loadImage(Const.imageWidth(), Const.imageHeight(),
                                    Const.coilSegmentedImagesMaskPath() + imageFileName)
            rgbImage.shape = [Const.imageHeight(), Const.imageWidth(), 3]
            rgbImage.shape = [Const.imageHeight(), Const.imageWidth() * 3]
            singleObject, objectMask = cvFcts.singlePronouncedObstacle(mask)  # Specs item 6
            if singleObject:
                logging.debug("There is one object")
                # Specs 6.1.3. Check for human label action and set label
                features = cvFcts.computeObjectFeatures(rgbImage, objectMask, Const.histogramBins())  # Specs item 6.1.5
                features = features.astype(numpy.float64)
                libpythoninterface.add_prototype(net1, features, objectNumber)
            else:
                print imageFileName + " not initialized"
    print "initialized prototypes"
    fileIterator = serial.SimpleFileIterator(Const.coilSegmentedImagesPath(), Const.imageFilesSuffix())
    AllFeatures = []
    AllLabels = []
    while True:
        imageFileName = fileIterator.getNextFileName()
        if imageFileName == None:
            break
        parts = imageFileName.split('_')
        objectNumber = int(parts[0].rsplit('obj')[1])
        viewAngle = int(parts[2].rsplit(Const.imageFilesSuffix())[0])

        if objectNumber <= 100:
            rgbImage = serial.loadImage(Const.imageWidth(), Const.imageHeight(),
                                        Const.coilSegmentedImagesPath() + imageFileName)
            mask = serial.loadImage(Const.imageWidth(), Const.imageHeight(),
                                    Const.coilSegmentedImagesMaskPath() + imageFileName)
            rgbImage.shape = [Const.imageHeight(), Const.imageWidth(), 3]
            rgbImage.shape = [Const.imageHeight(), Const.imageWidth() * 3]
            singleObject, objectMask = cvFcts.singlePronouncedObstacle(mask)  # Specs item 6
            if singleObject:
                logging.debug("There is one object")
                # Specs 6.1.3. Check for human label action and set label
                features = cvFcts.computeObjectFeatures(rgbImage, objectMask, Const.histogramBins())  # Specs item 6.1.5
                features = features.astype(numpy.float64)
                if (AllFeatures == []):
                    AllFeatures = features
                else:
                    AllFeatures = numpy.vstack([AllFeatures, features])
                AllLabels.append(objectNumber)

            else:
                print imageFileName + " not trained"
    print "Loaded training-Images"
    AllLabels = numpy.asarray(AllLabels)
    AllLabels = AllLabels.astype(numpy.int32)
    libpythoninterface.train_network(net1, AllFeatures, AllLabels)
    print "finished training"
    fileIterator = serial.SimpleFileIterator(Const.coilSegmentedImagesPath(), Const.imageFilesSuffix())
    allObjects = 0
    correct = 0
    AllFeatures = []
    AllLabels = []
    while True:
        imageFileName = fileIterator.getNextFileName()
        if imageFileName == None:
            break
        parts = imageFileName.split('_')
        objectNumber = int(parts[0].rsplit('obj')[1])
        viewAngle = int(parts[2].rsplit(Const.imageFilesSuffix())[0])
        if objectNumber <= 100:

            rgbImage = serial.loadImage(Const.imageWidth(), Const.imageHeight(),
                                        Const.coilSegmentedImagesPath() + imageFileName)
            mask = serial.loadImage(Const.imageWidth(), Const.imageHeight(),
                                    Const.coilSegmentedImagesMaskPath() + imageFileName)
            rgbImage.shape = [Const.imageHeight(), Const.imageWidth(), 3]
            rgbImage.shape = [Const.imageHeight(), Const.imageWidth() * 3]
            singleObject, objectMask = cvFcts.singlePronouncedObstacle(mask)  # Specs item 6
            if singleObject:
                allObjects += 1
                logging.debug("There is one object")
                features = cvFcts.computeObjectFeatures(rgbImage, objectMask)  # Specs item 6.1.5
                features = features.astype(numpy.float64)
                if (AllFeatures == []):
                    AllFeatures = features
                else:
                    AllFeatures = numpy.vstack([AllFeatures, features])
                AllLabels.append(objectNumber)
                #[label, index] = libpythoninterface.process_network(net1, features)
                #print "classifying " + imageFileName + " as " + str(label)
                #if label == objectNumber:
                #    correct += 1
            else:
                print imageFileName + " not classified"
    print("loaded Test-Images")
    [label, index] = libpythoninterface.process_network(net1, AllFeatures)
    for i in range(len(label)):
        if label[i] == AllLabels[i]:
            correct += 1

    print str(correct / float(len(AllLabels))) + " correct"



