'''
Created on Feb 1, 2012

@author: jqueisse
'''

import numpy
import scipy.misc
import matplotlib.pyplot
# import pyopencv
import cv2 as pyopencv
import libpythoninterface
import random

if __name__ == '__main__':
    #GRLVQ/GMLVQ Network example

    random.seed(0)



    #train Network:
    numofobjects = 10
    objstepsize = 10
    numofviews = 10
    viewstepsize = 20
    coilpath = "/hri/localdisk/jqueisse/database/coil-100/obj";

    traindata = None
    evaldata = None

    for obj in range(numofobjects):
        currobj = obj * objstepsize + 1;

        for view in range(numofviews):
            currview = view * viewstepsize;

            path = coilpath + str(currobj) + "__" + str(currview) + ".png";
            path2 = coilpath + str(currobj) + "__" + str(currview + viewstepsize / 2) + ".png";

            scaledimg = pyopencv.info.asMat(None);
            imgdata = pyopencv.highgui.imread(path, pyopencv.highgui.CV_LOAD_IMAGE_GRAYSCALE)

            scaledimg2 = pyopencv.info.asMat(None);
            imgdata2 = pyopencv.highgui.imread(path, pyopencv.highgui.CV_LOAD_IMAGE_GRAYSCALE)

            pyopencv.cv_hpp_ext.resize(imgdata, scaledimg, pyopencv.cxcore_hpp_ext.Size2i(), 0.1, 0.1)
            pyopencv.cv_hpp_ext.resize(imgdata2, scaledimg2, pyopencv.cxcore_hpp_ext.Size2i(), 0.1, 0.1)

            pyopencv.highgui.imshow("traindata", imgdata)
            pyopencv.highgui.waitKey(50)

            imgmat = scipy.misc.fromimage(pyopencv.interfaces._Mat_to_pil_image(scaledimg)).astype(float)
            imgmat2 = scipy.misc.fromimage(pyopencv.interfaces._Mat_to_pil_image(scaledimg2)).astype(float)

            if traindata == None:
                traindata = imgmat.flatten();
            else:
                traindata = numpy.vstack((traindata, imgmat.flatten()))

            if evaldata == None:
                evaldata = imgmat2.flatten();
            else:
                evaldata = numpy.vstack((evaldata, imgmat2.flatten()))

    (samples, dimensionality) = traindata.shape
    traindata /= 255.0
    evaldata /= 255.0


    #Create network:
    net1 = "net1"
    nettype = libpythoninterface.NETTYPE_RLVQ
    learnrate_per_node = libpythoninterface.NETWORK_LEARNRATE
    num_of_classes = numofobjects
    prototypes_per_class = 4
    trainsteps = 3000
    do_node_random_init = True
    threads_per_nodes = 1
    libpythoninterface.create_network(net1, dimensionality, nettype, learnrate_per_node, num_of_classes,
                                      prototypes_per_class, trainsteps, do_node_random_init, threads_per_nodes)

    for sample in range(trainsteps):
        objnr = random.randrange(0, numofobjects)
        viewnr = random.randrange(0, numofviews)
        curr_sample = objnr * numofviews + viewnr;
        label = objnr;

        dist = libpythoninterface.train_network(net1, traindata[curr_sample, :], [label],
                                                libpythoninterface.NETMODE_BOTH, False)

    [result, indices] = libpythoninterface.process_network(net1, evaldata)
    print "res:", result

    corr_counter = numpy.zeros((numofobjects));
    overallcounter = 0
    for obj in range(numofobjects):
        currobj = obj * objstepsize + 1;

        for view in range(numofviews):

            curr_sample = obj * numofviews + view;
            label = obj;

            if result[curr_sample] == label:
                corr_counter[obj] += 1
                overallcounter += 1

        print "Class " + str(obj) + " detection rate: " + str(corr_counter[obj] * 100.0 / numofviews) + " percent";

    print "Overall detection rate : " + str(overallcounter * 100.0 / (numofviews * numofobjects)) + " percent"

    lambdas = libpythoninterface.get_lambdas_network(net1, 1)
    #matplotlib.pyplot.imshow(lambdas);
    matplotlib.pyplot.gray()
    #matplotlib.pyplot.show()


    #Vis Protos
    protocount = libpythoninterface.get_numofprototypes_network(net1)
    for p in range(protocount):
        prototype = libpythoninterface.get_weights_network(net1, p)
        prototype.shape = (numpy.sqrt(dimensionality), numpy.sqrt(dimensionality))
        matplotlib.pyplot.imshow(prototype);
        matplotlib.pyplot.show()




    #Save to file:
    libpythoninterface.save_network(net1, net1 + ".net");

    #Free mem:
    libpythoninterface.delete_network(net1);

    #------------------------------------------------------------
    #Reload created Network:

