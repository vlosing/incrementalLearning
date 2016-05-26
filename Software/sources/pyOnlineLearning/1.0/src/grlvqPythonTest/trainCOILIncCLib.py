import json
import time

import numpy

import Constants as Const
import libpythoninterface as GLVQC


if __name__ == '__main__':
    net1 = "net1"
    num_of_classes = 100
    epochs = 7
    gMax = 10
    trainsteps = num_of_classes * 72 * epochs
    learnRatePerNode = True
    threads_per_nodes = 1
    dimensionality = Const.featureDim()
    GLVQC.create_network(net1, dimensionality, GLVQC.NETTYPE_GMLVQ, learnRatePerNode, trainsteps, threads_per_nodes)
    GLVQC.set_learnrate_start_network(net1, 1.0)
    GLVQC.set_learnrate_metricWeights_start_network(net1, 0.01)
    f = open('data/coil_Features.json', 'r')
    AllFeatures = numpy.array(json.load(f))
    f.close()
    f = open('data/coil_Labels.json', 'r')
    AllLabels = numpy.array(json.load(f))
    AllLabels = AllLabels.astype(numpy.int32)

    f.close()
    ProtoSamples = []
    ProtoLabels = []
    for i in range(len(AllLabels)):
        if (i % 72) == 0:
            if ProtoSamples == []:
                ProtoSamples = AllFeatures[i, :]
            else:
                ProtoSamples = numpy.vstack([ProtoSamples, AllFeatures[i, :]])
            ProtoLabels.append(AllLabels[i])
    TrainSamples = AllFeatures
    TrainLabels = AllLabels
    for i in range(len(ProtoLabels)):
        GLVQC.add_prototype(net1, ProtoSamples[i, :], ProtoLabels[i])
    print "initialized prototypes"

    tic = time.time()

    for i in range(epochs):
        adaptMetricWeights = i > 0
        doInsertions = i > 1
        GLVQC.incremental_train_network(net1, TrainSamples, TrainLabels, doInsertions, gMax, adaptMetricWeights)
    print "finished training"
    print str(GLVQC.get_numofprototypes_network(net1)) + " prototypes"

    TestSamples = AllFeatures
    TestLabels = AllLabels
    [resultLabels, index] = GLVQC.process_network(net1, TestSamples)
    print str(time.time() - tic) + " seconds"
    correctCount = 0
    for i in range(len(resultLabels)):
        if resultLabels[i] == TestLabels[i]:
            correctCount += 1
    print str(correctCount / float(len(TestSamples))) + " correct"



