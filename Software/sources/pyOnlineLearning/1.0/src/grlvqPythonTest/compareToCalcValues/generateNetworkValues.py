import json

import numpy

import libpythoninterface as GLVQC


def genNetworkValues(nettype, learnrate_per_node):
    f = open('data/traindataClass1.json', 'r')
    dataClass1 = numpy.array(json.load(f))
    f.close()
    f = open('data/traindataClass2.json', 'r')
    dataClass2 = numpy.array(json.load(f))
    f.close()

    net1 = "net1"
    num_of_classes = 2
    prototypes_per_class = 1
    trainsteps = num_of_classes * len(dataClass1)
    threads_per_nodes = 1
    dimensionality = 2
    GLVQC.create_network(net1, dimensionality, nettype, learnrate_per_node, trainsteps, threads_per_nodes)
    GLVQC.set_learnrate_start_network(net1, 1.0)
    GLVQC.set_learnrate_metricWeights_start_network(net1, 0.01)

    GLVQC.add_prototype(net1, dataClass1[0, :], 0)
    GLVQC.add_prototype(net1, dataClass2[0, :], 1)

    for sample in range(len(dataClass1) - 1):
        GLVQC.train_network(net1, dataClass1[sample + 1, :], [0])
        GLVQC.train_network(net1, dataClass2[sample + 1, :], [1])

    allMetricWeights = []
    prototypes = []
    if (nettype == GLVQC.NETTYPE_GRLVQ or nettype == GLVQC.NETTYPE_GMLVQ):
        metricWeights = GLVQC.get_metricWeights_network(net1, 0)
        allMetricWeights = metricWeights

    for c in range(num_of_classes):
        for p in range(prototypes_per_class):
            i = c * prototypes_per_class + p
            proto = GLVQC.get_weights_network(net1, i)

            if prototypes == []:
                prototypes = proto
            else:
                prototypes = numpy.vstack([prototypes, proto])
            if nettype == GLVQC.NETTYPE_LGRLVQ or nettype == GLVQC.NETTYPE_LGMLVQ:
                metricWeights = GLVQC.get_metricWeights_network(net1, i)
                if allMetricWeights == []:
                    allMetricWeights = metricWeights
                else:
                    allMetricWeights = numpy.vstack([allMetricWeights, metricWeights])
    GLVQC.delete_network(net1);

    prototypes = numpy.round(prototypes, 3)
    allMetricWeights = numpy.round(allMetricWeights, 3)
    return (prototypes, allMetricWeights)

