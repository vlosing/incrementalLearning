// -*- C++ -*-
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Amir Saffari, amir@ymer.org
 * Copyright (C) 2010 Amir Saffari, 
 *                    Institute for Computer Graphics and Vision, 
 *                    Graz University of Technology, Austria
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <string.h>
#include <libconfig.h++>

#include "data.h"
#include "experimenter.h"
#include "online_rf.h"
#include "linear_larank.h"
#include "online_mcboost.h"
#include "online_mclpboost.h"

using namespace std;
using namespace libconfig;

typedef enum {
    ORT, ORF, OMCBOOST, OMCLPBOOST, LARANK
} CLASSIFIER_TYPE;


int main(int argc, char *argv[]) {
	// Parsing command line
    int classifier = -1, doTraining = false, doTesting = false, doT2 = false, inputCounter = 1;
    if (argc < 11) {
        cout << "\tNot enough input arguments. 12 required" << endl;
        exit(EXIT_SUCCESS);
    }
    inputCounter = 1;
    string confFileName = argv[inputCounter++];
    string trainFeaturesFileName = argv[inputCounter++];
    string trainLabelsFileName = argv[inputCounter++];
    string testFeaturesFileName = argv[inputCounter++];
    string testLabelsFileName = argv[inputCounter++];
    int evaluationStepSize = atoi(argv[inputCounter++]);
    string evalDstPathPrefix = argv[inputCounter++];
    int splitTestSamples = atoi(argv[inputCounter++]);
    int numTrees = atoi(argv[inputCounter++]);
    int numRandomTests = atoi(argv[inputCounter++]);
    int maxDepth = atoi(argv[inputCounter++]);
    int counterThreshold = atoi(argv[inputCounter++]);

    Hyperparameters hp(confFileName, trainFeaturesFileName, trainLabelsFileName,
    		testFeaturesFileName, testLabelsFileName, numTrees, numRandomTests, maxDepth, counterThreshold);

    // Creating the train data
    DataSet dataset_tr, dataset_ts;
    dataset_tr.load(hp.trainData, hp.trainLabels);
    if (testFeaturesFileName != "_"){
    	dataset_ts.load(hp.testData, hp.testLabels);
    }
    Classifier* model = new OnlineRF(hp, dataset_tr.m_numClasses, dataset_tr.m_numFeatures,
                             dataset_tr.m_minFeatRange, dataset_tr.m_maxFeatRange);
    trainAndEvaluate(model, dataset_tr, hp, dataset_ts, evaluationStepSize, evalDstPathPrefix, splitTestSamples);
    return EXIT_SUCCESS;
}
