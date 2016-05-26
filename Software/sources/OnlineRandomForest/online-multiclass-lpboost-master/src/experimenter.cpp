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

#include <fstream>
#include <sys/time.h>

#include "experimenter.h"
#include <vector>
#include <utility>
#include <sstream>

using namespace std;

std::vector<int> trainSequence(Classifier* model, DataSet& dataset, Hyperparameters& hp,
		int startidx, int endIdx) {
	vector<int> predictedTrainLabels;
	for (int nSamp = startidx; nSamp <= endIdx; nSamp++) {
			Result result(dataset.m_numClasses);
			model->eval(dataset.m_samples[nSamp], result);
			predictedTrainLabels.push_back(result.prediction);
		//}
		model->update(dataset.m_samples[nSamp]);
	}
	return predictedTrainLabels;
}

std::vector<int> predict(Classifier * model, DataSet & dataset) {
	vector<int> results;
	for (int nSamp = 0; nSamp < dataset.m_numSamples; nSamp++) {
		Result result(dataset.m_numClasses);
		model->eval(dataset.m_samples[nSamp], result);
		results.push_back(result.prediction);
	}
	return results;
}

std::vector<int> predictSequence(Classifier * model, DataSet & dataset, int startidx, int endIdx) {
	vector<int> results;
	for (int nSamp = startidx; nSamp <= endIdx; nSamp++) {
		Result result(dataset.m_numClasses);
		model->eval(dataset.m_samples[nSamp], result);
		results.push_back(result.prediction);
	}
	return results;
}

std::vector<std::pair<int, int> > getArraySplits(int length, int numOfParts) {
	vector<pair<int, int> > result;
	int partsize = length / numOfParts;
	int modulo = length % numOfParts;
	int addNext = 0;
	for (int part = 0; part < numOfParts; part++) {
		int from = (part) * partsize + addNext;
		int to = from + partsize - 1;
		if (modulo > 0) {
			to = to + 1;
			modulo = modulo - 1;
			addNext = addNext + 1;
		}
		std::pair<int, int> split;
		split.first = from;
		split.second = to;
		result.push_back(split);
	}
	return result;
}

std::vector<std::pair<int, int> > getArraySplitsBySplitSize(int length, int splitSize) {
	vector<pair<int, int> > result;
	int from = 0;
    while (from < length){
        int to = std::min(from+splitSize, length)-1;
		std::pair<int, int> split;
		split.first = from;
		split.second = to;
		from = from + splitSize;
        result.push_back(split);
    }
	return result;
}

void trainAndEvaluate(Classifier* model, DataSet& dataset_tr,
		Hyperparameters& hp, DataSet& dataset_ts, int evaluationStepSize,
		const string& evalDstPathPrefix, const int splitTestSamples) {
	vector<pair<int, int> > splits = getArraySplitsBySplitSize(dataset_tr.m_numSamples,
			evaluationStepSize);
	int numOfSplits = splits.size();
	vector<int> complexities;
	vector<int> complexitiesNumParamMetric;
	for (int split = 0; split < numOfSplits; split++) {
		cout << "chunk" << split + 1 << "/" << numOfSplits << endl;
		vector<int> predictedTrainLabels;
		predictedTrainLabels = trainSequence(model, dataset_tr, hp, splits[split].first, splits[split].second);
		if (split > 0){
			stringstream sstr;
			sstr << evalDstPathPrefix << "_" << split + 1  << "of" << numOfSplits << "predictedTrainLabels.csv";
			std::ofstream file(sstr.str().c_str(), std::ofstream::out);
			for (size_t i = 0; i < predictedTrainLabels.size(); i++){
				file << predictedTrainLabels[i] << endl;
			}
			file.close();
		}
		if (dataset_ts.m_numSamples > 0){
			vector<int> predictedLables;
			if (splitTestSamples){
				predictedLables = predictSequence(model, dataset_ts, splits[split].first, splits[split].second);
			}else{
				predictedLables = predict(model, dataset_ts);
			}
			stringstream sstr;
			sstr << evalDstPathPrefix << "_" << split + 1  << "of" << numOfSplits <<".csv";
			std::ofstream file(sstr.str().c_str(), std::ofstream::out);
			for (size_t i = 0; i < predictedLables.size(); i++){
				file << predictedLables[i] << endl;
			}
			file.close();
		}
		complexities.push_back(model->getComplexity());
		complexitiesNumParamMetric.push_back(model->getComplexityNumParamMetric());
	}
	stringstream sstr;
	sstr << evalDstPathPrefix << "_of" << numOfSplits <<"complexities.csv";
	std::ofstream file(sstr.str().c_str(), std::ofstream::out);
	for (size_t i = 0; i < complexities.size(); i++){
		file << complexities[i] << endl;
	}
	file.close();

	sstr.str("");
	sstr << evalDstPathPrefix << "_of" << numOfSplits <<"complexitiesNumParamMetric.csv";
	std::ofstream file2(sstr.str().c_str(), std::ofstream::out);
	for (size_t i = 0; i < complexitiesNumParamMetric.size(); i++){
		file2 << complexitiesNumParamMetric[i] << endl;
	}
	file2.close();


}
void train(Classifier* model, DataSet& dataset, Hyperparameters& hp) {
	timeval startTime;
	gettimeofday(&startTime, NULL);

	vector<int> randIndex;
	int sampRatio = dataset.m_numSamples / 10;
	vector<double> trainError(hp.numEpochs, 0.0);
	for (int nEpoch = 0; nEpoch < hp.numEpochs; nEpoch++) {
		randPerm(dataset.m_numSamples, randIndex);
		for (int nSamp = 0; nSamp < dataset.m_numSamples; nSamp++) {
			if (hp.findTrainError) {
				Result result(dataset.m_numClasses);
				model->eval(dataset.m_samples[randIndex[nSamp]], result);
				if (result.prediction
						!= dataset.m_samples[randIndex[nSamp]].y) {
					trainError[nEpoch]++;
				}
			}

			model->update(dataset.m_samples[randIndex[nSamp]]);
			if (hp.verbose && (nSamp % sampRatio) == 0) {
				cout << "--- " << model->name() << " training --- Epoch: "
						<< nEpoch + 1 << " --- ";
				cout << (10 * nSamp) / sampRatio << "%";
				cout << " --- Training error = " << trainError[nEpoch] << "/"
						<< nSamp << endl;
			}
		}
	}

	timeval endTime;
	gettimeofday(&endTime, NULL);
	cout << "--- " << model->name() << " training time = ";
	cout
			<< (endTime.tv_sec - startTime.tv_sec
					+ (endTime.tv_usec - startTime.tv_usec) / 1e6)
			<< " seconds." << endl;
}

vector<Result> test(Classifier * model, DataSet & dataset,
		Hyperparameters & hp) {
	timeval startTime;
	gettimeofday(&startTime, NULL);

	vector<Result> results;
	for (int nSamp = 0; nSamp < dataset.m_numSamples; nSamp++) {
		Result result(dataset.m_numClasses);
		model->eval(dataset.m_samples[nSamp], result);
		results.push_back(result);
	}

	double error = compError(results, dataset);
	if (hp.verbose) {
		cout << "--- " << model->name() << " test error: " << error << endl;
	}

	timeval endTime;
	gettimeofday(&endTime, NULL);
	cout << "--- " << model->name() << " testing time = ";
	cout
			<< (endTime.tv_sec - startTime.tv_sec
					+ (endTime.tv_usec - startTime.tv_usec) / 1e6)
			<< " seconds." << endl;

	return results;
}

vector<Result> trainAndTest(Classifier * model, DataSet & dataset_tr,
		DataSet & dataset_ts, Hyperparameters & hp) {
	timeval startTime;
	gettimeofday(&startTime, NULL);

	vector<Result> results;
	vector<int> randIndex;
	int sampRatio = dataset_tr.m_numSamples / 10;
	vector<double> trainError(hp.numEpochs, 0.0);
	vector<double> testError;
	for (int nEpoch = 0; nEpoch < hp.numEpochs; nEpoch++) {
		randPerm(dataset_tr.m_numSamples, randIndex);
		for (int nSamp = 0; nSamp < dataset_tr.m_numSamples; nSamp++) {
			if (hp.findTrainError) {
				Result result(dataset_tr.m_numClasses);
				model->eval(dataset_tr.m_samples[randIndex[nSamp]], result);
				if (result.prediction
						!= dataset_tr.m_samples[randIndex[nSamp]].y) {
					trainError[nEpoch]++;
				}
			}

			model->update(dataset_tr.m_samples[randIndex[nSamp]]);
			if (hp.verbose && (nSamp % sampRatio) == 0) {
				cout << "--- " << model->name() << " training --- Epoch: "
						<< nEpoch + 1 << " --- ";
				cout << (10 * nSamp) / sampRatio << "%";
				cout << " --- Training error = " << trainError[nEpoch] << "/"
						<< nSamp << endl;
			}
		}

		results = test(model, dataset_ts, hp);
		for (int idx = 0; idx < results.size(); idx++) {
			cout << results[idx].confidence[1] << endl;
		}

		testError.push_back(compError(results, dataset_ts));
	}

	timeval endTime;
	gettimeofday(&endTime, NULL);
	cout << "--- Total training and testing time = ";
	cout
			<< (endTime.tv_sec - startTime.tv_sec
					+ (endTime.tv_usec - startTime.tv_usec) / 1e6)
			<< " seconds." << endl;

	if (hp.verbose) {
		cout << endl << "--- " << model->name() << " test error over epochs: ";
		dispErrors(testError);
	}

	// Write the results
	string saveFile = hp.savePath + ".errors";
	ofstream file(saveFile.c_str(), ios::binary);
	if (!file) {
		cout << "Could not access " << saveFile << endl;
		exit(EXIT_FAILURE);
	}
	file << hp.numEpochs << " 1" << endl;
	for (int nEpoch = 0; nEpoch < hp.numEpochs; nEpoch++) {
		file << testError[nEpoch] << endl;
	}
	file.close();

	return results;
}

double compError(const vector<Result>& results, const DataSet& dataset) {
	double error = 0.0;
	for (int nSamp = 0; nSamp < dataset.m_numSamples; nSamp++) {
		if (results[nSamp].prediction != dataset.m_samples[nSamp].y) {
			error++;
		}
	}

	return error / dataset.m_numSamples;
}

void dispErrors(const vector<double>& errors) {
	for (int nSamp = 0; nSamp < (int) errors.size(); nSamp++) {
		cout << nSamp + 1 << ": " << errors[nSamp] << " --- ";
	}
	cout << endl;
}
