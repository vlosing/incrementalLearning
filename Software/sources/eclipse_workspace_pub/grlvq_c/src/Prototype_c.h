/*
 * Prototype_c.h
 *
 *  Created on: Nov 21, 2012
 *      Author: Jeffrey F. Queisser
 */

#ifndef PROTOTYPE_H_
#define PROTOTYPE_H_


#define bool int
#define false   0
#define true    1





#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#define PROTOFRMT double
#define PROTOMAX DBL_MAX



struct Prototype_struct
{

	PROTOFRMT* weights;
	PROTOFRMT* metricWeights;
	PROTOFRMT* omegaMetricWeights;
	unsigned int dimensionality;
	int label;

	//only needed for prototype specific learnrate
	int sampleCounter;
	PROTOFRMT learnrate;
	PROTOFRMT learnrateMetricWeights;

};
typedef struct Prototype_struct Prototype;




Prototype *prototype_create(unsigned int dimensionality, int label, bool learnrate_per_node, PROTOFRMT* weights);


Prototype *prototype_createFile(FILE* fd, int lambda_len, bool learnrate_per_node,PROTOFRMT* target_lambdas);

void prototype_del(Prototype* target);

void prototype_serialize(Prototype* target, FILE* s, int lambda_len, bool learnrate_per_node);
Prototype *prototype_deserialize(FILE* s, int lambda_len, bool learnrate_per_node,PROTOFRMT* target_lambdas);

int prototype_getLabel(Prototype* target);
bool prototype_hasLabel(Prototype* target, int label);
void prototype_setWeights(Prototype* target, PROTOFRMT *data);


int prototype_getSampleCounter(Prototype* target);

PROTOFRMT prototype_getLearnrate(Prototype* target);

PROTOFRMT prototype_getLearnrateMetricWeights(Prototype* target);

void prototype_setSampleCounter(Prototype* target, int sampleCounter);

void prototype_setLearnrate(Prototype* target, PROTOFRMT epsilon);

void prototype_setLearnrateMetricWeights(Prototype* target, PROTOFRMT epsilonLambda);

void prototype_printWeights(Prototype* target);

PROTOFRMT prototype_dist(Prototype* target, PROTOFRMT* datasample);
PROTOFRMT prototype_distRel(Prototype* target, PROTOFRMT* datasample, PROTOFRMT* metricWeights);
PROTOFRMT prototype_distmat(Prototype* target, PROTOFRMT* datasample, PROTOFRMT* metricWeights);


#endif /* PROTOTYPE_H_ */
