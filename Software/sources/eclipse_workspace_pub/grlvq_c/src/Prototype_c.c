/*
 * Prototype_c.c
 *
 *  Created on: Nov 21, 2012
 *      Author: Jeffrey F. Queisser
 */

#include "Prototype_c.h"

Prototype *prototype_create(unsigned int dimensionality, int label, bool learnrate_per_node, PROTOFRMT* weights)
{
	Prototype *proto = (Prototype *)malloc(sizeof(Prototype));

	proto->dimensionality=dimensionality;
	proto->label=label;

	proto->weights = (PROTOFRMT*)malloc(sizeof(PROTOFRMT)*(dimensionality));

	proto->metricWeights= NULL;
	proto->omegaMetricWeights= NULL;

	if (learnrate_per_node)
	{
		prototype_setSampleCounter(proto, 0);
	}

	if (weights == NULL)
	{
		unsigned int i;
		for (i=0; i<dimensionality; i++)
		{
			proto->weights[i]= rand()/(PROTOFRMT)RAND_MAX;
			proto->weights[i] = floorf(proto->weights[i] * 1000 + 0.5) / 1000;
		}
	}
	else
		prototype_setWeights(proto, weights);

	return proto;
}

Prototype *prototype_createFile(FILE *fd, int lambda_len, bool learnrate_per_node,PROTOFRMT* target_lambdas)
{
	Prototype *target = prototype_deserialize(fd, lambda_len, learnrate_per_node,target_lambdas);

	return target;
}



void prototype_del(Prototype* target)
{
	free(target->metricWeights);
	free(target->omegaMetricWeights);
	free(target->weights);
}

void prototype_serialize(Prototype* target, FILE* s, int lambda_len, bool learnrate_per_node)
{
	int aditionaldimensions=0;
	if (learnrate_per_node)aditionaldimensions=3;


	fwrite(&(target->dimensionality), sizeof(target->dimensionality), 1, s);
	fwrite(&(target->label), sizeof(target->label), 1, s);

	fwrite(target->weights, sizeof(PROTOFRMT), target->dimensionality+aditionaldimensions, s);

	if (lambda_len>0)
		fwrite(target->metricWeights, sizeof(PROTOFRMT), lambda_len, s);

}

Prototype* prototype_deserialize(FILE* s, int lambda_len, bool learnrate_per_node,PROTOFRMT* target_lambdas)
{
	Prototype *target = (Prototype *)malloc(sizeof(Prototype));


	int aditionaldimensions=0;
	if (learnrate_per_node)
		aditionaldimensions=3;



	fread(&(target->dimensionality), sizeof(target->dimensionality), 1, s);
	fread(&(target->label), sizeof(target->label), 1, s);


	target->weights=(PROTOFRMT*)malloc(sizeof(PROTOFRMT)*(target->dimensionality+aditionaldimensions));

	if (learnrate_per_node)
		target->weights[target->dimensionality]=0;

	if (lambda_len>0)
		target->metricWeights=(PROTOFRMT*)malloc(sizeof(PROTOFRMT)*lambda_len);


	fread(target->weights, sizeof(PROTOFRMT), target->dimensionality+aditionaldimensions, s);
	if (lambda_len>0)
		fread(target->metricWeights, sizeof(PROTOFRMT), lambda_len, s);
	else
		target->metricWeights=target_lambdas;


	return target;
}

int prototype_getLabel(Prototype* target)
{
	return target->label;
}


bool prototype_hasLabel(Prototype* target, int label)
{
	return target->label==label;
}


void prototype_setWeights(Prototype* target, PROTOFRMT *data)
{
	unsigned int i;
	for (i=0; i<target->dimensionality; i++)
	{
		target->weights[i]=data[i];
	}
}

void prototype_printWeights(Prototype* target){
	unsigned int d;
	for (d=0;d< target->dimensionality; d++)
		printf("weights[%i] %f\n", d, target->weights[d]);
}
int prototype_getSampleCounter(Prototype* target)
{
	return target->sampleCounter;
}

PROTOFRMT prototype_getLearnrate(Prototype* target)
{
	return target->learnrate;
}

PROTOFRMT prototype_getLearnrateMetricWeights(Prototype* target)
{
	return target->learnrateMetricWeights;
}

void prototype_setSampleCounter(Prototype* target, int sampleCounter)
{
	target->sampleCounter = sampleCounter;
}

void prototype_setLearnrate(Prototype* target, PROTOFRMT learnrate)
{
	target->learnrate = learnrate;
}

void prototype_setLearnrateMetricWeights(Prototype* target, PROTOFRMT learnrateMetricWeights)
{
	target->learnrateMetricWeights = learnrateMetricWeights;
}


PROTOFRMT prototype_distRel(Prototype* target, PROTOFRMT* datasample, PROTOFRMT* metricWeights)
{
	PROTOFRMT sum=0;
	unsigned int i;
	for (i=0; i<target->dimensionality; i++)
	{
		PROTOFRMT diff = datasample[i]-target->weights[i];
		sum +=metricWeights[i]*diff*diff;
	}
	return sum;
}

void printArray(PROTOFRMT* array, int dim) {
	unsigned d;
	printf("[");
	for (d = 0; d < dim; d++) {
		printf(" %f\n", array[d]);
	}
	printf("]\n");

}

void print2DArray_(PROTOFRMT* array2D, int dim) {
	unsigned row, col;
	for (row = 0; row < dim; row++) {
		printf("[");
		for (col = 0; col < dim; col++) {
			printf(" %f", array2D[row * dim + col]);
		}
		printf("]\n");
	}

}

PROTOFRMT prototype_dist(Prototype* target, PROTOFRMT* datasample)
{
	PROTOFRMT sum=0;
	unsigned int i;
	for (i=0; i<target->dimensionality; i++)
	{
		PROTOFRMT diff = datasample[i]-target->weights[i];
		sum += diff*diff;
	}
	return sum;
}

PROTOFRMT prototype_distmat(Prototype* target, PROTOFRMT* datasample, PROTOFRMT* metricWeights)
{
	PROTOFRMT diffA[target->dimensionality];

	unsigned int d, col,row;
	for (d=0; d<target->dimensionality; d++)
	{
		diffA[d] = datasample[d]-target->weights[d];
	}
	//print2DArray_(metricWeights, target->dimensionality);
	PROTOFRMT sum=0;
	for (row=0; row<target->dimensionality; row++)
	{
		PROTOFRMT sum_j=0;
		for (col=0; col<target->dimensionality; col++)
		{
			sum_j+=metricWeights[row*target->dimensionality + col] *diffA[col];
		}

		sum+=sum_j*diffA[row];
	}

	return sum;

}





