/*
 * Network_c.h
 * GRLVQ, LGRLVQ, GMLVQ, LGMLVQ, RLVQ network implementation.
 * Train process implemented according to publications by Barbara Hammer.
 *
 *  Created on: Nov 21, 2012
 *      Author: Jeffrey F. Queisser
 */

#ifndef NETWORK_H_
#define NETWORK_H_


#include <float.h>
#include "Prototype_c.h"

#include <math.h>
#include <string.h>


enum Mode
{
	GLVQ,
	GRLVQ,
	GMLVQ,
	LGRLVQ,
	LGMLVQ
};

struct prototypeCandidate_struct
{
	int label;
	PROTOFRMT dist;
	PROTOFRMT *data;
	struct prototypeCandidate_struct *next;
};
typedef struct prototypeCandidate_struct prototypeCandidate;

void prototypeCandidate_init(prototypeCandidate **candidateList, PROTOFRMT* sample, int sampleLabel, PROTOFRMT distance, int dimensionality);

struct Network
{
	/*Data dimensionality*/
	unsigned int dimensionality;
	/*List of network prototypes*/
	struct pvec
	{
		unsigned int size;
		Prototype **entry;
	}prototypes;



	/*Array of metric weights and weight matrix if matrix mode is enabled*/
	PROTOFRMT *metricWeights;

	PROTOFRMT *omegaMetricWeights;


	/*Number of usable threads*/
	unsigned int max_threads;

	/*Learn rates for network*/
	PROTOFRMT learnrate, learnrate_start;
	/*Learn rates for metric weights*/
	PROTOFRMT learnrate_metricWeights, learnrate_metricWeights_start;

	/*Number of samples used for learn rate annealing*/
	unsigned int maxsamples;
	/*Current number of presented training samples*/
	unsigned int datasamples;

	/*Indicates if each node has its own learn rate*/
	bool learnrate_per_node;

	/*Current mode of network operation*/
	enum Mode mode;

	/*Misclassification counter for node insertion*/
	unsigned int currentErrorCount;

	/*List of failed prototypes including their label and error distance*/
	prototypeCandidate *candidateList;

	struct pvec2
	{
		unsigned int size;
		PROTOFRMT **entry;
	}sampleBag;
	unsigned int maxSampleBagSize;
};
typedef struct Network Network_c;

/**
 * Constructor that initializes a new network
 *
 * example default parameters:
 * network_create(enum Mode mode, unsigned int dimensionality, bool learnrate_per_node=false, unsigned int protocount=10, unsigned int maxsamples=10000, unsigned int maxthreads=1, unsigned int initialdimensions=0);
 *
 * @param mode					Network Type
 * @param dimensionality		Input data dimensionality
 * @param learnrate_per_node	Node specific learn rate
 * @param protocount			Uses the first -protocount- trainingsamples as prototypes
 * @param maxsamples			Number of training steps used for learn rate annealing
 * @param maxthreads			Partition nearest neighbor search into -maxthreads- thread (by nodes, not dimensionality)
 */
struct Network* network_create(enum Mode mode, unsigned int dimensionality, bool learnrate_per_node, unsigned int maxsamples, unsigned int maxthreads);


/**
 * Debug output.
 * Print current metric weights.
 */
void network_printMetricWeights(struct Network *net);

/**
 * Debug output.
 * Print the number of prototypes.
 */
void network_printStatus(struct Network *net);


/**
 * Set the number of used threads.
 * This defines the number of threads used for the nearest neighbor search.
 * Node search is distributed over the given number of threads,
 * this can reduce the searchtime for networks with many nodes.
 *
 * @param threads				Number of threads
 */
void network_setMaxThreads(struct Network *net, unsigned int threads);

/**
 * Define the metric weights-learn rate.
 * Learn rates are defined by a start value and a linear decent to start*0.001
 * at maxsamples given training views.
 * As proposed by Hammer, metric weigts learn rates should be decades smaller than
 * network learn rates.
 *
 * @param val					metric weights train rate
 */
void network_setLearnrateMetricWeightsStart(struct Network *net, PROTOFRMT val);


void network_setLearnrateStart(struct Network *net, PROTOFRMT val);


/**
 * Train the network.
 * This method trains the network using the update rule given by Hammer.
 * No node insertion is performed.
 *
 * example default parameters:
 * network_trainStep(struct Network *net, PROTOFRMT *data, int *label,unsigned int data_rows=1);
 *
 * @param data				Train data array, length: dimensionality*trainingsamples
 * @param label				Array containing labels for each training sample, length: trainingsamples
 * @param data_rows			Defines the number of trainingsamples
 */
PROTOFRMT network_train(struct Network *net, PROTOFRMT *data, int *label, unsigned int data_rows, bool adaptMetricWeights);

/**
 * Iterative network training.
 * See @see{trainStep} for parameter description
 *
 * example default parameters:
 * network_trainStepIncremental(struct Network *net, PROTOFRMT *data, int *label,int tmode=1, bool is_failure_increment=false, unsigned int g_max=10, unsigned int data_rows=1, bool random_selection=false, double *perftarget=NULL);
 *
 * @param doInsertions 	Activate node insertion
 * @param g_max					Miscalssification threshold for node insertion
 * @param perftarget			If not NULL: Receives the classification performance of the given samples.
 */
PROTOFRMT network_trainIncremental(struct Network *net, PROTOFRMT *data, int *label, bool doInsertions, unsigned int g_max, unsigned int data_rows, double *classificationPerformance, bool adaptMetricWeights);


/**
 * Add a new prototype to network.
 *
 * @param label					Prototype label
 * @param initdata				Prototype initialization data, is random if NULL is given
 */
void network_addPrototype(struct Network *net, int label, PROTOFRMT *weights);

/**
 * Process a number of input samples.
 *
 * @param data					Input data array, length: dimensionality*data_rows
 * @param result				Classification result, length: data_rows
 * @param data_rows				Number of samples
 * @param indices				Optional, receives the nearest prototype number if not NULL
 */
void network_processData(struct Network *net, PROTOFRMT *data, int *result, unsigned int data_rows, unsigned int *indices);

/**
 * Set the target for learnrate annealing.
 *
 * @param maxsamples			Defines the number of presented samples so that the trainingrate reaches start*0.001
 */
void network_setMaxsamples(struct Network *net, unsigned int maxsamples);


/**
 * Get the dimensionality
 *
 * @return 						dimensionality of network nodes
 */
unsigned int network_getDimensionality(struct Network *net);

/**
 * Clear network structure.
 * Release memory.
 */
void network_clear(struct Network *net);

/**
 * Get number of prototypes in network.
 *
 * @return 						Number of used Prototypes
 */
unsigned int network_getPrototypes(struct Network *net);

/**
 * Get label of specific prototype.
 *
 * @param protonr				Prototype ID
 * @return 						Label
 */
int network_getPrototypeLabel(struct Network *net, unsigned int protonr);

/**
 * Get weights of specific prototype.
 *
 * @param protonr				Prototype ID
 * @param target				Receives the prototype weights, length: dimensionality
 * @return 						True on success
 */
bool network_getPrototypeData(struct Network *net, unsigned int protonr, PROTOFRMT* target);

/**
 * Get metric weights.
 *
 * @param protonr				Prototype ID (only different results if per node weights is activated)
 * @param target				Receives the labda weights, length: dimensionality or dimensionality*dimensionality(matrix mode) @see{getMetricWeightsLen()}
 * @return 						True on success
 */
bool network_getMetricWeights(struct Network *net, unsigned int protonr, PROTOFRMT* target);

/**
 * Get matrix eingenvectors.
 * Use only with LGMLVQ or GMLVQ mode
 *
 * @param protonr				Prototype ID
 * @param target				Receives the eigenvectors, length: dimensionality*dimensionality
 * @return 						True on success
 */
bool network_getEigenvectors(struct Network *net, unsigned int protonr, PROTOFRMT *target);

/**
 * Returns the length of metric weights.
 *
 * @return 						dimensionality*dimensionality on matrix mode, else: dimensionality
 */
unsigned int network_getMetricWeightsLen(struct Network *net);


void network_addPrototypeToList(struct Network *net, Prototype *np);


#endif /* NETWORK_H_ */
