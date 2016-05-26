/*
 * Network_c.c
 * GRLVQ, LGRLVQ, GMLVQ, LGMLVQ, RLVQ network implementation.
 * Train process implemented according to publications by Barbara Hammer.
 *
 *  Created on: Nov 21, 2012
 *      Author: Jeffrey F. Queisser
 */

#include "Network_c.h"
#include <math.h>
#include "PrototypeSearch.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void prototypeCandidate_init(prototypeCandidate **candidateList,
PROTOFRMT* sample, int sampleLabel, PROTOFRMT distance, int dimensionality) {
	(*candidateList) = (prototypeCandidate*) malloc(sizeof(prototypeCandidate));
	(*candidateList)->next = 0;
	(*candidateList)->label = sampleLabel;
	(*candidateList)->dist = distance;
	(*candidateList)->data = (PROTOFRMT*) malloc(
			sizeof(PROTOFRMT) * dimensionality);
	memcpy((*candidateList)->data, sample, sizeof(PROTOFRMT) * dimensionality);
}

PROTOFRMT * getMetricWeights(struct Network *net, Prototype *proto) {
	switch (net->mode) {
	case GRLVQ:
	case GMLVQ:
		return net->metricWeights;
	case LGRLVQ:
	case LGMLVQ:
		return proto->metricWeights;
	default:
		return NULL;
	}

}

PROTOFRMT * getOmegaMetricWeights(struct Network *net, Prototype *proto) {
	switch (net->mode) {
	case GMLVQ:
		return net->omegaMetricWeights;
	case LGMLVQ:
		return proto->omegaMetricWeights;
	default:
		return NULL;
	}

}

/**
 * Internal helper method for metric weights adoption(normalization) in matrix mode.
 *
 * @param target				If not NULL: Select prototype if in LGMLVQ mode
 */
void network_refreshGMLVQMetricWeights(struct Network *net,
PROTOFRMT* metricWeights, PROTOFRMT* omegaMetricWeights) {
	unsigned int col, row, k;
	for (col = 0; col < net->dimensionality; col++) {
		for (row = 0; row < net->dimensionality; row++) {
			PROTOFRMT sum_k = 0;
			for (k = 0; k < net->dimensionality; k++) //sum over elements
					{
				sum_k += omegaMetricWeights[row * net->dimensionality + k]
						* omegaMetricWeights[k * net->dimensionality + col];
			}

			metricWeights[row * net->dimensionality + col] = sum_k;
		}
	}
}

void print2DArray(PROTOFRMT* array2D, int dim) {
	unsigned row, col;
	for (row = 0; row < dim; row++) {
		printf("[");
		for (col = 0; col < dim; col++) {
			printf(" %f", array2D[row * dim + col]);
		}
		printf("]\n");
	}

}

void net_initMetricWeights(struct Network* net) {
	unsigned int i, j;
	switch (net->mode) {
	case GRLVQ:
		net->metricWeights = (PROTOFRMT *) malloc(
				sizeof(PROTOFRMT) * net->dimensionality);

		for (i = 0; i < net->dimensionality; i++) {
			net->metricWeights[i] = 1. / net->dimensionality;
		}
		break;

	case GMLVQ:
		net->metricWeights = (PROTOFRMT*) malloc(
				sizeof(PROTOFRMT) * net->dimensionality * net->dimensionality);
		net->omegaMetricWeights = (PROTOFRMT*) malloc(
				sizeof(PROTOFRMT) * net->dimensionality * net->dimensionality);
		PROTOFRMT targetval = sqrt(1.0 / net->dimensionality);
		for (i = 0; i < net->dimensionality; i++) {
			for (j = 0; j < net->dimensionality; j++) {
				if (i == j)
					net->omegaMetricWeights[i * net->dimensionality + j] =
							targetval;
				else
					net->omegaMetricWeights[i * net->dimensionality + j] = 0;
			}
		}
		network_refreshGMLVQMetricWeights(net, net->metricWeights,
				net->omegaMetricWeights);
		break;
	default:
		net->metricWeights = NULL;
		break;
	}

}

/**
 * Internal method.
 * Refresh learn rates.
 * Gets called by @see{trainStep} and should NOT get called manually!
 *
 * @param target				Select node dependent learn rate if not NULL
 */
void network_refreshLearnrateIntern(struct Network *net, Prototype *proto) {
	PROTOFRMT min = 0.001;
	PROTOFRMT fac;

	if (proto) {
		int counter = prototype_getSampleCounter(proto);
		counter++;
		prototype_setSampleCounter(proto, counter);
		fac = MAX(
				1 -(1 - min)*((PROTOFRMT)counter / (PROTOFRMT) net->maxsamples),
				min);

		prototype_setLearnrate(proto, net->learnrate_start * fac);
		prototype_setLearnrateMetricWeights(proto,
				net->learnrate_metricWeights_start * fac);
	} else {
		net->datasamples++;
		if (net->datasamples < net->maxsamples)
			fac =
					MAX(
							1 -(1 - min)*(net->datasamples / (PROTOFRMT) net->maxsamples),
							min);

		net->learnrate = net->learnrate_start * fac;
		net->learnrate_metricWeights = net->learnrate_metricWeights_start * fac;
	}
}

void network_refreshLearnrate(struct Network *net, Prototype *winner,
		Prototype *looser) {
	if (net->learnrate_per_node) {
		network_refreshLearnrateIntern(net, winner);
		network_refreshLearnrateIntern(net, looser);
	} else {
		network_refreshLearnrateIntern(net, NULL);
	}
}

/**
 * Constructor that initializes a new network
 *
 * @param mode					Network Type
 * @param dimensionality		Input data dimensionality
 * @param learnrate_per_node	Node specific learn rate
 * @param protocount			Uses the first -protocount- trainingsamples as prototypes
 * @param maxsamples			Number of training steps used for learn rate annealing
 * @param maxthreads			Partition nearest neighbor search into -maxthreads- thread (by nodes, not dimensionality)
 */
struct Network* network_create(enum Mode mode, unsigned int dimensionality,
bool learnrate_per_node, unsigned int maxsamples, unsigned int maxthreads) {
	struct Network* net = (struct Network*) malloc(sizeof(struct Network));

	net->prototypes.size = 0;
	net->prototypes.entry = NULL;

	net->maxSampleBagSize = 1000;
	net->sampleBag.size = 0;
	net->sampleBag.entry = NULL;

	net->candidateList = NULL;

	net->learnrate_metricWeights = net->learnrate_metricWeights_start = 0.001;
	net->learnrate = net->learnrate_start = 0.1;
	net->metricWeights = NULL;
	net->omegaMetricWeights = NULL;
	net->mode = mode;
	net->learnrate_per_node = learnrate_per_node;

	net->currentErrorCount = 0;

	net->maxsamples = maxsamples;
	net->datasamples = 0;

	net->dimensionality = dimensionality;
	net->max_threads = maxthreads;

	net_initMetricWeights(net);



	return net;
}
;

/**
 * Set the number of used threads.
 * This defines the number of threads used for the nearest neighbor search.
 * Node search is distributed over the given number of threads,
 * this can reduce the searchtime for networks with many nodes.
 *
 * @param threads				Number of threads
 */
void network_setMaxThreads(struct Network *net, unsigned int threads) {
	net->max_threads = threads;
}
;

/**
 * Define the metricWeights-learn rate.
 * Learn rates are defined by a start value and a linear decent to start*0.001
 * at maxsamples given training views.
 * As proposed by Hammer, metricWeights learn rates should be decades smaller than
 * network learn rates.
 *
 * @param val					metricWeights train rate
 */
void network_setLearnrateMetricWeightsStart(struct Network *net, PROTOFRMT val) {
	net->learnrate_metricWeights_start = val;
	net->learnrate_metricWeights = val;
}
;

void network_setLearnrateStart(struct Network *net, PROTOFRMT val) {
	net->learnrate_start = val;
	net->learnrate = val;
}

/**
 * Debug output.
 * Print current metric weights.
 */
void network_printMetricWeights(struct Network *net) {
	printf("metric weights:\n");

	unsigned int d;
	switch (net->mode) {
	case GRLVQ:
		for (d = 0; d < net->dimensionality; d++) {
			printf("%f\n", net->metricWeights[d]);
		}
		break;
	case GMLVQ:
		for (d = 0; d < net->dimensionality; d++) {
			printf("%f\n", net->metricWeights[d * net->dimensionality + d]);
		}
		break;
	default:
		printf("not a proper type!!\n");
		break;
	}
}
;

/**
 * Debug output.
 * Print the number of prototypes.
 */
void network_printStatus(struct Network *net) {
	printf("Prototypes:%u\n", net->prototypes.size);
}
;

int getFacValues(PROTOFRMT winner_dist, PROTOFRMT looser_dist,
PROTOFRMT *fac_winner, PROTOFRMT *fac_looser) {
	PROTOFRMT diff = (winner_dist + looser_dist) * (winner_dist + looser_dist);
	if (diff <= 0.001)
		return 0;
	*fac_winner = looser_dist / diff;
	*fac_looser = winner_dist / diff;
	return 1;
}

void updateWeights(struct Network *net, Prototype *winner, Prototype *looser,
PROTOFRMT fac_winner, PROTOFRMT fac_looser, PROTOFRMT* delta_winner,
PROTOFRMT* delta_looser, PROTOFRMT *delta_winner_result,
PROTOFRMT *delta_looser_result) {
	PROTOFRMT learnrate_winner;
	PROTOFRMT learnrate_looser;

	if (net->learnrate_per_node) {
		learnrate_winner = prototype_getLearnrate(winner);
		learnrate_looser = prototype_getLearnrate(looser);
	} else {
		learnrate_winner = learnrate_looser = net->learnrate;
	}
	unsigned int d;
	for (d = 0; d < net->dimensionality; d++) {
		PROTOFRMT update_winner;
		PROTOFRMT update_looser;
		update_winner = learnrate_winner * fac_winner * delta_winner[d];
		update_looser = learnrate_looser * fac_looser * delta_looser[d];

		if (net->mode != GLVQ) {
			update_winner *= getMetricWeights(net, winner)[d];
			update_looser *= getMetricWeights(net, looser)[d];
		}

		*delta_winner_result += update_winner * update_winner;
		winner->weights[d] += update_winner;

		*delta_looser_result += update_looser * update_looser;
		looser->weights[d] -= update_looser;
	}
}

void getLearnrateMetricWeights_RLVQ(struct Network *net, Prototype *winner,
		Prototype *looser, PROTOFRMT *learnrate_metricWeights_winner,
		PROTOFRMT *learnrate_metricWeights_looser) {
	if (net->learnrate_per_node) {
		if (net->mode == GRLVQ)
			*learnrate_metricWeights_winner =
					(prototype_getLearnrateMetricWeights(winner)
							+ prototype_getLearnrateMetricWeights(looser))
							/ 2.0;
		else if (net->mode == LGRLVQ)
			*learnrate_metricWeights_winner =
					prototype_getLearnrateMetricWeights(winner);
		*learnrate_metricWeights_looser = prototype_getLearnrateMetricWeights(
				looser);
	} else {
		*learnrate_metricWeights_winner = net->learnrate_metricWeights;
		*learnrate_metricWeights_looser = net->learnrate_metricWeights;
	}
}

void updateMetricWeight_RLVQ(PROTOFRMT *metricWeight,
PROTOFRMT learnrateMetricWeights,
PROTOFRMT fac_winner, PROTOFRMT fac_looser, PROTOFRMT delta_winner,
PROTOFRMT delta_looser) {
	*metricWeight -= learnrateMetricWeights
			* (fac_winner * delta_winner * delta_winner
					- fac_looser * delta_looser * delta_looser);
	if (*metricWeight <= 0)
		*metricWeight = 0;
}

void normalizeMetricWeights(int size, PROTOFRMT *metricWeights, PROTOFRMT value) {
	unsigned int d;
	for (d = 0; d < size; d++) {
		metricWeights[d] /= value;
	}
}

void updateMetricWeightsGRLVQ(struct Network *net, Prototype *winner,
		Prototype *looser, PROTOFRMT fac_winner, PROTOFRMT fac_looser,
		PROTOFRMT* delta_winner, PROTOFRMT* delta_looser) {
	PROTOFRMT learnrateMetricWeights, dummy;
	getLearnrateMetricWeights_RLVQ(net, winner, looser, &learnrateMetricWeights,
			&dummy);

	PROTOFRMT sum_pos = 0;
	unsigned int d;
	for (d = 0; d < net->dimensionality; d++) {
		updateMetricWeight_RLVQ(&(net->metricWeights[d]),
				learnrateMetricWeights, fac_winner, fac_looser, delta_winner[d],
				delta_looser[d]);
		sum_pos += net->metricWeights[d];
	}
	normalizeMetricWeights(net->dimensionality, net->metricWeights, sum_pos);
}

void updateMetricWeightsLGRLVQ(struct Network *net, Prototype *winner,
		Prototype *looser, PROTOFRMT fac_winner, PROTOFRMT fac_looser,
		PROTOFRMT* delta_winner, PROTOFRMT* delta_looser) {
	PROTOFRMT learnrate_metricWeights_winner, learnrate_metricWeights_looser;
	getLearnrateMetricWeights_RLVQ(net, winner, looser,
			&learnrate_metricWeights_winner, &learnrate_metricWeights_looser);

	PROTOFRMT sum_pos = 0;
	PROTOFRMT sum_neg = 0;
	unsigned int d;
	for (d = 0; d < net->dimensionality; d++) {
		updateMetricWeight_RLVQ(&(winner->metricWeights[d]),
				learnrate_metricWeights_winner, fac_winner, fac_looser,
				delta_winner[d], delta_looser[d]);
		updateMetricWeight_RLVQ(&(looser->metricWeights[d]),
				learnrate_metricWeights_looser, fac_winner, fac_looser,
				delta_winner[d], delta_looser[d]);
		sum_pos += winner->metricWeights[d];
		sum_neg += looser->metricWeights[d];
	}
	normalizeMetricWeights(net->dimensionality, winner->metricWeights, sum_pos);
	normalizeMetricWeights(net->dimensionality, looser->metricWeights, sum_neg);
}

void getDeltas(struct Network *net, Prototype *winner, Prototype *looser,
PROTOFRMT *data, PROTOFRMT * delta_winner, PROTOFRMT * delta_looser) {
	unsigned int d;
	for (d = 0; d < net->dimensionality; d++) {
		delta_winner[d] = data[d] - winner->weights[d];
		delta_looser[d] = data[d] - looser->weights[d];
	}
}

void updateWeights_GMLVQ(struct Network *net, Prototype *winner,
		Prototype *looser, PROTOFRMT fac_winner, PROTOFRMT fac_looser,
		PROTOFRMT* delta_winner, PROTOFRMT* delta_looser,
		PROTOFRMT* delta_winner_result, PROTOFRMT* delta_looser_result) {
	PROTOFRMT metricWeights_winner[net->dimensionality];
	PROTOFRMT metricWeights_looser[net->dimensionality];
	unsigned int row, col;
	PROTOFRMT* winnerMetricWeights = getMetricWeights(net, winner);
	PROTOFRMT* looserMetricWeights = getMetricWeights(net, looser);

	for (row = 0; row < net->dimensionality; row++) {
		metricWeights_winner[row] = 0;
		metricWeights_looser[row] = 0;
		for (col = 0; col < net->dimensionality; col++) {

			PROTOFRMT metricWeightVal_winner = winnerMetricWeights[row
					* net->dimensionality + col];
			PROTOFRMT metricWeightVal_looser = looserMetricWeights[row
					* net->dimensionality + col];
			metricWeights_winner[row] += delta_winner[col]
					* metricWeightVal_winner;
			metricWeights_looser[row] += delta_looser[col]
					* metricWeightVal_looser;
		}
	}

	PROTOFRMT learnrate_winner, learnrate_looser;
	if (net->learnrate_per_node) {
		learnrate_winner = prototype_getLearnrate(winner);
		learnrate_looser = prototype_getLearnrate(looser);
	} else {
		learnrate_winner = learnrate_looser = net->learnrate;
	}
	unsigned int d;
	for (d = 0; d < net->dimensionality; d++) {
		PROTOFRMT delta_pos_ = learnrate_winner * fac_winner
				* metricWeights_winner[d];
		*delta_winner_result += delta_pos_ * delta_pos_;
		winner->weights[d] += delta_pos_;

		PROTOFRMT delta_neg_ = learnrate_looser * fac_looser
				* metricWeights_looser[d];
		*delta_looser_result += delta_neg_ * delta_neg_;
		looser->weights[d] -= delta_neg_;
	}
}

void updateMetricWeights_GMLVQ(struct Network *net, Prototype *winner,
		Prototype *looser, PROTOFRMT fac_winner, PROTOFRMT fac_looser,
		PROTOFRMT* delta_winner, PROTOFRMT* delta_looser) {

	PROTOFRMT* winnerOmegaMetricWeights = getOmegaMetricWeights(net, winner);
	PROTOFRMT* looserOmegaMetricWeights = getOmegaMetricWeights(net, looser);

	unsigned int row, col;
	PROTOFRMT omega_pos_pos[net->dimensionality];
	PROTOFRMT omega_neg_pos[net->dimensionality];
	PROTOFRMT omega_pos_neg[net->dimensionality];
	PROTOFRMT omega_neg_neg[net->dimensionality];
	for (row = 0; row < net->dimensionality; row++) {
		omega_pos_pos[row] = 0;
		omega_neg_pos[row] = 0;
		omega_pos_neg[row] = 0;
		omega_neg_neg[row] = 0;
		for (col = 0; col < net->dimensionality; col++) {
			PROTOFRMT omegaval_winner = winnerOmegaMetricWeights[row
					* net->dimensionality + col];
			omega_pos_pos[row] += delta_winner[col] * omegaval_winner;
			omega_neg_pos[row] += delta_looser[col] * omegaval_winner;
			if (net->mode == LGMLVQ) {
				PROTOFRMT omegaval_looser = looserOmegaMetricWeights[row
						* net->dimensionality + col];
				omega_pos_neg[row] += delta_winner[col] * omegaval_looser;
				omega_neg_neg[row] += delta_looser[col] * omegaval_looser;
			}
		}
	}

	PROTOFRMT learnrate_metricWeights_winner, learnrate_metricWeights_looser;
	if (net->learnrate_per_node) {
		learnrate_metricWeights_winner = prototype_getLearnrateMetricWeights(
				winner);
		learnrate_metricWeights_looser = prototype_getLearnrateMetricWeights(
				looser);
	} else {
		learnrate_metricWeights_winner = learnrate_metricWeights_looser =
				net->learnrate_metricWeights;
	}

	PROTOFRMT sum_pos = 0;
	PROTOFRMT sum_neg = 0;
	for (row = 0; row < net->dimensionality; row++) {
		for (col = 0; col < net->dimensionality; col++) {
			int pos = row * net->dimensionality + col;
			PROTOFRMT part_pos_pos = omega_pos_pos[col] * delta_winner[row]
					+ omega_pos_pos[row] * delta_winner[col];
			PROTOFRMT part_neg_pos = omega_neg_pos[col] * delta_looser[row]
					+ omega_neg_pos[row] * delta_looser[col];
			winnerOmegaMetricWeights[pos] -= learnrate_metricWeights_winner
					* (fac_winner * part_pos_pos - fac_looser * part_neg_pos);

			if (net->mode == LGMLVQ) {
				PROTOFRMT part_pos_neg = omega_pos_neg[col] * delta_winner[row]
						+ omega_pos_neg[row] * delta_winner[col];
				PROTOFRMT part_neg_neg = omega_neg_neg[col] * delta_looser[row]
						+ omega_neg_neg[row] * delta_looser[col];
				looserOmegaMetricWeights[pos] -=
						learnrate_metricWeights_looser
								* (fac_winner * part_pos_neg
										- fac_looser * part_neg_neg);

				sum_neg += looserOmegaMetricWeights[pos]
						* looserOmegaMetricWeights[pos];
			}
			sum_pos += winnerOmegaMetricWeights[pos]
					* winnerOmegaMetricWeights[pos];
		}
	}
	normalizeMetricWeights(net->dimensionality * net->dimensionality,
			winnerOmegaMetricWeights, sum_pos);
	if (net->mode == LGMLVQ)
		normalizeMetricWeights(net->dimensionality * net->dimensionality,
				looserOmegaMetricWeights, sum_neg);
}

/**
 * Internal method.
 * Perform a GRLVQ, LGRLVQ or RLVQ training step
 * Gets called by train methods and should not get called manually!
 *
 * @param data					Training data, length: dimensionality
 * @param winner				Closest prototype, same class to curr sample
 * @param winner_dist			Distance of the given winner prototype
 * @param looser				Closest prototype, another class
 * @param looser_dist			Distance of the given looser prototype to curr sample
 */
PROTOFRMT network_adapt_GRLVQ(struct Network *net, PROTOFRMT *data,
		Prototype *winner, PROTOFRMT winner_dist, Prototype *looser,
		PROTOFRMT looser_dist, bool adaptMetricWeights) {
	PROTOFRMT delta_winner_result = 0;
	PROTOFRMT delta_looser_result = 0;
	PROTOFRMT fac_winner = 0;
	PROTOFRMT fac_looser = 0;
	if (getFacValues(winner_dist, looser_dist, &fac_winner, &fac_looser) == 0)
		return 0;

	PROTOFRMT delta_winner[net->dimensionality];
	PROTOFRMT delta_looser[net->dimensionality];
	getDeltas(net, winner, looser, data, delta_winner, delta_looser);

	updateWeights(net, winner, looser, fac_winner, fac_looser, delta_winner,
			delta_looser, &delta_winner_result, &delta_looser_result);
	if (adaptMetricWeights) {
		switch (net->mode) {
		case GRLVQ: {
			updateMetricWeightsGRLVQ(net, winner, looser, fac_winner,
					fac_looser, delta_winner, delta_looser);
			break;
		}
		case LGRLVQ: {
			updateMetricWeightsLGRLVQ(net, winner, looser, fac_winner,
					fac_looser, delta_winner, delta_looser);

			break;
		}
		default:
			break;
		}
	}
	return (delta_winner_result + delta_looser_result) / 2;
}

/**
 * Internal method.
 * Perform a GMLVQ or LGMLVQ training step
 * Gets called by train methods and should not get called manually!
 *
 * @param data					Training data, length: dimensionality
 * @param winner				Closest prototype, same class to curr. sample
 * @param winner_dist			Distance of the given winner prototype
 * @param looser				Closest prototype, another class
 * @param looser_dist			Distance of the given looser prototype to curr. sample
 */
PROTOFRMT network_adapt_GMLVQ(struct Network *net, PROTOFRMT *data,
		Prototype *winner, PROTOFRMT winner_dist, Prototype *looser,
		PROTOFRMT looser_dist, bool adaptMetricWeights) {
	PROTOFRMT delta_winner_result = 0;
	PROTOFRMT delta_looser_result = 0;

	PROTOFRMT fac_winner = 0;
	PROTOFRMT fac_looser = 0;
	if (getFacValues(winner_dist, looser_dist, &fac_winner, &fac_looser) == 0)
		return 0;
	PROTOFRMT delta_winner[net->dimensionality];
	PROTOFRMT delta_looser[net->dimensionality];
	getDeltas(net, winner, looser, data, delta_winner, delta_looser);

	updateWeights_GMLVQ(net, winner, looser, fac_winner, fac_looser,
			delta_winner, delta_looser, &delta_winner_result,
			&delta_looser_result);
	if (adaptMetricWeights) {
		updateMetricWeights_GMLVQ(net, winner, looser, fac_winner, fac_looser,
				delta_winner, delta_looser);

		if (net->mode == LGMLVQ) {
			network_refreshGMLVQMetricWeights(net, winner->metricWeights,
					winner->omegaMetricWeights);
			network_refreshGMLVQMetricWeights(net, looser->metricWeights,
					looser->omegaMetricWeights);
		} else {
			network_refreshGMLVQMetricWeights(net, net->metricWeights,
					net->omegaMetricWeights);
		}
	}

	return (delta_winner_result + delta_looser_result) / 2;
}

bool getWinnerLooserProto(struct Network *net, PROTOFRMT *inputData,
		int inputLabel, Prototype** winner, Prototype** looser,
		PROTOFRMT *winner_dist, PROTOFRMT *looser_dist) {

	unsigned int winner_index = network_findWinnerProto(net, inputData,
			inputLabel, winner_dist);

	unsigned int looser_index = network_findLooserProto(net, inputData,
			inputLabel, looser_dist);

	if (*winner_dist == PROTOMAX) {
		printf("no winner prototype found for label %d -> new inserted\n",
				inputLabel);
		network_addPrototype(net, inputLabel, inputData);
		return false;
	} else if (*looser_dist == PROTOMAX) {
		printf("no looser prototype found for label %d -> sample is ignored\n",
				inputLabel);
		return false;
	}
	*winner = net->prototypes.entry[winner_index];
	*looser = net->prototypes.entry[looser_index];
	return true;
}

/**
 * Train the network.
 * This method trains the network using the update rule given by Hammer.
 * No node insertion is performed.
 *
 * @param data				Train data array, length: dimensionality*trainingsamples
 * @param label				Array containing labels for each training sample, length: trainingsamples
 * @param data_rows			Defines the number of trainingsamples
 * @param random_selection	Select training samples in a random order
 */
PROTOFRMT network_train(struct Network *net, PROTOFRMT *data, int *label,
		unsigned int data_rows, bool adaptMetricWeights) {

	PROTOFRMT delta_sum = 0;
	unsigned int i;
	for (i = 0; i < data_rows; i++) {
		PROTOFRMT* sampledata = data + net->dimensionality * i;
		int samplelabel = label[i];
		Prototype* winner = NULL;
		Prototype* looser = NULL;
		PROTOFRMT winner_dist;
		PROTOFRMT looser_dist;

		if (getWinnerLooserProto(net, sampledata, samplelabel, &winner, &looser,
				&winner_dist, &looser_dist)) {
			switch (net->mode) {
			case GLVQ:
			case GRLVQ:
			case LGRLVQ:
				delta_sum += network_adapt_GRLVQ(net, sampledata, winner,
						winner_dist, looser, looser_dist, adaptMetricWeights);
				break;
			case GMLVQ:
			case LGMLVQ:
				delta_sum += network_adapt_GMLVQ(net, sampledata, winner,
						winner_dist, looser, looser_dist, adaptMetricWeights);
				break;
			}
			network_refreshLearnrate(net, winner, looser);
		}

	}

	return delta_sum / data_rows;

}

void checkSampleAsCandidate(prototypeCandidate **candidates, PROTOFRMT *sample,
		int sampleLabel, PROTOFRMT looserDistance, int dimensionality) {
	if (!(*candidates)) {
		prototypeCandidate_init(candidates, sample, sampleLabel, looserDistance,
				dimensionality);
	} else {
		while (*candidates) {
			if ((*candidates)->label == sampleLabel) {
				if ((*candidates)->dist > looserDistance) {
					(*candidates)->dist = looserDistance;
					memcpy((*candidates)->data, sample,
							sizeof(PROTOFRMT) * dimensionality);
				}

				break;
			}

			candidates = &((*candidates)->next);
			if (!(*candidates)) {
				prototypeCandidate_init(candidates, sample, sampleLabel,
						looserDistance, dimensionality);
			}
		}
	}
}


void network_addSampleToList(struct Network *net, PROTOFRMT *sample) {
	PROTOFRMT * newEntry = (PROTOFRMT*)malloc(sizeof(PROTOFRMT)*(net->dimensionality));
	unsigned d;
	for(d=0; d < net->dimensionality; d++)
		newEntry[d]=sample[d];
	net->sampleBag.size++;

	if (net->sampleBag.size == 0) {
		net->sampleBag.entry = (PROTOFRMT **) malloc(sizeof(PROTOFRMT *));
		net->sampleBag.entry[net->sampleBag.size - 1] = newEntry;
	} else if (net->sampleBag.size > net->maxSampleBagSize){
		unsigned int delIdx = rand() % net->maxSampleBagSize;
		free(net->sampleBag.entry[delIdx]);
		net->sampleBag.entry[delIdx] = newEntry;
		net->sampleBag.size--;
	}else{
		net->sampleBag.entry = (PROTOFRMT **) realloc(net->sampleBag.entry,
				sizeof(PROTOFRMT *) * net->sampleBag.size);
		net->sampleBag.entry[net->sampleBag.size - 1] = newEntry;
	}



}


//XXVL laesst sich mit train zusammenfassen
/**
 * Iterative network training.
 * See @see{trainStep} for parameter description
 *
 * @param is_failure_increment 	Activate node insertion
 * @param g_max					Miscalssification threshold for node insertion
 * @param perftarget			If not NULL: Receives the classification performance of the given samples.
 */
PROTOFRMT network_trainIncremental(struct Network *net, PROTOFRMT *data,
		int *label, bool doInsertions, unsigned int g_max,
		unsigned int data_rows, double *classificationPerformance,
		bool adaptMetricWeights) {
	PROTOFRMT delta_sum = 0;
	unsigned int corr_count = 0;
	unsigned int i;
	for (i = 0; i < data_rows; i++) {
		PROTOFRMT* sampledata = data + net->dimensionality * i;
		int samplelabel = label[i];
		Prototype* winner = NULL;
		Prototype* looser = NULL;
		PROTOFRMT winner_dist;
		PROTOFRMT looser_dist;

		getWinnerLooserProto(net, sampledata, samplelabel, &winner, &looser,
				&winner_dist, &looser_dist);

		network_addSampleToList(net, sampledata);

		if (getWinnerLooserProto(net, sampledata, samplelabel, &winner, &looser,
				&winner_dist, &looser_dist)) {
			switch (net->mode) {
			case GLVQ:
			case GRLVQ:
			case LGRLVQ:
				delta_sum += network_adapt_GRLVQ(net, sampledata, winner,
						winner_dist, looser, looser_dist, adaptMetricWeights);
				break;
			case GMLVQ:
			case LGMLVQ:
				delta_sum += network_adapt_GMLVQ(net, sampledata, winner,
						winner_dist, looser, looser_dist, adaptMetricWeights);
				break;
			}
			network_refreshLearnrate(net, winner, looser);

			if (winner_dist <= looser_dist)
				corr_count++;
			else if (doInsertions) {
				net->currentErrorCount++;
				prototypeCandidate **candidates = &(net->candidateList);
				//XXVL Ist eigentlich falsch die Prototypen bewegen sich waehrend training,
				//kann sein dass am ende gar nicht der naechste falsche gewaehlt wird.
				checkSampleAsCandidate(candidates, sampledata, samplelabel,
						looser_dist, net->dimensionality);
				if (net->currentErrorCount >= g_max){
					net->currentErrorCount = 0;
					//add Node:
					prototypeCandidate *candidate = net->candidateList;
					while (candidate) {
						network_addPrototype(net, candidate->label,
								candidate->data);
						//printf("inserted Node for class %d\n", candidate->label);
						free(candidate->data);
						prototypeCandidate *tmp = candidate;
						candidate = candidate->next;
						free(tmp);

					};
					net->candidateList = NULL;
				}
			}
		}

	}

	if (classificationPerformance) {
		*classificationPerformance = ((double) corr_count)
				/ ((double) data_rows);
	}
	return delta_sum / data_rows;
}
;



void network_addPrototypeToList(struct Network *net, Prototype *proto) {
	if (net->prototypes.size == 0) {
		net->prototypes.size++;
		net->prototypes.entry = (Prototype **) malloc(sizeof(Prototype *));
	} else {
		net->prototypes.size++;
		net->prototypes.entry = (Prototype **) realloc(net->prototypes.entry,
				sizeof(Prototype *) * net->prototypes.size);
	}
	net->prototypes.entry[net->prototypes.size - 1] = proto;

}

void initPrototypeMetricWeights(struct Network *net, Prototype *proto) {
	unsigned int i, j;
	switch (net->mode) {
	case LGRLVQ: {
		PROTOFRMT* metricWeights = (PROTOFRMT*) malloc(
				sizeof(PROTOFRMT) * net->dimensionality);
		for (i = 0; i < net->dimensionality; i++) {
			metricWeights[i] = 1. / net->dimensionality;
		}
		proto->metricWeights = metricWeights;
		break;
	}
	case LGMLVQ: {

		PROTOFRMT targetval = sqrt(1.0 / net->dimensionality);
		PROTOFRMT* metricWeights = (PROTOFRMT*) malloc(
				sizeof(PROTOFRMT) * net->dimensionality * net->dimensionality);
		PROTOFRMT* omegaMetricWeights = (PROTOFRMT*) malloc(
				sizeof(PROTOFRMT) * net->dimensionality * net->dimensionality);
		for (i = 0; i < net->dimensionality; i++) {
			for (j = 0; j < net->dimensionality; j++) {
				if (i == j)
					omegaMetricWeights[i * net->dimensionality + j] = targetval;
				else
					omegaMetricWeights[i * net->dimensionality + j] = 0;
			}
		}
		network_refreshGMLVQMetricWeights(net, metricWeights,
				omegaMetricWeights);
		proto->metricWeights = metricWeights;
		proto->omegaMetricWeights = omegaMetricWeights;
		break;
	}
	default:
		break;

	}
}
/**
 * Add a new prototype to network.
 *
 * @param label					Prototype label
 * @param initdata				Prototype initialization data, is random if NULL is given
 */
void network_addPrototype(struct Network *net, int label,
PROTOFRMT *weights) {
	Prototype *newProto = prototype_create(net->dimensionality, label,
			net->learnrate_per_node, weights);

	initPrototypeMetricWeights(net, newProto);

	if (net->learnrate_per_node) {
		prototype_setLearnrate(newProto, net->learnrate_start);
		prototype_setLearnrateMetricWeights(newProto,
				net->learnrate_metricWeights_start);
		prototype_setSampleCounter(newProto, 0);
	}
	network_addPrototypeToList(net, newProto);

}
;

/**
 * Process a number of input samples.
 *
 * @param data					Input data array, length: dimensionality*data_rows
 * @param result				Classification result, length: data_rows
 * @param data_rows				Number of samples
 * @param indices				Optional, receives the nearest prototype number if not NULL
 */
void network_processData(struct Network *net, PROTOFRMT *data, int *result,
		unsigned int data_rows, unsigned int *indices) {
	unsigned int i;
	for (i = 0; i < data_rows; i++) {
		PROTOFRMT* sampledata = data + net->dimensionality * i;
		PROTOFRMT dummy;
		unsigned int index = network_findClosestProto(net, sampledata, &dummy);
		result[i] = prototype_getLabel(net->prototypes.entry[index]);
		if (indices)
			indices[i] = index;
	}
}

/**
 * Set the target for learnrate annealing.
 *
 * @param maxsamples			Defines the number of presented samples so that the trainingrate reaches start*0.001
 */
void network_setMaxsamples(struct Network *net, unsigned int maxsamples) {
	net->maxsamples = maxsamples;
}

/**
 * Get the dimensionality
 *
 * @return 						dimensionality of network nodes
 */
unsigned int network_getDimensionality(struct Network *net) {
	return net->dimensionality;
}

/**
 * Clear network structure.
 * Release memory.
 */
void network_clear(struct Network *net) {
	free(net->metricWeights);
	free(net->omegaMetricWeights);

	unsigned int i;
	for (i = 0; i < net->prototypes.size; i++) {
		prototype_del(net->prototypes.entry[i]);
	}

	for (i = 0; i < net->sampleBag.size; i++) {
		free(net->sampleBag.entry[i]);
	}


//XXVL check if this free is enough!
	if (net->prototypes.size > 0)
		free(net->prototypes.entry);
	net->prototypes.size = 0;
	if (net->sampleBag.size > 0)
		free(net->sampleBag.entry);
	net->sampleBag.size = 0;

	prototypeCandidate *current_proto = net->candidateList;

	while (current_proto) {
		free(current_proto->data);

		prototypeCandidate *old_proto = current_proto;
		current_proto = current_proto->next;
		free(old_proto);
	}

	net->candidateList = 0;
	net->currentErrorCount = 0;
}

/**
 * Get number of prototypes in network.
 *
 * @return 						Number of used Prototypes
 */
unsigned int network_getPrototypes(struct Network *net) {

	return net->prototypes.size;
}

/**
 * Get label of specific prototype.
 *
 * @param protonr				Prototype ID
 * @return 						Label
 */
int network_getPrototypeLabel(struct Network *net, unsigned int protonr) {
	if (net->prototypes.size <= protonr)
		return -1;

	return net->prototypes.entry[protonr]->label;
}

/**
 * Get weights of specific prototype.
 *
 * @param protonr				Prototype ID
 * @param target				Receives the prototype weights, length: dimensionality
 * @return 						True on success
 */
bool network_getPrototypeData(struct Network *net, unsigned int protonr,
PROTOFRMT* target) {
	if (net->prototypes.size <= protonr)
		return false;

	memcpy(target, net->prototypes.entry[protonr]->weights,
			net->dimensionality * sizeof(PROTOFRMT));
	return true;
}

/**
 * Get metric weights.
 *
 * @param protonr				Prototype ID (only different results if per node weights is activated)
 * @param target				Receives the metric weights, length: dimensionality or dimensionality*dimensionality(matrix mode) @see{getMetricWeightsLen()}
 * @return 						True on success
 */
bool network_getMetricWeights(struct Network *net, unsigned int protonr,
PROTOFRMT* target) {
	if (net->prototypes.size <= protonr)
		return false;
	unsigned int length = network_getMetricWeightsLen(net);
	PROTOFRMT* metricWeights = getMetricWeights(net,
			net->prototypes.entry[protonr]);
	memcpy(target, metricWeights, length * sizeof(PROTOFRMT));
	return true;
}

/**
 * Get matrix eingenvectors.
 * Use only with LGMLVQ or GMLVQ mode
 *
 * @param protonr				Prototype ID
 * @param target				Receives the eigenvectors, length: dimensionality*dimensionality
 * @return 						True on success
 */
bool network_getEigenvectors(struct Network *net, unsigned int protonr,
PROTOFRMT *target) {
	if (net->prototypes.size <= protonr)
		return false;
	unsigned int length = network_getMetricWeightsLen(net);
	if (length != net->dimensionality) {
		memcpy(target, net->prototypes.entry[protonr]->metricWeights,
				length * sizeof(PROTOFRMT));
	} else {
		memset(target, 0, length * length * sizeof(PROTOFRMT));
		int d;
		for (d = 0; d < net->dimensionality; d++) {
			target[d * net->dimensionality + d] =
					net->prototypes.entry[protonr]->metricWeights[d];
		}
	}
	return true;
}

/**
 * Returns the length of metric weights.
 *
 * @return 						dimensionality*dimensionality on matrix mode, else: dimensionality
 */
unsigned int network_getMetricWeightsLen(struct Network *net) {
	switch (net->mode) {
	case GRLVQ:
	case LGRLVQ:
		return net->dimensionality;
	case GMLVQ:
	case LGMLVQ:
		return net->dimensionality * net->dimensionality;
	default:
		return 0;
	}

}

