/*
 * PrototypeSearch.c
 *
 *  Created on: Jun 11, 2014
 *      Author: vlosing
 */
#include "PrototypeSearch.h"
#include <pthread.h>
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

struct ThreadData_struct
{
	PROTOFRMT* data;
	unsigned int start;
	unsigned int len;
	PROTOFRMT* dist;
	unsigned int *index;
	int label;

	struct Network* owner;
};
typedef struct ThreadData_struct ThreadData;


void ThreadData_init(ThreadData *target, struct Network* owner, PROTOFRMT* data, unsigned int start, unsigned int len, PROTOFRMT* dist, unsigned int* index, unsigned int label)
{
	target->data=data;
	target->start=start;
	target->len=len;
	target->dist=dist;
	target->index=index;
	target->owner=owner;
	target->label=label;
}

PROTOFRMT getDistToPrototype(struct Network *net, Prototype* proto, PROTOFRMT *data)
{
	switch (net->mode) {
	case GLVQ:
		return prototype_dist(proto, data);
	case GRLVQ:
		return prototype_distRel(proto, data, net->metricWeights);
	case LGRLVQ:
		return prototype_distRel(proto, data, proto->metricWeights);
	case GMLVQ:
		return prototype_distmat(proto, data, net->metricWeights);
	case LGMLVQ:
		return prototype_distmat(proto, data, proto->metricWeights);
	default:
		printf("Error: Network::findPrototypePart: unknown mode!");
		return PROTOMAX;
	}
}

int isSameLabel(int label1, int label2){
	return (label1==label2);
}

int isDifferentLabel(int label1, int label2){
	return (label1!=label2);
}

int DummyFct(int label1, int label2){
	return true;
}

unsigned int network_findProtoPart(struct Network *net,
PROTOFRMT *data, int label, unsigned int start, unsigned int len,
PROTOFRMT *dist, int (*labelCompareFunction)(int, int)) {
	PROTOFRMT distance_res = PROTOMAX;
	unsigned int index_res = 0;

	unsigned int i;
	for (i = start; i < start + len; i++) {
		int curr_label = prototype_getLabel(net->prototypes.entry[i]);
		if (labelCompareFunction(curr_label, label)) {
			PROTOFRMT curr_dist = getDistToPrototype(net, net->prototypes.entry[i], data);
			if (curr_dist < distance_res) {
				distance_res = curr_dist;
				index_res = i;
			}
		}
	}

	*dist = distance_res;
	return index_res;
}




unsigned int network_findProto(struct Network *net, PROTOFRMT *data,
		int label, PROTOFRMT *dist, int (*labelCompareFunction)(int, int),
		void* (*ThreadFct)(void *data)){
	unsigned int indices[net->max_threads];
	PROTOFRMT distances[net->max_threads];

	unsigned int protocount = net->prototypes.size;
	unsigned int threads = MIN(net->max_threads, protocount);
	unsigned int length = ceil(net->prototypes.size / (float) threads);

	if (threads == 0)
		return -1;

	pthread_t threadids[threads - 1];
	ThreadData tdata[threads - 1];

	unsigned int i;
	for (i = 0; i < threads; i++) {
		unsigned int part_start = i * length;
		unsigned int part_len = MIN(length, protocount - part_start);

		if (i == 0) {
			indices[i] = network_findProtoPart(net, data, label,
					part_start, part_len, &(distances[i]), labelCompareFunction);
		} else {
			ThreadData_init(&(tdata[i - 1]), net, data, part_start, part_len,
					&(distances[i]), &(indices[i]), label);
			pthread_create(&(threadids[i - 1]), NULL,
					ThreadFct, &(tdata[i - 1]));
		}
	}

	unsigned int index_res = indices[0];
	PROTOFRMT distance_res = distances[0];
	for (i = 1; i < threads; i++) {
		pthread_join(threadids[i - 1], NULL);
		if (distances[i] < distance_res) {
			distance_res = distances[i];
			index_res = indices[i];
		}
	}

	*dist = distance_res;
	return index_res;
}

/**
 * Internal method related to @see{findPrototype}.
 * Process prototype search subtask (for threads)
 */
void *network_findClosestPrototPartThread(void *data) {
	ThreadData *tdata = (ThreadData*) data;
	*(tdata->index) = network_findProtoPart(tdata->owner, tdata->data, 0,
			tdata->start, tdata->len, tdata->dist, DummyFct);
	return 0;
}
/**
 * Find closest prototype for a given input.
 *
 * @param data					Input data
 * @param dist					Returns the distance to the nearest prototype, if not NULL
 *
 * @return 						ID of the nearest prototype
 */
unsigned int network_findClosestProto(struct Network *net, PROTOFRMT *data,
PROTOFRMT *dist) {
	return network_findProto(net, data, -1, dist, DummyFct, network_findClosestPrototPartThread);
}

/**
 * Internal method related to @see{findCorrectPrototype}.
 * Process prototype search subtask (for threads)
 */
void *network_findWinnerPrototPartThread(void *data) {
	ThreadData *tdata = (ThreadData*) data;
	*(tdata->index) = network_findProtoPart(tdata->owner, tdata->data,
			tdata->label, tdata->start, tdata->len, tdata->dist, isSameLabel);
	return 0;
}


/**
 * Find closest prototype of another class for a given input.
 *
 * @param data					Input data
 * @param label					Label
 * @param dist					Returns the distance to the nearest prototype, if not NULL
 *
 * @return 						ID of the nearest prototype
 */
unsigned int network_findWinnerProto(struct Network *net, PROTOFRMT *data,
		int label, PROTOFRMT *dist) {
	return network_findProto(net, data, label, dist, isSameLabel, network_findWinnerPrototPartThread);
}


/**
 * Internal method related to @see{findCorrectPrototype}.
 * Process prototype search subtask (for threads)
 */
void *network_findLooserProtoPartThread(void *data) {
	ThreadData *tdata = (ThreadData*) data;
	*(tdata->index) = network_findProtoPart(tdata->owner, tdata->data,
			tdata->label, tdata->start, tdata->len, tdata->dist, isDifferentLabel);
	return 0;
}


/**
 * Find closest prototype of the same class for a given input.
 *
 * @param data					Input data
 * @param label					label
 * @param dist					Returns the distance to the nearest prototype, if not NULL
 *
 * @return 						ID of the nearest prototype
 */
unsigned int network_findLooserProto(struct Network *net, PROTOFRMT *data,
		int label, PROTOFRMT *dist) {
	return network_findProto(net, data, label, dist, isDifferentLabel, network_findLooserProtoPartThread);

}
