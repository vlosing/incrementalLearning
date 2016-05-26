/*
 * NetworkSerialization.c
 *
 *  Created on: Jun 11, 2014
 *      Author: vlosing
 */
#include "NetworkSerialization.h"



/**
 * Constructor that initializes a network by a given file.
 *
 * @param file					Input file name
 */
struct Network* network_createFile(char *file) {
	struct Network* net = (struct Network*) malloc(sizeof(struct Network));

	net->candidateList = 0;
	net->prototypes.size = 0;

	net->metricWeights = NULL;
	network_loadFile(net, file);

	return net;
}


/**
 * Serialize network into binary stream.
 * @see{Serializer}
 */
void network_serialize(struct Network *net, FILE *s) {

	fwrite(&(net->dimensionality), sizeof(net->dimensionality), 1, s);
	fwrite(&(net->max_threads), sizeof(net->max_threads), 1, s);
	fwrite(&(net->learnrate), sizeof(net->learnrate), 1, s);
	fwrite(&(net->learnrate_start), sizeof(net->learnrate_start), 1, s);
	fwrite(&(net->learnrate_metricWeights), sizeof(net->learnrate_metricWeights), 1, s);
	fwrite(&(net->learnrate_metricWeights_start), sizeof(net->learnrate_metricWeights_start), 1,
			s);

	fwrite(&(net->maxsamples), sizeof(net->maxsamples), 1, s);
	fwrite(&(net->datasamples), sizeof(net->datasamples), 1, s);
	fwrite(&(net->learnrate_per_node), sizeof(net->learnrate_per_node), 1, s);

	fwrite(&(net->mode), sizeof(net->mode), 1, s);
	fwrite(&(net->max_threads), sizeof(net->max_threads), 1, s);

	int num_protos = net->prototypes.size;
	fwrite(&(num_protos), sizeof(num_protos), 1, s);

	int lambda_len = 0;
	switch (net->mode) {
	case GLVQ:
	case GRLVQ:
		lambda_len = net->dimensionality;
		break;

	case GMLVQ:
		lambda_len = net->dimensionality * net->dimensionality * 2;
		break;

	default:
		break;
	}

	if (lambda_len > 0) {
		fwrite(net->metricWeights, sizeof(PROTOFRMT), lambda_len, s);
	}

	lambda_len = 0;
	if (net->mode == LGRLVQ) {
		lambda_len = net->dimensionality;
	} else if (net->mode == LGMLVQ) {
		lambda_len = net->dimensionality * net->dimensionality * 2;
	}

	int n;
	for (n = 0; n < num_protos; ++n) {
		prototype_serialize(net->prototypes.entry[n], s, lambda_len,
				net->learnrate_per_node);
	}

}

/**
 * Deserialize network from binary stream.
 * @see{Serializer}
 */
void network_deserialize(struct Network *net, FILE *s) {
	network_clear(net);

	fread(&(net->dimensionality), sizeof(net->dimensionality), 1, s);
	fread(&(net->max_threads), sizeof(net->max_threads), 1, s);
	fread(&(net->learnrate), sizeof(net->learnrate), 1, s);
	fread(&(net->learnrate_start), sizeof(net->learnrate_start), 1, s);
	fread(&(net->learnrate_metricWeights), sizeof(net->learnrate_metricWeights), 1, s);
	fread(&(net->learnrate_metricWeights_start), sizeof(net->learnrate_metricWeights_start), 1,
			s);

	fread(&(net->maxsamples), sizeof(net->maxsamples), 1, s);
	fread(&(net->datasamples), sizeof(net->datasamples), 1, s);
	fread(&(net->learnrate_per_node), sizeof(net->learnrate_per_node), 1, s);

	fread(&(net->mode), sizeof(net->mode), 1, s);
	fread(&(net->max_threads), sizeof(net->max_threads), 1, s);
	int num_protos;
	fread(&(num_protos), sizeof(num_protos), 1, s);

	int lambda_len = 0;
	switch (net->mode) {
	case GLVQ:
	case GRLVQ:
		lambda_len = net->dimensionality;
		break;

	case GMLVQ:
		lambda_len = net->dimensionality * net->dimensionality * 2;
		break;

	default:
		break;
	}

	if (lambda_len > 0) {
		net->metricWeights = (PROTOFRMT*) malloc(sizeof(PROTOFRMT) * lambda_len);
		fread(net->metricWeights, sizeof(PROTOFRMT), lambda_len, s);
	}

	lambda_len = 0;
	if (net->mode == LGRLVQ) {
		lambda_len = net->dimensionality;
	} else if (net->mode == LGMLVQ) {
		lambda_len = net->dimensionality * net->dimensionality * 2;
	}

	int n;
	for (n = 0; n < num_protos; ++n) {
		Prototype* np = prototype_createFile(s, lambda_len,
				net->learnrate_per_node, net->metricWeights);
		network_addPrototypeToList(net, np);
	}

}

/**
 * Load network state from file.
 *
 * @param file					Input filename
 */
void network_loadFile(struct Network *net, char *file) {
	FILE *fp;
	fp = fopen(file, "rb");
	network_deserialize(net, fp);
	fclose(fp);
}

/**
 * Save network state into file (binary).
 *
 * @param file					Output filename
 */
void network_saveFile(struct Network *net, char *file) {
	FILE *fp;
	fp = fopen(file, "wb");
	network_serialize(net, fp);
	fclose(fp);
}

;
