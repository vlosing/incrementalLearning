/*
 * NetworkSerialization.h
 *
 *  Created on: Jun 11, 2014
 *      Author: vlosing
 */

#ifndef NETWORKSERIALIZATION_H_
#define NETWORKSERIALIZATION_H_

#include <float.h>
#include "Network_c.h"

#include <math.h>
#include <string.h>

/**
 * Constructor that initializes a network by a given file.
 *
 * @param file					Input file name
 */
struct Network* network_createFile(char *file);

/**
 * Serialize network into binary stream.
 * @see{Serializer}
 */
void network_serialize(struct Network *net, FILE *s);

/**
 * Deserialize network from binary stream.
 * @see{Serializer}
 */
void network_deserialize(struct Network *net, FILE *s);

/**
 * Load network state from file.
 *
 * @param file					Input filename
 */
void network_loadFile(struct Network *net, char *file);

/**
 * Save network state into file (binary).
 *
 * @param file					Output filename
 */
void network_saveFile(struct Network *net, char *file);


#endif /* NETWORKSERIALIZATION_H_ */
