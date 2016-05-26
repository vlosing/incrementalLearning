/*
 * PrototypeSearch.h
 *
 *  Created on: Jun 11, 2014
 *      Author: vlosing
 */

#ifndef PROTOTYPESEARCH_H_
#define PROTOTYPESEARCH_H_
#include "Prototype_c.h"
#include "Network_c.h"

/**
 * Find closest prototype for a given input.
 *
 * @param data					Input data
 * @param dist					Returns the distance to the nearest prototype, if not NULL
 *
 * @return 						ID of the nearest prototype
 */
unsigned int network_findClosestProto(struct Network *net, PROTOFRMT *data, PROTOFRMT *dist);



/**
 * Find closest prototype of same class for a given input.
 *
 * @param data					Input data
 * @param label					Label
 * @param dist					Returns the distance to the nearest prototype, if not NULL
 *
 * @return 						ID of the nearest prototype
 */
unsigned int network_findWinnerProto(struct Network *net, PROTOFRMT *data, int label, PROTOFRMT *dist);


/**
 * Find closest prototype another class for a given input.
 *
 * @param data					Input data
 * @param label					Label
 * @param dist					Returns the distance to the nearest prototype, if not NULL
 *
 * @return 						ID of the nearest prototype
 */
unsigned int network_findLooserProto(struct Network *net, PROTOFRMT *data, int label, PROTOFRMT *dist);



#endif /* PROTOTYPESEARCH_H_ */
