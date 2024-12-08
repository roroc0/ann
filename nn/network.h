/**
 * @file Network header file
 * @brief Network function prototypes declaration
 */

#ifndef NN_NETWORK_H
#define NN_NETWORK_H

/* Includes ------------------------------------------ */
#include "layer.h"
/* --------------------------------------------------- */

/**
 * @struct Network
 * @brief This struct represents the Artificial Neuronal Network
 * - `input_Layer` A struct from type Layer that represents the input layer
 * - `hidden_Layer` pointer to a struct from type Layer that represents the hidden layers
 * - `hidden_Sizes` array that contains the number of neurons of each hidden layer
 * - `num_Hidden_Layers` total number of hidden layers
 * - `output_Layer` A struct from type Layer that represents the output layer
 */
struct Network {
    struct Layer input_Layer;       /**< Input layer of the network */
    struct Layer *hidden_Layer;     /**< Pointer to array of hidden layers */
    int *hidden_Sizes;              /**< Array representing the number of neurons in each hidden layer */
    int num_Hidden_Layers;          /**< Total number of hidden layers in the network */
    struct Layer output_Layer;      /**< Output layer of the network */
};
/* --------------------------------------------------- */

 /**
  * @brief Initializes an artificial neuronal network
  * @param network pointer to the network struct that is going to be initialize
  * @param input_Size the number of neurons in the input layer
  * @param hidden_Sizes array containing the different sizes of each layer
  * @param num_Hidden_Layers the number of neurons in the hidden layer
  * @param output_Size the number of neurons in the output layer
  */
void init_Network(struct Network *network, int input_Size, int *hidden_Sizes, int num_Hidden_Layers, int output_Size);

/* --------------------------------------------------- */

/**
 * @brief Delete the network struct previously initialized
 * @param network pointer to the network struct that is going to be deleted
 */
void free_Network(struct Network *network);

/* --------------------------------------------------- */
#endif //NN_NETWORK_H
