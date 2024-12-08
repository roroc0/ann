/**
 * @file Network header file
 * @brief Network function prototypes declaration
 */
#ifndef NN_LAYER_H
#define NN_LAYER_H

/* Includes ------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "net_parameters.h"
/* --------------------------------------------------- */

/**
 * @struct Layer
 * @brief Represents a neural network layer.
 *
 * The `Layer` struct is used to represent a single layer in a neural network.
 * It contains the following fields:
 * - `outputs`: An array storing the output values of each neuron in the layer.
 * - `weights`: A 2D array storing pointers to arrays of weights for each neuron.
 * - `num_Neurons`: The number of neurons in the layer.
 * - `errors`: An array storing error values used during backpropagation.
 */
struct Layer {
    double *outputs;    /**< Array to store the output values of each neuron in the layer */
    double **weights;   /**< 2D array to store the weights for each neuron's connections */
    int num_Neurons;    /**< Number of neurons in the layer */
    double *errors;     /**< Array to store error values for backpropagation */
};
/* --------------------------------------------------- */

/**
 * @brief Initialize a Layer in the neuronal network
 * @param layer pointer to the layer struct that is going to be initialized
 * @param num_Neurons number of neurons in the layer
 * @param num_Inputs_Per_Neurons number of connections per Neuron in to previous layer and
 * in case of input layer to the data
 *
 * This function initializes a layer by allocating memory for the output of each output values
 * of each neuron and the weights associated with each input connection to those neurons.
 * It initializes the outputs to 0.0 and the weights to random numbers between [0.0, 1.0]
 */
void init_Layer(struct Layer *layer, int num_Neurons, int num_Inputs_Per_Neurons);
/* --------------------------------------------------- */

/**
 * @brief Delete the layer struct previously initialized
 * @param layer pointer to the layer struct that is going to be deleted
 */
void free_Layer(struct Layer *layer);
/* --------------------------------------------------- */

#endif //NN_LAYER_H

