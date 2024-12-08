/**
 * @file Training header file
 * @brief Training functions prototypes will be declared
 */

#ifndef NN_TRAINING_H
#define NN_TRAINING_H
/* Includes ------------------------------------------ */
#include "mathfunctions.h"
#include "network.h"
#include "mnist.h"
/* --------------------------------------------------- */


/**
 * @brief Predict output data with given input data
 *
 * This function defines the forward propagation or inference,
 * which is the 1st step of training an artificial neuronal network.
 *
 * @param network Pointer to the network struct
 * @param inputs The data set that is going to be input in the network
 */
void forward_propagate(struct Network *network, double *inputs);
/* --------------------------------------------------- */

/**
 * @brief Perform backpropagation to update network weights
 *
 * This function performs backpropagation for training the neural network
 * by calculating the error between the predicted and expected output, and
 * updating the weights of the network based on this error.
 *
 * @param network Pointer to the network struct
 * @param expected_output The expected output data set
 * @param learning_rate The learning rate used for weight updates
 */
void backward_propagate(struct Network *network, double *expected_output, double learning_rate);
/* --------------------------------------------------- */

/**
 * @brief Train the neural network
 *
 * This function trains the neural network over a specified number of epochs,
 * using the given input and output data sets, and a specified learning rate.
 *
 * @param network Pointer to the network struct
 * @param epochs The number of training epochs
 * @param learning_rate The learning rate used for training
 * @param input_data The input data set for training
 * @param output_data The output data set for training
 * @param num_samples The number of samples in the data sets
 */
void training(struct Network *network, int epochs, double learning_rate, double **input_data, double **output_data, int num_samples);
/* --------------------------------------------------- */

/**
 * @brief Get the predicted label from the network
 *
 * This function gets the predicted label by finding the maximum output value
 * from the network's output layer.
 *
 * @param network Pointer to the network struct
 * @return The index of the neuron with the highest output value
 */
int get_predicted_label(struct Network *network);
/* --------------------------------------------------- */

/**
 * @brief Calculate the accuracy of the network
 *
 * This function calculates the accuracy of the network by comparing the
 * network's predicted outputs with the true labels.
 *
 * @param network Pointer to the network struct
 * @param values The input data set for testing
 * @param labels The true labels for the input data set
 */
void calculate_accuracy(struct Network *network, double **values, double **labels);
/* --------------------------------------------------- */

/**
 * @brief Calculate the errors in the network during backpropagation
 *
 * This function calculates the errors for each neuron in the network
 * during backpropagation. The errors are computed by comparing the
 * predicted output to the expected output and are propagated backward
 * through the network.
 *
 * @param network Pointer to the network struct
 * @param expected_output The expected output data set
 */
void calculate_errors(struct Network *network, double *expected_output);
/* --------------------------------------------------- */

/**
 * @brief Update the network's weights during training
 *
 * This function updates the weights of the neural network during the
 * training process. The weights are adjusted based on the calculated
 * errors and the learning rate.
 *
 * @param network Pointer to the network struct
 * @param learning_rate The learning rate used for updating the weights
 */
void update_weights(struct Network *network, double learning_rate);
/* --------------------------------------------------- */

/**
 * @brief Print the weights of a specific layer
 *
 * This function prints the weights of the specified layer of the neural network.
 *
 * @param layer Pointer to the layer struct
 */
void print_layer_weights(struct Layer *layer);
/* --------------------------------------------------- */

/**
 * @brief Print the weights of the entire network
 *
 * This function prints the weights of all the layers in the neural network.
 *
 * @param network Pointer to the network struct
 */
void print_weights(struct Network *network);
/* --------------------------------------------------- */

#endif //NN_TRAINING_H
/* -------------------- EOF -------------------------- */
