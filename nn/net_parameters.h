/**
 * @file net_parameters.h
 * @brief Preprocessor Defines for easy tweaking
 */

#ifndef NN_NET_PARAMETERS_H
#define NN_NET_PARAMETERS_H

/* Defines- ------------------------------------------ */ 
// net structure
#define INPUT_LAYER_SIZE 784
#define OUTPUT_LAYER_SIZE 10
#define NUMBER_HIDDEN_LAYERS 1
#define HIDDEN_LAYER_SIZE {10};
#define MAX_HIDDEN_LAYERS 10  // Define a reasonable maximum of possible number of hidden layers


// for training
#define LOG 1 // output logging info e.g. 0=no logs, 1=accuracy each epoch, 2= accuracy + weights before and after training
#define EPOCHS 4
#define L_RATE 0.001
#define BATCH_SIZE 32 // Size of mini-batches
#define EARLY_STOPPING_PATIENCE 5// Number of epochs to wait for improvement

#endif //NN_NET_PARAMETERS_H