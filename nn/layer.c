/**
 * @file Network source file
 * @brief Network function definitions
 */

/* Includes ------------------------------------------ */
#include "layer.h"

/* --------------------------------------------------- */
void init_Layer(struct Layer *layer, int num_Neurons, int num_Inputs_Per_Neurons){
    layer->num_Neurons = num_Neurons; /* number of neurons of the layer are num_Neurons passed as argument */
    /* allocate memory for outputs array */
    layer->outputs = (double *)malloc(num_Neurons * sizeof(double));
    /* check malloc is healthy */
    if (layer->outputs == NULL){
        fprintf(stderr, "Could not allocate layer->outputs!");
        exit(-1);
    }

    /* allocate memory for weights 2D array */
    layer->weights = (double **)malloc(num_Neurons * sizeof(double *));
    /* check malloc is healthy */
    if (layer->weights == NULL){
        fprintf(stderr, "Could not allocate layer->weights!");
        exit(-1);
    }

    /* allocate memory for errors array */
    layer->errors = (double *)malloc(num_Neurons * sizeof(double));
    if (layer->errors == NULL){
        fprintf(stderr, "Could not allocate layer->errors!");
        exit(-1);
    }

    /* initialize the weights randomly and outputs to 0 */
    for(int i = 0; i < num_Neurons; i++){
        layer->outputs[i] = 0.0;

        /* Allocate memory for the weights of the current neuron */
        layer->weights[i] = (double *)malloc(num_Inputs_Per_Neurons * sizeof(double));
        /* check malloc is healthy */
        if (layer->weights[i] == NULL){
            fprintf(stderr, "Could not allocate layer->weights[%d]", i);
            exit(-1);
        }
        /* Second for-loop because weights is a 2D Array */
        /* Iterates through num_Inputs_Per_Neuron and for each connection, it initializes the weights */
        for (int j = 0; j < num_Inputs_Per_Neurons; ++j) {
            /* seed for the random function should be defined in the main program, so that it is only called once */
            layer->weights[i][j] = ((double)rand() / RAND_MAX); /* initialize randomly weights from 0 to 1*/
        }
    }
}
/* --------------------------------------------------- */

void free_Layer(struct Layer *layer){
    if (layer == NULL){
        fprintf(stderr,"Layer does not exist!\n"
                      "Exiting Program!\n");
        return;
    }
    //free all weights[i] and weights
    for (int i = 0; i < layer->num_Neurons; ++i) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    //free all outputs
    free(layer->outputs);
    // free errors
    free(layer->errors);


}
/* --------------------------------------------------- */
