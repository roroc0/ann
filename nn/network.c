/**
 * @file Network source file
 * @brief Network function definitions
 */

/* Includes ------------------------------------------ */
#include "network.h"
/* --------------------------------------------------- */

void init_Network(struct Network *network, int input_Size, int *hidden_Sizes, int num_Hidden_Layers, int output_Size){
    network->hidden_Sizes = hidden_Sizes;
    network->num_Hidden_Layers = num_Hidden_Layers;

    /* initializes input layer */
    init_Layer(&network->input_Layer, input_Size, 0);

    /* initialize hidden layers */
    network->hidden_Layer = (struct Layer *)malloc(num_Hidden_Layers * sizeof(struct Layer));
    /* check malloc is healthy */
    if (network->hidden_Layer == NULL){
        fprintf(stderr, "Could not allocate network->hidden_Layer!");
        exit(-1);
    }

    for (int i = 0; i < network->num_Hidden_Layers; ++i) {
        if (i == 0){
            //equals the first element of the array
            init_Layer(&network->hidden_Layer[i], network->hidden_Sizes[i] ,input_Size);
        }else if (i > 0){
            // equals i - 1, for number of previous layer
            init_Layer(&network->hidden_Layer[i], network->hidden_Sizes[i], network->hidden_Sizes[i - 1] );
        }
    }


    /* initialize output layers */
    int num_Neurons_Last_Hidden_Layer = 0;
    if (num_Hidden_Layers > 0){
        num_Neurons_Last_Hidden_Layer = network->hidden_Sizes[network->num_Hidden_Layers - 1];
    }else {
        num_Neurons_Last_Hidden_Layer = input_Size;
    }

    init_Layer(&network->output_Layer, output_Size, num_Neurons_Last_Hidden_Layer);

}
/* --------------------------------------------------- */

void free_Network(struct Network *network){
    if (network == NULL){
        printf("network does not exist!\n"
               "exiting program!\n");
        return;
    }
    free_Layer(&network->input_Layer);
    // Free hidden layers
    for (int i = 0; i < network->num_Hidden_Layers; ++i) {
        free_Layer(&network->hidden_Layer[i]);
    }
    free(network->hidden_Layer);

    free_Layer(&network->output_Layer);
}
/* --------------------------------------------------- */