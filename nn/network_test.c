/**
 * @brief Test for functions in layer.c
 */
/* Includes ------------------------------------------ */
#include "layer.c"
#include "network.c"
#include <assert.h>
/* --------------------------------------------------- */
static void test_init_Network();
/* --------------------------------------------------- */

/**
 * @brief Function to test the function to initialize the artificial neuronal network
 *
 * A Network-struct (network) is defined. The function to be tested is called passing the following arguments:
 *      - the address of where the layer-struct is allocated
 *      - the number of neurons in the input layer --> 2
 *      - An array containing the number of neurons in each hidden layer --> 2 in the first hidden layer and 2 in the second
 *      - the number of hidden layers --> 2
 *      - the number of neurons in the output layer
 * The testing happens when asserting the different elements of the network-strut to the expected value.
 */
static void test_init_Network(){
    struct Network network;
    int input_Size = 2; /* Input Layer has 2 neurons*/
    int hidden_Sizes[] = {2, 2}; /* Number of neurons in each hidden layer*/
    int num_Hidden_Layers = 2; /* Number of hidden layers */
    int output_Size = 2 ; /* Number of neurons in the output layer */
    srand(time(0));
    init_Network(&network, input_Size, hidden_Sizes, num_Hidden_Layers, output_Size);
    assert(network.input_Layer.num_Neurons == input_Size); /* assert the number of neurons of input layer */
    for (int i = 0; i < num_Hidden_Layers; ++i) {
        assert(network.hidden_Layer[i].num_Neurons == hidden_Sizes[i]);
    }
    assert(network.output_Layer.num_Neurons == output_Size);

    /* Asserting outputs and weights initialisation */
    /* For input layer */
    for (int i = 0; i < network.input_Layer.num_Neurons; ++i) {
        assert(network.input_Layer.outputs[i] == 0.0);
        for (int j = 0; j < input_Size; ++j) {
            assert(network.input_Layer.weights[i][j] >= 0.0 && network.input_Layer.weights[i][j] <= 1.0);
        }
    }

    /* For hidden layers */
    for (int i = 0; i < num_Hidden_Layers; ++i) {
        for (int j = 0; j < network.hidden_Layer[i].num_Neurons; ++j) {
            assert(network.hidden_Layer[j].outputs[j] == 0.0);
            for (int k = 0; k < hidden_Sizes[i - 1]; ++k) {
                assert(network.hidden_Layer[j].weights[j][k] >= 0.0 && network.hidden_Layer[j].weights[j][k] <= 1.0);
            }
        }
    }

    /* For output layer */
    for (int i = 0; i < network.output_Layer.num_Neurons; ++i) {
        assert(network.output_Layer.outputs[i] == 0.0);
        for (int j = 0; j < output_Size; ++j) {
            assert(network.output_Layer.weights[i][j] >= 0.0 && network.output_Layer.weights[i][j] <= 1.0);
        }
    }

    free_Network(&network);
}

/* --------------------------------------------------- */

/**
 * Main entry for the test.
 */
int main(int argc, char **argv)
{

    test_init_Network();
    return 0;
}
/* -------------------- EOF -------------------------- */