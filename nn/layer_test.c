/**
 * @brief Test for functions in layer.c
 */
/* Includes ------------------------------------------ */
#include "layer.c"
#include <assert.h>
/* --------------------------------------------------- */
static void test_init_Layer();
/**
 * @brief Function to test the function to initialize the artificial neuronal network layer
 *
 * A Layer-struct (layer) is defined. The function to be tested is called passing the following arguments:
 *      - the address of where the layer-struct is allocated
 *      - the number of neurons in the layer --> 5
 *      - the number of inputs per neuron in the layer --> 3
 * The testing happens when asserting the different elements of the layer-strut to the expected value.
 */
static void test_init_Layer(){
    struct Layer layer;
    /* Initializes a Layer (layer) with 5 Neurons and each Neuron is connected to 3 Neuron from previous layer*/
    init_Layer(&layer, 5,3);
    assert(layer.num_Neurons == 5);
    for (int i = 0; i < layer.num_Neurons; ++i) {
        assert(layer.outputs[i] == 0.0);
        for (int j = 0; j < 3; ++j) {
            assert(layer.weights[i][j] >= 0.0 && layer.weights[i][j] <= 1.0);
        }
    }


    free_Layer(&layer);

}
/* --------------------------------------------------- */

/**
 * Main entry for the test.
 */
int main(int argc, char **argv)
{
    test_init_Layer();

    return 0;



}
/* --------------------------------------------------- */