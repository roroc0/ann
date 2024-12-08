/**
 * @brief Test for functions in training.c
 */
/* Includes ------------------------------------------ */
#include "layer.c"
#include "network.c"
#include "mathfunctions.c"
#include "training.c"
#include <assert.h>

#define EPOCH 100
#define L_RATE 0.001

/* PROTOTYPES */
void test_forward_propagation();
void test_back_propagation();
void test_training();

/* --------------------------------------------------- */
void test_forward_propagation()
{
    struct Network network;
    int input_Size = 3;        // 2 input neurons + 1 bias
    int hidden_Sizes[] = {3};  // 3 hidden neurons
    int num_Hidden_Layers = 1; // 1 hidden layer
    int output_Size = 1;       // 1 output neuron

    init_Network(&network, input_Size, hidden_Sizes, num_Hidden_Layers, output_Size);

    double inputs[4][3] = {
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 1.0}
    };
    double expected_outputs[] = {0.0, 1.0, 1.0, 0.0};

    // Setting weights manually
    network.hidden_Layer[0].weights[0][0] = 2.0;
    network.hidden_Layer[0].weights[0][1] = 2.0;
    network.hidden_Layer[0].weights[0][2] = -3.0;

    network.hidden_Layer[0].weights[1][0] = -2.0;
    network.hidden_Layer[0].weights[1][1] = -2.0;
    network.hidden_Layer[0].weights[1][2] = 3.0;

    network.output_Layer.weights[0][0] = 2.0;
    network.output_Layer.weights[0][1] = -3.0;
    network.output_Layer.weights[0][2] = 2.0;

    for (int i = 0; i < 4; ++i)
    {
        forward_propagate(&network, inputs[i]);

        double output = network.output_Layer.outputs[0];
        printf("Test Case %d: Input: %f, %f -> Predicted: %f, Expected: %f\n",
               i + 1, inputs[i][0], inputs[i][1], output, expected_outputs[i]);
    }

    free_Network(&network);
}

void test_back_propagation()
{
    struct Network network;
    int input_Size = 3;        // 2 input neurons + 1 bias
    int hidden_Sizes[] = {2};  // 2 hidden neurons
    int num_Hidden_Layers = 1; // 1 hidden layer
    int output_Size = 1;       // 1 output neuron

    init_Network(&network, input_Size, hidden_Sizes, num_Hidden_Layers, output_Size);

    double inputs[4][3] = {
        {0.0, 0.0, 1.0}, // Adding bias
        {0.0, 1.0, 1.0}, // Adding bias
        {1.0, 0.0, 1.0}, // Adding bias
        {1.0, 1.0, 1.0}  // Adding bias
    };
    double expected_outputs[] = {0.0, 1.0, 1.0, 0.0};

    // Setting weights manually
    network.hidden_Layer[0].weights[0][0] = 2.0;
    network.hidden_Layer[0].weights[0][1] = 2.0;
    network.hidden_Layer[0].weights[0][2] = -3.0;

    network.hidden_Layer[0].weights[1][0] = -2.0;
    network.hidden_Layer[0].weights[1][1] = -2.0;
    network.hidden_Layer[0].weights[1][2] = 3.0;

    network.output_Layer.weights[0][0] = 2.0;
    network.output_Layer.weights[0][1] = -3.0;
    network.output_Layer.weights[0][2] = 2.0;

    // Training XOR with backpropagation
    for (int epoch = 0; epoch < EPOCH; ++epoch)
    {
        for (int i = 0; i < 4; ++i)
        {
            forward_propagate(&network, inputs[i]);
            backward_propagate(&network, &expected_outputs[i], L_RATE);
        }
    }

    // Testing after training
    for (int i = 0; i < 4; ++i)
    {
        forward_propagate(&network, inputs[i]);

        double output = network.output_Layer.outputs[0];
        printf("Test Case %d: Input: %f, %f -> Predicted: %f, Expected: %f\n",
               i + 1, inputs[i][0], inputs[i][1], output, expected_outputs[i]);
    }

    free_Network(&network);
}

void test_training()
{
    struct Network network;
    int input_Size = 2;        // 2 input neurons
    int hidden_Sizes[] = {2};  // 2 hidden neurons
    int num_Hidden_Layers = 1; // 1 hidden layer
    int output_Size = 1;       // 1 output neuron

    init_Network(&network, input_Size, hidden_Sizes, num_Hidden_Layers, output_Size);

    double inputs[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    double expected_outputs[4][1] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Setting weights manually
    network.hidden_Layer[0].weights[0][0] = 2.0;
    network.hidden_Layer[0].weights[0][1] = 2.0;
    network.hidden_Layer[0].weights[0][2] = -3.0;

    network.hidden_Layer[0].weights[1][0] = -2.0;
    network.hidden_Layer[0].weights[1][1] = -2.0;
    network.hidden_Layer[0].weights[1][2] = 3.0;

    network.output_Layer.weights[0][0] = 2.0;
    network.output_Layer.weights[0][1] = -3.0;
    network.output_Layer.weights[0][2] = 2.0;


    // Test with validation set
    double validation_inputs[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    double validation_outputs[4][1] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };



    training(&network, EPOCHS,L_RATE,validation_inputs,validation_outputs,4);
    // Testing after training
    for (int i = 0; i < 4; ++i)
    {
        forward_propagate(&network, inputs[i]);

        double output = network.output_Layer.outputs[0];
        printf("Test Case %d: Input: %f, %f -> Predicted: %f, Expected: %f\n",
               i + 1, inputs[i][0], inputs[i][1], output, expected_outputs[i][0]);

        // Assert the output is close to the expected value
        //assert(fabs(output - expected_outputs[i][0]) < 0.1 && "Test failed for training");
    }

    free_Network(&network);
}

/**
 * Main entry for the test.
 */
int main(int argc, char **argv)
{
    //test_forward_propagation();
    //test_back_propagation();
    //test_training();
    return 0;
}
/* -------------------- EOF -------------------------- */