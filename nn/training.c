/**
 * @file Training source file
 * @brief Training function definitions
 */

/* Includes ------------------------------------------ */
#include "training.h"

/* --------------------------------------------------- */
void forward_propagate(struct Network *network, double *inputs)
{
    /* Set inputs and outputs of input layer */
    for (int i = 0; i < network->input_Layer.num_Neurons; ++i)
    {
        network->input_Layer.outputs[i] = inputs[i];
    }
    /* Initialize errors in hidden layers to 0 */
    for (int i = 0; i < network->num_Hidden_Layers; ++i)
    {
        for (int j = 0; j < network->hidden_Layer[i].num_Neurons; ++j)
        {
            network->hidden_Layer[i].errors[j] = 0.0;
        }
    }
    /* Initialize errors in output layer to 0 */
    for (int i = 0; i < network->output_Layer.num_Neurons; ++i)
    {
        network->output_Layer.errors[i] = 0.0;
    }

    /* Forward propagate through hidden layers */
    for (int i = 0; i < network->num_Hidden_Layers; ++i)
    {
        for (int j = 0; j < network->hidden_Sizes[i]; ++j)
        {
            double sum = 0.0;
            if (i > 0)
            {
                sum = dotp(network->hidden_Layer[i - 1].outputs, network->hidden_Layer[i].weights[j], network->hidden_Sizes[i - 1]);
            }
            else
            {
                sum = dotp(network->input_Layer.outputs, network->hidden_Layer[i].weights[j], network->input_Layer.num_Neurons);
            }
            network->hidden_Layer[i].outputs[j] = sigmoid(sum);
        }
    }

    /* Forward propagate through output layer */
    for (int i = 0; i < network->output_Layer.num_Neurons; ++i)
    {
        double sum;
        sum = dotp(network->hidden_Layer[network->num_Hidden_Layers - 1].outputs, network->output_Layer.weights[i], network->hidden_Layer[network->num_Hidden_Layers - 1].num_Neurons);
        network->output_Layer.outputs[i] = sigmoid(sum);
    }
}


/* --------------------------------------------------- */
void calculate_errors(struct Network *network, double *expected_output)
{
    // Calculate output layer errors
    for (int i = 0; i < network->output_Layer.num_Neurons; ++i)
    {
        double output = network->output_Layer.outputs[i];
        double error = expected_output[i] - output;
        network->output_Layer.errors[i] = error * d_sigmoid(output);
    }

    // Calculate hidden layer errors
    for (int i = network->num_Hidden_Layers - 1; i >= 0; --i)
    {
        for (int j = 0; j < network->hidden_Layer[i].num_Neurons; ++j)
        {
            double error = 0.0;
            if (i == network->num_Hidden_Layers - 1)
            {
                for (int k = 0; k < network->output_Layer.num_Neurons; ++k)
                {
                    error += network->output_Layer.errors[k] * network->output_Layer.weights[k][j];
                }
            }
            else
            {
                for (int k = 0; k < network->hidden_Layer[i + 1].num_Neurons; ++k)
                {
                    error += network->hidden_Layer[i + 1].errors[k] * network->hidden_Layer[i + 1].weights[k][j];
                }
            }
            network->hidden_Layer[i].errors[j] = error * d_sigmoid(network->hidden_Layer[i].outputs[j]);
        }
    }
}

/* --------------------------------------------------- */
void update_weights(struct Network *network, double learning_rate)
{
    // Update output layer weights
    for (int i = 0; i < network->output_Layer.num_Neurons; ++i)
    {
        for (int j = 0; j < network->hidden_Layer[network->num_Hidden_Layers - 1].num_Neurons; ++j)
        {
            network->output_Layer.weights[i][j] += learning_rate * network->output_Layer.errors[i] * network->hidden_Layer[network->num_Hidden_Layers - 1].outputs[j];
        }
    }

    // Update hidden layer weights
    for (int i = network->num_Hidden_Layers - 1; i >= 0; --i)
    {
        for (int j = 0; j < network->hidden_Layer[i].num_Neurons; ++j)
        {
            for (int k = 0; k < ((i > 0) ? network->hidden_Layer[i - 1].num_Neurons : network->input_Layer.num_Neurons); ++k)
            {
                network->hidden_Layer[i].weights[j][k] += learning_rate * network->hidden_Layer[i].errors[j] * ((i > 0) ? network->hidden_Layer[i - 1].outputs[k] : network->input_Layer.outputs[k]);
            }
        }
    }
}

/* --------------------------------------------------- */
void backward_propagate(struct Network *network, double *expected_output, double learning_rate)
{
    // Calculate errors
    calculate_errors(network, expected_output);

    // Update weights
    update_weights(network, learning_rate);
}

/* --------------------------------------------------- */
void training(struct Network *network, int epochs, double learning_rate, double **input_data, double **output_data, int num_samples)
{
    int max_num_correct = 0;
    int patience = 0;
    // Iterate through epochs
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        // Log epoch information
        // printf("Epoch %d\n", epoch);
        int num_correct = 0;
        // Iterate through all samples, processing in mini-batches
        for (int batch_start = 0; batch_start < num_samples; batch_start += BATCH_SIZE)
        {
            int batch_end = batch_start + BATCH_SIZE < num_samples ? batch_start + BATCH_SIZE : num_samples;

            // Process each mini-batch
            for (int i = batch_start; i < batch_end; i++)
            {
                forward_propagate(network, input_data[i]);
                backward_propagate(network, output_data[i], learning_rate);

                // Calculate accuracy on-the-fly for each epoch (with training data) in order to stop training if no improvement
                int predictedlabel = get_predicted_label(network);

                int true_label = 0;
                for (int j = 0; j < network->output_Layer.num_Neurons; j++)
                {
                    if (output_data[i][j] == 1.0)
                    {
                        true_label = j;
                        break;
                    }
                }

                if (predictedlabel == true_label)
                {
                    num_correct++;
                }
            }
        }
        // Calculate and log accuracy after each epoch
        double accuracy = ((double)num_correct / num_samples) * 100.0;
        if (num_correct > max_num_correct)
        {
            max_num_correct = num_correct;
            patience = 0; // Reset patience on improvement
        }
        else
        {
            patience++;
            if (LOG >= 1)
            {
                fprintf(stdout, "Max Num Correct = %d \nPatience = %d\n", max_num_correct, patience);
            }
        }

        if (LOG >= 1)
        {
            printf("Accuracy after epoch %d: %.2f%% (%d/%d)\n", epoch, accuracy, num_correct, num_samples);
        }
        if (patience > EARLY_STOPPING_PATIENCE)
        {   
            fprintf(stdout, "==============================\n");
            fprintf(stdout, "No progress after %d consecutive Epochs - Stopping training at epoch %d\n", EARLY_STOPPING_PATIENCE, epoch);
            break;
        }
    }
}

/* --------------------------------------------------- */
void calculate_accuracy(struct Network *network, double **values, double **labels)
{
    int num_correct = 0;

    for (int i = 0; i < MAX_ROWS_TEST; i++)
    {
        forward_propagate(network, values[i]);

        int predicted_label = get_predicted_label(network);

        int true_label = 0;
        for (int j = 0; j < network->output_Layer.num_Neurons; j++)
        {
            if (labels[i][j] == 1.0)
            {
                true_label = j;
                break;
            }
        }

        // Log the prediction details
        if (LOG >= 2)
        {
            printf("Sample %d:\n", i);
            printf("Predicted Label: %d, True Label: %d\n", predicted_label, true_label);
            printf("Network Outputs: ");
            for (int j = 0; j < network->output_Layer.num_Neurons; j++)
            {
                printf("%f ", network->output_Layer.outputs[j]);
            }
            printf("\n");
        }

        if (predicted_label == true_label)
        {
            num_correct++;
        }
    }
    fprintf(stdout, "Total number of correct predictions with unseen data = %d/%d\n", num_correct, MAX_ROWS_TEST);
    double accuracy = ((double)num_correct / MAX_ROWS_TEST) * 100.0;
    printf("Final Accuracy [with unseen data]: %.2f%%\n", accuracy);
}

/* --------------------------------------------------- */
int get_predicted_label(struct Network *network)
{
    int max_index = 0;
    double max_output = network->output_Layer.outputs[0];

    for (int i = 1; i < network->output_Layer.num_Neurons; ++i)
    {
        if (network->output_Layer.outputs[i] > max_output)
        {
            max_output = network->output_Layer.outputs[i];
            max_index = i;
        }
    }

    return max_index;
}

/* --------------------------------------------------- */
void print_layer_weights(struct Layer *layer)
{
    for (int i = 0; i < layer->num_Neurons; ++i)
    {
        fprintf(stdout, "Neuron %d Weights: ", i + 1);
        for (int j = 0; j < layer->num_Neurons; ++j)
        {
            fprintf(stdout, "%f ", layer->weights[i][j]);
        }
        fprintf(stdout, "\n");
    }
}

/* --------------------------------------------------- */
void print_weights(struct Network *network)
{
    fprintf(stdout, "Input Layer Weights:\n");
    print_layer_weights(&network->input_Layer);

    for (int i = 0; i < network->num_Hidden_Layers; ++i)
    {
        fprintf(stdout, "Hidden Layer %d Weights:\n", i + 1);
        print_layer_weights(&network->hidden_Layer[i]);
    }

    fprintf(stdout, "Output Layer Weights:\n");
    print_layer_weights(&network->output_Layer);

    fprintf(stdout, "\n");
}
/* -------------------- EOF -------------------------- */