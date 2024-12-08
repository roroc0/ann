/**
 * @file main application program
 */

/* Includes ------------------------------------------ */
#include "training.h"
#include "ctype.h"

/* Defines- ------------------------------------------ */
#define TRAIN_CSV "./data/mnist_train.csv"
#define TEST_CSV "./data/mnist_test.csv"

/* Prototypes----------------------------------------- */
int parse_config_file(const char *config_file, int *input_Size, int *hidden_Sizes, int *num_Hidden_Layers, int *output_Size);
void print_network_structure(struct Network *network);

/* Main Entry ---------------------------------------- */
int main(int argc, char **argv)
{
    // Print which mode is chosen based on the compilation flag
#if defined(SEQ)
    fprintf(stdout, "Sequential Processing\n");
    fprintf(stdout, "==============================\n");
#elif defined(PARALLEL)
    fprintf(stdout, "Parallel Processing - OMP\n");
    fprintf(stdout, "==============================\n");
#elif defined(SIMD)
    fprintf(stdout, "SIMD Processing\n");
    fprintf(stdout, "==============================\n");
#endif

    struct Network network;
    int input_Size = INPUT_LAYER_SIZE;            // Default input size
    int hidden_Sizes[MAX_HIDDEN_LAYERS];          // Allocate space for hidden layers
    int num_Hidden_Layers = NUMBER_HIDDEN_LAYERS; // Default number of hidden layers
    int output_Size = OUTPUT_LAYER_SIZE;          // Default output size

    // Set default hidden layer sizes
    int default_hidden_Sizes[] = HIDDEN_LAYER_SIZE;
    memcpy(hidden_Sizes, default_hidden_Sizes, num_Hidden_Layers * sizeof(int));

    if (argc == 2) // Case 1: Config file is passed
    {
        const char *config_file = argv[1];
        if (parse_config_file(config_file, &input_Size, hidden_Sizes, &num_Hidden_Layers, &output_Size))
        {
            fprintf(stdout, "Using network structure from config file: %s\n", config_file);
        }
        else
        {
            fprintf(stderr, "Failed to parse config file: %s. Using default network structure.\n", config_file);
        }
    }
    else if (argc >= 4) // Case 2: Command-line arguments are passed
    {
        input_Size = atoi(argv[1]);
        num_Hidden_Layers = atoi(argv[2]);

        if (num_Hidden_Layers > MAX_HIDDEN_LAYERS)
        {
            fprintf(stderr, "Error: Number of hidden layers exceeds the maximum allowed (%d).\n", MAX_HIDDEN_LAYERS);
            exit(EXIT_FAILURE);
        }

        // Parse hidden layer sizes, assumed to be comma-separated
        char *token = strtok(argv[3], ",");
        for (int i = 0; i < num_Hidden_Layers && token != NULL; i++)
        {
            hidden_Sizes[i] = atoi(token);
            token = strtok(NULL, ",");
        }

        output_Size = atoi(argv[4]);
        fprintf(stdout, "Using network structure from command-line arguments\n");
    }
    else if (argc == 3) // Case: Not enough args
    {
        fprintf(stdout, "Not enough arguments passed, using default configuration\n");
    }
    else // Case 3: No arguments provided, use default configuration
    {
        fprintf(stdout, "No config file or arguments provided. Using default network structure.\n");
    }

    srand(0); /* for weights random initialisation */

    init_Network(&network, input_Size, hidden_Sizes, num_Hidden_Layers, output_Size);
    print_network_structure(&network);
    fprintf(stdout, "Epochs = %d\nLearning Rate = %f\nBatch Size = %d\n", EPOCHS, L_RATE, BATCH_SIZE);
    fprintf(stdout, "Stopping Training after %d epochs without improvement\n", EARLY_STOPPING_PATIENCE);
    fprintf(stdout, "==============================\n");


    // Prepare dataset
    struct Data train_data = parse_MNIST_CSV_and_normalize(TRAIN_CSV, MAX_ROWS_TRAIN, 10);
    struct Data test_data = parse_MNIST_CSV_and_normalize(TEST_CSV, MAX_ROWS_TEST, 10);

    if (LOG >= 2)
    {
        fprintf(stdout, "Printing weights before training\n");
        print_weights(&network);
    }

    fprintf(stdout, "==============================\n");
    fprintf(stdout, "Starting to train\n");
    training(&network, EPOCHS, L_RATE, train_data.values, train_data.labels, MAX_ROWS_TRAIN);
    fprintf(stdout, "==============================\n");

    if (LOG >= 2)
    {
        fprintf(stdout, "Printing weights after training\n");
        print_weights(&network);
    }

    fprintf(stdout, "==============================\n");
    calculate_accuracy(&network, test_data.values, test_data.labels);
    fprintf(stdout, "==============================\n");

    // Free allocated memory
    free_Data(&train_data, MAX_ROWS_TRAIN);
    free_Data(&test_data, MAX_ROWS_TEST);
    free_Network(&network);

    return 0;
}

/* --------------------------------------------------- */
int parse_config_file(const char *config_file, int *input_Size, int *hidden_Sizes, int *num_Hidden_Layers, int *output_Size)
{
    FILE *file = fopen(config_file, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error: Could not open config file %s\n", config_file);
        return 0;
    }

    char line[256];
    while (fgets(line, sizeof(line), file))
    {
        // Remove any trailing newline character
        line[strcspn(line, "\n")] = 0;

        // Split the line into key and value
        char *key = strtok(line, "=");
        char *value = strtok(NULL, "=");

        if (key && value)
        {
            // Trim any leading/trailing whitespace from key and value
            while (isspace((unsigned char)*key))
                key++;
            while (isspace((unsigned char)*value))
                value++;
            char *end_key = key + strlen(key) - 1;
            char *end_value = value + strlen(value) - 1;
            while (end_key > key && isspace((unsigned char)*end_key))
                end_key--;
            while (end_value > value && isspace((unsigned char)*end_value))
                end_value--;
            *(end_key + 1) = '\0';
            *(end_value + 1) = '\0';

            // Process the key-value pairs
            if (strcmp(key, "input_size") == 0)
            {
                *input_Size = atoi(value);
            }
            else if (strcmp(key, "hidden_sizes") == 0)
            {
                char *token = strtok(value, ",");
                int i = 0;
                while (token != NULL)
                {
                    if (i >= MAX_HIDDEN_LAYERS)
                    {
                        fprintf(stderr, "Error: Config file defines more hidden layers than the maximum allowed (%d).\n", MAX_HIDDEN_LAYERS);
                        exit(EXIT_FAILURE);
                    }
                    hidden_Sizes[i++] = atoi(token);
                    token = strtok(NULL, ",");
                }
                *num_Hidden_Layers = i; // Update the number of hidden layers based on the count
            }

            else if (strcmp(key, "num_hidden_layers") == 0)
            {
                *num_Hidden_Layers = atoi(value);
            }
            else if (strcmp(key, "output_size") == 0)
            {
                *output_Size = atoi(value);
            }
            else
            {
                fprintf(stderr, "Warning: Unknown key in config file: %s\n", key);
            }
        }
    }

    fclose(file);
    return 1; // Return 1 to indicate success
}
/* --------------------------------------------------- */
void print_network_structure(struct Network *network)
{

    fprintf(stdout, "Input Layer: %d Neurons \nOutput Layer %d Neurons \n", network->input_Layer.num_Neurons, network->output_Layer.num_Neurons);
    fprintf(stdout, "%d Hidden Layers\n", network->num_Hidden_Layers);
    for (int i = 0; i < network->num_Hidden_Layers; i++)
    {
        fprintf(stdout, "Hidden Layer Nr %d has %d Neurons\n", i + 1, network->hidden_Layer[i].num_Neurons);
    }
}
/* -------------------- EOF -------------------------- */