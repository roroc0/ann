/**
 * @file Mathfunctions source file
 * @brief Different Math function definitions
 */

/* Includes ------------------------------------------ */
#include "mnist.h"

/* --------------------------------------------------- */
struct Data parse_MNIST_CSV_and_normalize(const char *filename, int num_rows, int num_classes)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Allocate memory for struct data
    struct Data dataset;
    dataset.values = malloc(num_rows * sizeof(double *));
    dataset.labels = malloc(num_rows * sizeof(double *));

    char line[4096]; // Assuming lines won't exceed 4096 characters
    int row_count = 0;

    while (fgets(line, sizeof(line), file) && row_count < num_rows)
    {
        // Allocate memory for values and labels arrays in each row
        dataset.values[row_count] = malloc((MAX_COLUMNS - 1) * sizeof(double));
        dataset.labels[row_count] = malloc(num_classes * sizeof(double));

        // Split the line by comma
        char *token = strtok(line, ",");
        int col_count = 0;

        // Extract label from the first element and perform one-hot encoding
        int label = atoi(token);
        for (int i = 0; i < num_classes; i++)
        {
            dataset.labels[row_count][i] = (i == label) ? 1.0 : 0.0;
        }

        // Extract values from the remaining elements
        token = strtok(NULL, ",");
        while (token != NULL && col_count < (MAX_COLUMNS - 1))
        {
            dataset.values[row_count][col_count] = atof(token);
            token = strtok(NULL, ",");
            col_count++;
        }

        row_count++;
    }

    fclose(file);

    // Normalize the data to [0, 1] range
    normalize_data(dataset.values, num_rows, MAX_COLUMNS - 1, 255.0, 0.0);

    return dataset;
}

/* --------------------------------------------------- */
void normalize_data(double **x, int rows, int cols, double max, double min) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            //double original_value = x[i][j];
            x[i][j] = (x[i][j] - min) / (max - min);
            //printf("Original: %lf, Normalized: %lf (row: %d, col: %d)\n", original_value, x[i][j], i, j);
        }
    }
}

/* --------------------------------------------------- */
void free_Data(struct Data *dataset, int max_rows)
{
    for (int i = 0; i < max_rows; i++)
    {
        free(dataset->values[i]);
        free(dataset->labels[i]);
    }
    free(dataset->values);
    free(dataset->labels);
}

/* -------------------- EOF -------------------------- */