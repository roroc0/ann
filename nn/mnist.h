/**
 * @file MNIST Dataset helper functions header file
 * @brief Header file containing helper functions for parsing and normalizing the MNIST dataset.
 */

#ifndef NN_MNIST_H
#define NN_MNIST_H

/* Includes ------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------- */
#define MAX_COLUMNS 785
#define MAX_ROWS_TRAIN 60000
#define MAX_ROWS_TEST 10000

/**
 * @brief Struct to store MNIST dataset values and labels.
 *
 * This struct holds pointers to two-dimensional arrays:
 * - `values`: The input data (features).
 * - `labels`: The corresponding labels (outputs).
 */
struct Data
{
    double **values; /**< Pointer to the 2D array storing the feature values */
    double **labels; /**< Pointer to the 2D array storing the labels */
};

/**
 * @brief Parse and normalize MNIST data from a CSV file.
 *
 * This function reads MNIST data from a CSV file, normalizes the data,
 * and stores it in the `Data` struct. The function expects the CSV to 
 * contain image data and labels in a format suitable for MNIST.
 *
 * @param filename The path to the CSV file containing the MNIST data.
 * @param num_rows The number of rows (samples) to read from the file.
 * @param num_classes The number of classes (labels) in the dataset.
 * @return A `Data` struct containing the normalized values and labels.
 */
struct Data parse_MNIST_CSV_and_normalize(const char *filename, int num_rows, int num_classes);

/**
 * @brief Normalize the data to a given range.
 *
 * This function normalizes the input data to a specified range defined by `min` and `max`.
 * Typically, MNIST data is normalized to a range of 0 to 1 or -1 to 1.
 *
 * @param x Pointer to the 2D array of data to be normalized.
 * @param rows The number of rows (samples) in the data.
 * @param cols The number of columns (features) in the data.
 * @param max The maximum value in the target normalization range.
 * @param min The minimum value in the target normalization range.
 */
void normalize_data(double **x, int rows, int cols, double max, double min);

/**
 * @brief Free the memory allocated for the MNIST dataset.
 *
 * This function frees the memory allocated for the `values` and `labels` in a `Data` struct.
 *
 * @param dataset Pointer to the `Data` struct containing the MNIST dataset.
 * @param max_rows The maximum number of rows (samples) in the dataset.
 */
void free_Data(struct Data *dataset, int max_rows);

/* --------------------------------------------------- */

#endif //NN_MNIST_H
