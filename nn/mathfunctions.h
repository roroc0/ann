/**
 * @file Mathfunctions header file
 * @brief Different Math function prototypes declaration
 */

#ifndef NN_MATHFUNCTIONS_H
#define NN_MATHFUNCTIONS_H

/* Includes ------------------------------------------ */
#include <math.h>
#include <stdio.h>
/* --------------------------------------------------- */

/**
 * @brief Sigmoid activation function
 * @param x the variable
 * @return value x after calculation
 */
double sigmoid(double x);
/* --------------------------------------------------- */

/**
 * @brief Sigmoid derivative function
 * @param x the variable
 * @return value x after calculation
 */
double d_sigmoid(double x);
/* --------------------------------------------------- */

/**
 * @brief Calculates the scalar product of two vectors
 * @param a vector a
 * @param b vector b
 * @param size number of elements in vectors
 * @return the scalar product
 */
 double dotp(const double *a, const double *b, int size);
/* --------------------------------------------------- */

#endif //NN_MATHFUNCTIONS_H
