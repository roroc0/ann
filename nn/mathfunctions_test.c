/**
 * @brief Test for functions in mathfunctions.c
 */
/* Includes ------------------------------------------ */
#include "mathfunctions.c"
#include <assert.h>

/* --------------------------------------------------- */
void test_sigmoid();
void test_d_sigmoid();
void test_dotp();

/* --------------------------------------------------- */
void test_sigmoid()
{
    // Test sigmoid with 0
    double result = sigmoid(0);
    assert(fabs(result - 0.5) < 1e-9); // Sigmoid of 0 should be 0.5

    // Test sigmoid with positive input
    result = sigmoid(1);
    assert(fabs(result - 0.7310586) < 1e-7); // Sigmoid of 1 should be approximately 0.7310586

    // Test sigmoid with negative input
    result = sigmoid(-1);
    assert(fabs(result - 0.2689414) < 1e-7); // Sigmoid of -1 should be approximately 0.2689414
}

/* --------------------------------------------------- */
void test_d_sigmoid()
{
    // Test derivative of sigmoid at 0
    double result = d_sigmoid(0);
    assert(fabs(result - 0.25) < 1e-9); // Derivative of sigmoid at 0 should be 0.25

    // Test derivative of sigmoid with positive input
    result = d_sigmoid(1);
    assert(fabs(result - 0.19661193) < 1e-7); // Derivative at 1 should be approximately 0.19661193

    // Test derivative of sigmoid with negative input
    result = d_sigmoid(-1);
    assert(fabs(result - 0.19661193) < 1e-7); // Derivative at -1 should be approximately 0.19661193
}

/* --------------------------------------------------- */
void test_dotp()
{
    // Test dot product with two vectors of size 3
    double a[] = {1.0, 2.0, 3.0};
    double b[] = {4.0, 5.0, 6.0};
    double result = dotp(a, b, 3);
    assert(fabs(result - 32.0) < 1e-9); // Dot product should be 1*4 + 2*5 + 3*6 = 32

    // Test dot product with vectors containing zeros
    double c[] = {0.0, 0.0, 0.0};
    double d[] = {7.0, 8.0, 9.0};
    result = dotp(c, d, 3);
    assert(fabs(result - 0.0) < 1e-9); // Dot product should be 0

    // Test dot product with negative numbers
    double e[] = {-1.0, -2.0, -3.0};
    double f[] = {1.0, 2.0, 3.0};
    result = dotp(e, f, 3);
    assert(fabs(result + 14.0) < 1e-9); // Dot product should be -1*1 + -2*2 + -3*3 = -14
}

/**
 * Main entry for the test.
 */
int main()
{
    test_sigmoid();
    test_d_sigmoid();
    test_dotp();
    return 0;
}
/* -------------------- EOF -------------------------- */