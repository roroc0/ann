/**
 * @file Mathfunctions source file
 * @brief Different Math function definitions
 */

/* Includes ------------------------------------------ */
#include "mathfunctions.h"
#include <omp.h>
#include <immintrin.h>

/* --------------------------------------------------- */
double sigmoid(double x){
    return (1/(1+ exp(-x)));
}

/* --------------------------------------------------- */
double d_sigmoid(double x){
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// Default to SEQ if no flag is defined
#if !defined(SEQ) && !defined(PARALLEL) && !defined(SIMD)
#define SEQ
#endif

#if defined(SEQ)  // Sequential version
double dotp(const double *a, const double *b, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#elif defined(PARALLEL)  // OpenMP parallel version
double dotp(const double *a, const double *b, int size) {
    double sum = 0.0;
      #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#elif defined(SIMD)  // SIMD version with AVX instructions
double dotp(const double *a, const double *b, int size) {
    __m256d sum = _mm256_setzero_pd();  // accumulator for partial sums

    int i;
    for (i = 0; i <= size - 4; i += 4) {
        // Load four double-precision floats from each array
        __m256d vec1 = _mm256_loadu_pd(&a[i]);
        __m256d vec2 = _mm256_loadu_pd(&b[i]);

        // Perform element-wise multiplication
        __m256d prod = _mm256_mul_pd(vec1, vec2);

        // Accumulate the results
        sum = _mm256_add_pd(sum, prod);
    }

    // Horizontal addition of the 4 elements in the AVX register
    double sums[4];
    _mm256_storeu_pd(sums, sum);
    double final_sum = sums[0] + sums[1] + sums[2] + sums[3];

    // Handle remaining elements
    for (; i < size; ++i) {
        final_sum += a[i] * b[i];
    }

    return final_sum;
}
#endif
/* -------------------- EOF -------------------------- */


