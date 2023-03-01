#include "evpi.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

bool is_const(const double array[], int n) {
    const int a0 = array[0];
    for (int i = 1; i < n; i++) {
        if (array[i] != a0)
            return false;
    }
    return true;
}

double* mean_samples(double** matrix, size_t n_samples, size_t n_vars) {
    double* result = malloc(n_vars * sizeof(double));
    for (size_t j = 0; j < n_vars; j++) {
        double sum = 0;
        for (size_t i = 0; i < n_samples; i++) {
            sum += matrix[i][j];
        }
        result[j] = sum / n_samples;
    }
    return result;
}

double maximum(double* vector, size_t length) {
    double result = vector[0];
    for (size_t i = 1; i < length; i++) {
        if (vector[i] > result) {
            result = vector[i];
        }
    }
    return result;
}

double evppi(double* x, double** y, size_t n_samples, size_t n_options) {
    /*
    Check if there is variance (-> uncertainty) in the input.
    If there is none, then there can't be any value in reducing it.
    */
    if (is_const(x, n_samples)) {
        return 0;
    }

    // Use cubic root of sample number as default bin number.
    unsigned int n_bins = (unsigned int)cbrt(n_samples);

    // expected maximum value
    double emv = maximum(mean_samples(y, n_samples, n_options), n_options);
    return 1;
}