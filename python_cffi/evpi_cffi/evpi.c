#include "evpi.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
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

double minimum(double* vector, size_t length) {
    double result = vector[0];
    for (size_t i = 1; i < length; i++) {
        if (vector[i] < result) {
            result = vector[i];
        }
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

double* get_histogram_bins(double* vector, size_t length, unsigned int n_bins) {
    double* result = malloc((n_bins + 1) * sizeof(double));
    double min = minimum(vector, length);
    double max = maximum(vector, length);
    double step = (max - min) / (double)n_bins;
    for (unsigned int i = 0; i < n_bins + 1; i++) {
        result[i] = min + (double)i * step;
    }
    return result;
}

double calc_ev_pi(double* x, double** y, size_t n_samples, size_t n_options,
                  unsigned int n_bins) {
    double* histogram_bins = get_histogram_bins(x, n_samples, n_bins);
    double* y_subset_sum = malloc(n_options * sizeof(double));
    double sum_res = 0;
    // iterate over all bins
    for (unsigned int bin_i = 0; bin_i < n_bins; bin_i++) {
        // (re)set y_subset_sum to zero
        for (size_t option_i = 0; option_i < n_options; option_i++) {
            y_subset_sum[option_i] = 0;
        }
        // iteratre over all samples
        for (size_t sample_i = 0; sample_i < n_samples; sample_i++) {
            // check if x value is in bin (continue otherwise)
            if (bin_i == n_bins - 1) {
                if (histogram_bins[bin_i] > x[sample_i] ||
                    x[sample_i] > histogram_bins[bin_i + 1])
                    continue;
            } else {
                if (histogram_bins[bin_i] > x[sample_i] ||
                    x[sample_i] >= histogram_bins[bin_i + 1])
                    continue;
            }
            // if in bin iterate over decision options
            for (size_t option_i = 0; option_i < n_options; option_i++) {
                y_subset_sum[option_i] += y[sample_i][option_i];
            }
        }
        sum_res += maximum(y_subset_sum, n_options);
    }
    return sum_res / (double)n_samples;
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
    double ev_pi = calc_ev_pi(x, y, n_samples, n_options, n_bins);
    return ev_pi - emv;
}