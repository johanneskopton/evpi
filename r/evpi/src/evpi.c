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
            sum += matrix[j][i];
        }
        result[j] = sum / n_samples;
    }
    return result;
}

double* max_vars(double** matrix, size_t n_samples, size_t n_vars) {
    double* result = malloc(n_samples * sizeof(double));
    double max_val;
    for (size_t i = 0; i < n_samples; i++) {
        max_val = matrix[0][i];
        for (size_t j = 1; j < n_vars; j++) {
            if (matrix[j][i] > max_val) {
                max_val = matrix[j][i];
            }
        }
        result[i] = max_val;
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

double mean(double* vector, size_t length) {
    double sum = 0;
    for (size_t i = 0; i < length; i++) {
        sum += vector[i];
    }
    double result = sum / length;
    return result;
}

double* get_row(double** matrix, size_t row_idx) { return matrix[row_idx]; }

double calc_ev_pi(double* x, double** y, size_t n_samples, size_t n_options,
                  unsigned int n_bins) {
    double* y_subset_sum = malloc(n_options * sizeof(double));
    double sum_res = 0;

    unsigned int* bin_idxs = malloc(n_samples * sizeof(unsigned int));
    double x_min = minimum(x, n_samples);
    double x_max = maximum(x, n_samples);
    double bin_size = (x_max - x_min) / n_bins;
    // iterate over all samples
    for (size_t sample_i = 0; sample_i < n_samples; sample_i++) {
        bin_idxs[sample_i] = (int)((x[sample_i] - x_min) / bin_size);
    }

    // iterate over all bins
    for (unsigned int bin_i = 0; bin_i < n_bins; bin_i++) {
        // (re)set y_subset_sum to zero
        for (size_t option_i = 0; option_i < n_options; option_i++) {
            y_subset_sum[option_i] = 0;
        }
        // iterate over all samples
        for (size_t sample_i = 0; sample_i < n_samples; sample_i++) {
            if (bin_idxs[sample_i] != bin_i)
                continue;
            // if in bin iterate over decision options
            for (size_t option_i = 0; option_i < n_options; option_i++) {
                y_subset_sum[option_i] += y[option_i][sample_i];
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

double evpi(double** y, size_t n_samples, size_t n_options) {
    double emv = maximum(mean_samples(y, n_samples, n_options), n_options);
    double ev_pi = mean(max_vars(y, n_samples, n_options), n_samples);
    return ev_pi - emv;
}

double* multi_evppi(double** x, double** y, size_t n_samples,
                    size_t n_variables, size_t n_options, double threshold) {
    double evpi_val = evpi(y, n_samples, n_options);
    double* x_var;
    double* evppi_val = malloc(n_variables * sizeof(double));
    for (size_t variable_i = 0; variable_i < n_variables; variable_i++) {
        x_var = get_row(x, variable_i);
        evppi_val[variable_i] = evppi(x_var, y, n_samples, n_options);
        if (evppi_val[variable_i] < evpi_val * threshold)
            evppi_val[variable_i] = 0;
    }
    return evppi_val;
}