#ifndef EVPI_H
#define EVPI_H

#include <stddef.h>

/*
x : 1D array
    Monte Carlo samples from the probability distribution of the
    considered estimates or "input" variables. Samples are rows,
    variables are columns.
y : 2D array
    The respective utility (aka outcome) samples calculated using the
    estimate samples of x. This criterion is considered to be the (only)
    decision criterion for a risk-neutral decision maker, that chooses
    the option with the highest expected utility. Samples are rows,
    decision options are columns.
*/
double evppi(double* x, double** y, size_t n_samples, size_t n_options);

#endif