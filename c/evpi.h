#ifndef EVPI_H
#define EVPI_H

#include <stddef.h>

/*
    Calculates EVPPI for one estimate and one decision criterion.
    EVPI means "Expected Value of Perfect Parameter Information" and can be
    described as a measure for what a decision maker would be willing to pay
    for zero uncertainty on a certain variable.

    Parameters
    ----------
    x : 1D array
        Monte Carlo samples from the probability distribution of the
        considered estimates or "input" variables. Samples are columns,
        variables are rows(!).
    y : 2D array
        The respective utility (aka outcome) samples calculated using the
        estimate samples of x. This criterion is considered to be the (only)
        decision criterion for a risk-neutral decision maker, that chooses
        the option with the highest expected utility. Samples are columns,
        decision options are rows(!).
*/
double evppi(double* x, double** y, size_t n_samples, size_t n_options);

/*
    "Calculate EVPPI for multiple input variables and one output variable.

    Parameters
    ----------
    x : 2D array_like
        Monte Carlo samples from the probability distribution of the considered
        parameter(aka estimates aka "input" variables). Rows are variables,
        columns are samples(!).
    y : 2D array_like
        The respective utility(aka outcome) samples calculated using the
        estimate samples of x. This criterion is considered to be the(only)
        decision criterion for a risk-neutral decision maker, that chooses the
        option with the highest expected utility. Samples are columns, decision
        options are rows(!).
    threshold : float
        Percentage of the total EVPI, below which EVPI values will be set to
        zero, since really small positive values are mostly numerical
        artifacts.
*/
double* multi_evppi(double** x, double** y, size_t n_samples,
                    size_t n_variables, size_t n_options, double threshold);

/*
    Total EVPI.
    Expected value of making always the best decision. If the model itself is
    deterministic, i.e. the only source of uncertainty are the input variables,
    this value should be less than the sum of all individual comparative
    EVPIs.

    Parameters
    ----------
    y : 2D array_like
        The respective utility (aka outcome) samples calculated using the
        estimate samples of x. This criterion is considered to be the (only)
        decision criterion for a risk-neutral decision maker, that chooses
        the option with the highest expected utility. Samples are columns,
        decision options are rows(!).
*/
double evpi(double** y, size_t n_samples, size_t n_options);

#endif