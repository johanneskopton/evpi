import numpy as np

from evpi import evpi, evppi, multi_evppi


def _fill_y(y):
    return np.c_[y, np.zeros(y.shape[0])]


def binary_evppi(x, y):
    """Calculates EVPPI for one estimate and one decision criterion.
    EVPI means "Expected Value of Perfect Parameter Information" and can be
    described as a measure for what a decision maker would be willing to pay
    for zero uncertainty on a certain parameter.

    Parameters
    ----------
    x : 1D array_like
        Monte Carlo samples from the probability distribution of the
        considered parameter (aka estimate aka "input" variable).
    y : 1D array_like
        The respective utility (aka outcome) samples calculated using the
        estimate samples above. This utility is considered to be the (only)
        decision criterion for a risk-neutral decision maker facing a binary
        decision, so that a positive expected value will lead to `yes` and a
        negative one to `no`.
    """
    return evppi(x, _fill_y(y))


def binary_evpi(y):
    """Total EVPI.
    Expected value of making always the best decision. If the model itself is
    deterministic, i.e. the only source of uncertainty are the input variables,
    this value should correspond to the sum of all individual EVPIs.

    Parameters
    ----------
    y : 1D array_like
        Monte carlo samples of model output (utility). S. `evppi`.
    """
    return evpi(_fill_y(y))


def binary_multi_evppi(x, y, significance_threshold=5e-2):
    """Calculate evppi for multiple input variables and one output variable.

    Parameters
    ----------
    x : 2D array_like
        Monte Carlo samples from the probability distribution of the
        considered estimates or "input" variables. Columns are variables,
        rows are samples.
    y : 1D array_like
        The respective utility (aka outcome) samples calculated using the
        estimate samples above. This utility is considered to be the (only)
        decision criterion for a risk-neutral decision maker facing a binary
        decision, so that a positive expected value will lead to `yes` and a
        negative one to `no`.
    significance_threshold : float
        Percentage of the total EVPI, below which EVPI values will be set to
        zero, since really small positive values are mostly numerical
        artifacts.
    """
    return multi_evppi(x, _fill_y(y), significance_threshold)
