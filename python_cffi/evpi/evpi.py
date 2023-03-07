import numpy as np

from _evpi import ffi, lib


def evppi(x, y):
    """Calculates EVPPI for one estimate and one decision criterion.
    EVPI means "Expected Value of Perfect Parameter Information" and can be
    described as a measure for what a decision maker would be willing to pay
    for zero uncertainty on a certain variable.

    Parameters
    ----------
    x : 1D array_like
        Monte Carlo samples from the probability distribution of the
        considered estimates or "input" variables. Samples are rows,
        variables are columns.
    y : 2D array_like
        The respective utility (aka outcome) samples calculated using the
        estimate samples of x. This criterion is considered to be the (only)
        decision criterion for a risk-neutral decision maker, that chooses
        the option with the highest expected utility. Samples are rows,
        decision options are columns.
    Returns
    -------
    evppi : float
        Expected Value of Perfect Parameter Information
    """
    x = np.asfortranarray(x, dtype=float)
    y = np.asfortranarray(y, dtype=float)

    xx = ffi.cast("double *", x.ctypes.data)

    yy = ffi.new("double* [%d]" % (y.shape[1]))
    for i in range(y.shape[1]):
        yy[i] = ffi.cast("double *", y[:, i].ctypes.data)

    res = lib.evppi(xx,
                    yy,
                    x.shape[0],
                    y.shape[1])

    return res


def evpi(y):
    """Total EVPI.
    Expected value of making always the best decision. If the model itself is
    deterministic, i.e. the only source of uncertainty are the input variables,
    this value should correspond to the sum of all individual comparative
    EVPIs.

    Parameters
    ----------
    y : 2D array_like
        The respective utility (aka outcome) samples calculated using the
        estimate samples of x. This criterion is considered to be the (only)
        decision criterion for a risk-neutral decision maker, that chooses
        the option with the highest expected utility. Samples are rows,
        decision options are columns.
    """
    y = np.asfortranarray(y, dtype=float)

    yy = ffi.new("double* [%d]" % (y.shape[1]))
    for i in range(y.shape[1]):
        yy[i] = ffi.cast("double *", y[:, i].ctypes.data)

    res = lib.evpi(yy,
                   y.shape[0],
                   y.shape[1])

    return res


def multi_evppi(x, y, significance_threshold=1e-3):
    """Calculate EVPPI for multiple input variables and one output variable.

    Parameters
    ----------
    x : 2D array_like
        Monte Carlo samples from the probability distribution of the
        considered parameter (aka estimates aka "input" variables). Columns
        are variables, rows are samples.
    y : 2D array_like
        The respective utility (aka outcome) samples calculated using the
        estimate samples of x. This criterion is considered to be the (only)
        decision criterion for a risk-neutral decision maker, that chooses
        the option with the highest expected utility. Samples are rows,
        decision options are columns.
    significance_threshold : float
        Percentage of the total EVPI, below which EVPI values will be set to
        zero, since really small positive values are mostly numerical
        artifacts.
    """
    x = np.asfortranarray(x, dtype=float)
    y = np.asfortranarray(y, dtype=float)

    xx = ffi.new("double* [%d]" % (x.shape[1]))
    yy = ffi.new("double* [%d]" % (y.shape[1]))
    for i in range(x.shape[1]):
        xx[i] = ffi.cast("double *", x[:, i].ctypes.data)

    for i in range(y.shape[1]):
        yy[i] = ffi.cast("double *", y[:, i].ctypes.data)

    res_cdata = lib.multi_evppi(xx,
                                yy,
                                x.shape[0],
                                x.shape[1],
                                y.shape[1],
                                significance_threshold)
    res = np.frombuffer(ffi.buffer(
        res_cdata, x.shape[1]*np.dtype(float).itemsize), float)
    return res
