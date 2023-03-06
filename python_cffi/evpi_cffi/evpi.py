import numpy as np

from _evpi_cffi import ffi, lib


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
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    xx = ffi.cast("double *", x.ctypes.data)

    yy = ffi.new("double* [%d]" % (y.shape[0]))
    y_list = [np.array(y_row, dtype=np.float64) for y_row in y]
    for i in range(y.shape[0]):
        yy[i] = ffi.cast("double *", y_list[i].ctypes.data)

    res = lib.evppi(xx,
                    yy,
                    x.shape[0],
                    y.shape[1])

    return res
