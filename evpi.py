import numpy as np


def ev_pi_bins_sum(x, y, n_bins):
    """Loops through the bins of a histogram over the input and yields the sum
    of the respective output samples if positive, zero otherwise.

    Parameters
    ----------
    x : 1D array
        Input samples.
    y : 1D array
        Output samples.
    n_bins : int
        Number of non-empty histogram bins.

    Returns
    ------
    int
        Sum of optimal decision option outputs for each bin. Can be interpreted
        as the expected value weighted with the number of samples in the bin.
    """

    total_n_bins = n_bins
    filled_n_bins = 0
    while filled_n_bins < n_bins and total_n_bins < 500:
        total_n_bins += filled_n_bins

        # divide the estimate samples into `bins`
        hist, hist_bins = np.histogram(x, bins=total_n_bins)
        filled_n_bins = np.count_nonzero(hist)

    sum_res = 0
    for i in range(total_n_bins):
        # create a binary mask for samples inside the bin
        if i == total_n_bins-1:
            subset_mask = (hist_bins[i] <= x) * (x <= hist_bins[i+1])
        else:
            subset_mask = (hist_bins[i] <= x) * (x < hist_bins[i+1])

        # apply this mask on the output
        y_subset = y[subset_mask]
        # return sum of this output if positive, otherwise zero
        sum_res += max(np.sum(y_subset), 0)

    return sum_res


def evpi(x, y, n_bins=0):
    """Calculates EVPI for one estimate and one decision criterion.
    EVPI means "Expected Value of Perfect Information" and can be described
    as a measure for what a decision maker would be willing to pay for zero
    uncertainty on a certain variable.

    Parameters
    ----------
    x : 1D array_like
        Monte Carlo samples from the probability distribution of the
        considered estimate or "input" variable.
    y : 1D array_like
        The respective utility (aka outcome) samples calculated using the
        estimate samples above. This utility is considered to be the (only)
        decision criterion for a risk-neutral decision maker facing a binary
        decision, so that a positive expected value will lead to `yes` and a
        negative one to `no`.
    n_bins : int
        Number of non-empty bins to use for the histogram. Defaults to 3rd
        root of sample number.
    """
    x = np.array(x)
    if np.all(x == x[0]):
        return 0

    y = np.array(y)

    # expected value in the case of "yes"
    ev_yes = np.mean(y)

    n_samples = len(x)

    # use cubic root of sample number as default
    if n_bins == 0:
        n_bins = int(np.cbrt(n_samples))

    # expected value in the case of "no"
    ev_no = 0

    # expected maximum value
    emv = max(ev_no, ev_yes)

    # expected value in case of perfect information on variable
    ev_pi = ev_pi_bins_sum(x, y, n_bins) / n_samples

    # expected value of perfect information
    evpi = ev_pi - emv

    return evpi
