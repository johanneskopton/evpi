import numpy as np


    """Loops through the bins of a histogram over the input and yields the sum
    of the respective output samples if positive, zero otherwise.
def calc_ev_pi(x, y, n_bins, min_samples_per_bin):

    Parameters
    ----------
    x : 1D array
        Input samples.
    y : 1D array
        Output samples.
    n_bins : int
        Number of non-empty histogram bins.
    min_samples_per_bin: int
        Since the expected value for just a handful of samples will
        always describe this bin a lot better than the expected
        value of the entire population (even though the sampling might
        have been completely random), this will seem like there is
        significant information value, when in fact there is not.
        Therefore bins with too few samples are kicked out.

    Returns
    ------
    int
        Sum of optimal decision option outputs for each bin. Can be interpreted
        as the expected value weighted with the number of samples in the bin.
    """

    # increase the number of total bins, so we have at least `n_bins`
    # bins with enough samples in them
    total_n_bins = n_bins
    n_bins_sufficient = 0
    # the 1000 is just a safety limit to avoid infinite loops for
    # really weird input distributions
    while n_bins_sufficient < n_bins and total_n_bins < 1000:
        total_n_bins += n_bins_sufficient

        # divide the estimate samples into `bins`
        hist, hist_bins = np.histogram(x, bins=total_n_bins)
        # check which histogram bins have enough samples
        sufficiency_mask = hist >= min_samples_per_bin
        # count number of bins with enough samples
        n_bins_sufficient = np.count_nonzero(sufficiency_mask)

    # get indices of bins with enough samples
    sufficient_bin_idxs = np.nonzero(sufficiency_mask)[0]
    # calculate the total number of samples in these bins (for
    # normalization later)
    n_samples_considered = np.sum(hist[sufficiency_mask])

    sum_res = 0
    for i in sufficient_bin_idxs:
        # create a binary mask for samples inside the bin
        if i == total_n_bins-1:
            subset_mask = (hist_bins[i] <= x) * (x <= hist_bins[i+1])
        else:
            subset_mask = (hist_bins[i] <= x) * (x < hist_bins[i+1])

        # apply this mask on the output
        y_subset = y[subset_mask]
        # return sum of this output if positive, otherwise zero
        sum_res += max(np.sum(y_subset), 0)

    # now the normalization
    ev_pi = sum_res / n_samples_considered
    return ev_pi


def evpi(x, y, n_bins=0, min_samples_per_bin=10):
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
    min_samples_per_bin: int, default=10
        Since the expected value for just a handful of samples will
        always describe this bin a lot better than the expected
        value of the entire population (even though the sampling might
        have been completely random), this will seem like there is
        significant information value, when in fact there is not.
        Therefore bins with too few samples are kicked out.
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
    ev_pi = calc_ev_pi(x, y, n_bins, min_samples_per_bin)

    # expected value of perfect information
    evpi = ev_pi - emv

    return evpi
