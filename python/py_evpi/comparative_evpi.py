import numpy as np


def _calc_comparative_ev_pi(x, y, n_bins):
    """Loops through the bins of a histogram over the input and returns the
    highest sum of the respective output samples.

    Parameters
    ----------
    x : 1D array
        Input samples.
    y : 2D array
        Output samples.
    n_bins : int
        Number of non-empty histogram bins.

    Returns
    ------
    int
        Sum of optimal decision option output for each bin. Can be
        interpreted as the expected value weighted with the number of
        samples in the bin.
    """

    # Since the expected value for just a handful of samples will
    # always describe this bin a lot better than the expected
    # value of the entire population (even though the sampling might
    # have been completely random), this might seem like there is
    # significant information value, when in fact there is not.
    # Therefore bins with too few samples are kicked out.
    # (Up to this point, this value showed no big impact in my tests.)
    MIN_SAMPLES_PER_BIN = 1

    # Just a safety limit to avoid infinite loops for really weird input
    # distributions
    MAX_N_BINS = 1000

    # increase the number of total bins, so we have at least `n_bins`
    # bins with enough samples in them
    total_n_bins = n_bins
    n_bins_sufficient = 0
    while n_bins_sufficient < n_bins and total_n_bins < MAX_N_BINS:
        total_n_bins += n_bins_sufficient

        # divide the estimate samples into histogram bins
        hist, hist_bins = np.histogram(x, bins=total_n_bins)
        # check which histogram bins have enough samples
        sufficiency_mask = hist >= MIN_SAMPLES_PER_BIN
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
        y_subset = y[subset_mask, :]
        # get the sum over all samples (aka weighted expected value)
        y_subset_sum = np.sum(y_subset, axis=0)
        # `np.sum(y_subset)` can be considered the expected outcome for
        # this bin multiplied by number of samples in this bin. Since we
        # use the maximum value among all choices, we simulate knowing
        # that this bin contains the true sample.
        # By summing and normalization with `n_samples_considered`, we get the
        # weighted sum of expected outcomes.
        sum_res += np.max(y_subset_sum)

    # now the normalization
    ev_pi = sum_res / n_samples_considered
    return ev_pi


def comparative_evppi(x, y, n_bins=None):
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
    n_bins : int
        Number of non-empty bins to use for the histogram. Defaults to 3rd
        root of sample number.
    """
    x = np.array(x)
    if np.all(x == x[0]):
        return 0
    y = np.array(y)

    n_samples = x.shape[0]

    # use cubic root of sample number as default
    if n_bins is None:
        n_bins = int(np.cbrt(n_samples))

    # expected values for all options
    ev = np.mean(y, axis=0)

    # expected maximum value
    emv = np.max(ev)

    ev_pi = _calc_comparative_ev_pi(x, y, n_bins)
    evppi = ev_pi - emv

    return evppi


def comparative_evpi(y):
    """Comparative total EVPI.
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

    y = np.array(y)

    # expected value in the case of "yes"
    ev = np.mean(y, axis=0)

    # expected maximum value
    emv = np.max(ev)

    # outcome given perfect information
    y_pi = np.max(y, axis=1)

    # expected value given perfect information
    ev_pi = np.mean(y_pi)

    # mean and max are basically swapped

    # expected value of perfect information
    evpi = ev_pi - emv

    return evpi


def comparative_multi_evppi(x, y, n_bins=None, significance_threshold=1e-3):
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
    n_bins : int
        Number of non-empty bins to use for the histogram.
    significance_threshold : float
        Percentage of the total EVPI, below which EVPI values will be set to
        zero, since really small positive values are mostly numerical
        artifacts.
    """
    x = np.array(x)
    y = np.array(y)

    n_variables = x.shape[1]
    evppi_results = np.zeros(n_variables)
    evpi_result = comparative_evpi(y)
    for i in range(n_variables):
        this_evpi = comparative_evppi(x[:, i], y, n_bins)

        # Since this method tends to overestimate EVPIs, that are actually
        # zero, we want to test, if the EVPI is "significant" (not in the
        # sense of a statistical test).
        if this_evpi < evpi_result * significance_threshold:
            this_evpi = 0
        evppi_results[i] = this_evpi

    return evppi_results
