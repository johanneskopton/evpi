import numpy as np


def evpi(x, y, n_bins):
    """Calculates EVPI for one estimate and one decision criterion.
    EVPI means "Expected Value of Perfect Information" and can be described
    as a measure for what a decision maker would be willing to pay for zero
    uncertainty on a certain variable.

    Parameters
    ----------
    x : array_like
        Monte Carlo samples from the probability distribution of the
        considered estimate or "input" variable.
    y : array_like
        The respective criterion or "output" samples calculated using the
        estimate samples above. This criterion is considered to be the (only)
        decision criterion for a risk-neutral decision maker facing a binary
        decision, so that a positive expected value will lead to `yes` and a
        negative one to `no`.
    n_bins : int
        Number of bins to use for the histogram.
    """

    # expected value in the case of "yes"
    ev_yes = np.mean(y)

    # expected value in the case of "no"
    ev_no = 0

    # expected maximum value
    emv = max(ev_no, ev_yes)

    def ev_pi_bins(x, y, n_bins):
        """Loops through the bins of a histogram over the input and yields the
        sum of the respective output samples if positive, zero otherwise.

        Parameters
        ----------
        x : array_like
            Input samples.
        y : array_like
            Output samples.
        n_bins : int
            Number of histogram bins.

        Yields
        ------
        int
            Sum of optimal decision option output for each bin. Can be
            interpreted as the expected value weighted with the number of
            samples in the bin.
        """

        # divide the estimate samples into `n_bins`
        hist_bins = np.histogram(x, bins=n_bins)[1]

        for i in range(n_bins):
            # create a binary mask for samples inside the bin
            subset_mask = (hist_bins[i] <= x) * (x < hist_bins[i+1])
            # apply this mask on the output
            y_subset = y[subset_mask]
            # return sum of this output if positive, otherwise zero
            yield max(np.sum(y_subset), 0)

    ev_pi = sum(ev_pi_bins(x, y, n_bins)) / len(x)
    evpi = ev_pi - emv

    return evpi
