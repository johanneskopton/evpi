
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10728646.svg)](https://doi.org/10.5281/zenodo.10728646)


# A fast implementation of the Expected Value of Perfect Parameter Information (EVPPI) for large Monte Carlo simulations

In this repository you can find 4 things:

* a Python/Numpy implementation in [`python`](./python/README.md)
* a C implementation in `c`
* R bindings to the C implementation in [`r`](./r/evpi/README.md)
* Python bindings to the C implementation using CFFI in [`python_cffi`](./python_cffi/README.md)

The _Expected Value of Perfect Parameter Information_ (EVPPI) is a concept from decision analysis (modeling decisions under uncertainty). It can be described as a measure for what a (rational) decision-maker would be willing to pay for zero uncertainty on a certain variable.

In general, the functions in this repository take in samples from a Monte Carlo model that predicts utility as a function of uncertain input parameters. Here, `x` denotes the values of the (uncertain) parameter inputs and `y` the resulting utility. More detailed documentation can be found in the respective packages.

Running the C implementation from R was found to be many times faster than existing R implementations, especially for a large number of Monte Carlo samples.

Details and limitations regarding the algorithmic approach can be found in _Brennan et al. (2007)_[^1]. There are more sophisticated approaches [^2] [^3] with advantages in some use cases, but a fast and stable implementation of this basic algorithm was considered useful for science and practice.

## References
[^1]: Brennan, A., Kharroubi, S., O’Hagan, A., & Chilcott, J. (2007). Calculating Partial Expected Value of Perfect Information via Monte Carlo Sampling Algorithms. Medical Decision Making, 27(4), 448–470. https://doi.org/10.1177/0272989X07302555

[^2]: Strong, M., & Oakley, J. E. (2013). An Efficient Method for Computing Single-Parameter Partial Expected Value of Perfect Information. https://journals.sagepub.com/doi/10.1177/0272989X12465123

[^3]: Strong, M., Oakley, J. E., & Brennan, A. (2014). Estimating Multiparameter Partial Expected Value of Perfect Information from a Probabilistic Sensitivity Analysis Sample: A Nonparametric Regression Approach. Medical Decision Making, 34(3), 311–326. https://doi.org/10.1177/0272989X13505910
