# R wrapper of C implementation of EVPI

## Install

```
devtools::install_github("johanneskopton/evpi", subdir="r/evpi")
```

## Run

```
evpi::multi_evppi(x, y)
```

with x being the Monte Carlo inputs (rows are samples, cols are variables) and y being the Monte Carlo outputs (rows are samples, cols are decision options). Format can be everything, that is allowed by data.matrix e.g. a dataframe. The result is an unnamed vector with the EVPPIs in the order of the columns of x.

If you want to put in a difference between two decision options (as in [decisionSupport](https://github.com/eikeluedeling/decisionSupport)), you can use

```
evpi::binary_multi_evppi(x, y)
```

with y having just one column.
