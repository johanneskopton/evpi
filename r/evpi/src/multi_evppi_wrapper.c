#include "evpi.h"
#include <R.h>
#include <Rdefines.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

double **create_c_matrix(double *flat_fortran_array, size_t n_rows,
                         size_t n_cols) {
  double *data = malloc(n_cols * n_rows * sizeof(double));
  double **res = malloc(n_cols * sizeof(double *));
  for (size_t i = 0; i < n_cols; i++) {
    res[i] = data + i * n_rows;
    for (size_t j = 0; j < n_rows; j++) {
      res[i][j] = flat_fortran_array[j + i * n_rows];
    }
  }
  return res;
}

SEXP multi_evppi_wrapper(SEXP x, SEXP y, SEXP significance_threshold) {
  size_t n_samples = nrows(x);
  size_t n_variables = ncols(x);
  size_t n_options = ncols(y);

  double *c_out;
  double *c_multi_evppi = malloc(n_variables * sizeof(double));

  double **xx = create_c_matrix(REAL(x), n_samples, n_variables);
  double **yy = create_c_matrix(REAL(y), n_samples, n_options);

  c_multi_evppi = multi_evppi(xx, yy, n_samples, n_variables, n_options,
                              asReal(significance_threshold));

  SEXP out = PROTECT(allocVector(REALSXP, n_variables));
  c_out = REAL(out);

  for (size_t i = 0; i < n_variables; i++) {
    c_out[i] = c_multi_evppi[i];
  }

  UNPROTECT(1);
  return out;
}