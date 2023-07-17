#' Calculate Expected Value of Perfect Parameter Information (EVPPI) for
#' multiple input variables and multiple decision options.
#'
#' @param x Monte Carlo samples from the probability distribution of the
#' considered parameter (aka estimates aka "input" variables). Columns are
#' variables, rows are samples.
#' @param y The respective utility (aka outcome) samples calculated using the
#' estimate samples of x. This criterion is considered to be the (only)
#' decision criterion for a risk-neutral decision maker, that chooses
#' the option with the highest expected utility. Samples are rows,
#' decision options are columns.
#' @return Vector of EVPPI values in the order of the columns of `x`.
multi_evppi <- function(x, y){
  x = data.matrix(x)
  y = data.matrix(y)
  if(nrow(x)!=nrow(y)){
   stop("Number of rows must match!") 
  }
  result <- .Call("multi_evppi_wrapper", x, y, 1e-3)
  return(result)
}

#' Calculate Expected Value of Perfect Parameter Information (EVPPI) for
#' multiple input variables and two decision options (yes/no).
#'
#' @param x Monte Carlo samples from the probability distribution of the
#' considered parameter (aka estimates aka "input" variables). Columns are
#' variables, rows are samples.
#' @param y The respective utility (aka outcome) samples calculated using the
#' estimate samples above. This utility is considered to be the (only)
#' decision criterion for a risk-neutral decision maker facing a binary
#' decision, so that a positive expected value will lead to `yes` and a
#' negative one to `no`.
#' @return Vector of EVPPI values in the order of the columns of x.
binary_multi_evppi <- function(x, y){
  x = data.matrix(x)
  y = data.matrix(y)
  if(nrow(x)!=nrow(y) || ncol(y)!=1){
   stop("Number of rows must match!") 
  }
  y_full = cbind(y, 0)
  result <- .Call("multi_evppi_wrapper", x, y_full, 1e-3)
  return(result)
}
