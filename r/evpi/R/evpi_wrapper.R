multi_evppi <- function(x, y){
  x = data.matrix(x)
  y = data.matrix(y)
  if(nrow(x)!=nrow(y)){
   stop("Number of rows must match!") 
  }
  result <- .Call("multi_evppi_wrapper", x, y, 1e-3)
  return(result)
}

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