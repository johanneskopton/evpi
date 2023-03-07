from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef("""
double evppi(double* x, double** y, size_t n_samples, size_t n_options);
double* multi_evppi(double** x, double** y, size_t n_samples,
                    size_t n_variables, size_t n_options, double threshold);
double evpi(double** y, size_t n_samples, size_t n_options);
""")

ffibuilder.set_source("_evpi",
                      """
                      #include "evpi.h"
                      """,
                      sources=["evpi/evpi.c"],
                      libraries=["m"],
                      include_dirs=["evpi"])

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
