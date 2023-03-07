#include "evpi.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCHAR 100

unsigned int count_samples(FILE* fp) {
    unsigned int samplecount = 0;
    for (char c = getc(fp); c != EOF; c = getc(fp))
        if (c == '\n')
            samplecount++;
    rewind(fp);
    return samplecount - 1;
}

unsigned int count_vars(FILE* fp) {
    unsigned int varcount = 0;
    for (char c = getc(fp); c != '\n'; c = getc(fp))
        if (c == ',')
            varcount++;
    rewind(fp);
    return varcount;
}

double** get_vals(FILE* fp, size_t n_samples, size_t n_vars) {
    // set up 2d array as pointer of pointers
    double* data = malloc(n_samples * n_vars * sizeof(double));
    double** result = malloc(n_vars * sizeof(double*));
    for (size_t i = 0; i < n_vars; i++)
        result[i] = data + i * n_samples;

    // parse csv into this array
    char row[MAXCHAR];
    char* token;
    size_t i, j;
    i = 0;
    fgets(row, MAXCHAR, fp); // skip first row (header)
    while (feof(fp) != true) {
        fgets(row, MAXCHAR, fp);
        token = strtok(row, ",");
        token = strtok(NULL, ","); // skip first col (index)
        j = 0;
        while (token != NULL) {
            result[j][i] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    return result;
}

double** parse_csv(char* path, size_t* n_samples, size_t* n_vars) {
    FILE* fp;
    fp = fopen(path, "r");
    *n_samples = count_samples(fp);
    *n_vars = count_vars(fp);
    double** matrix = get_vals(fp, *n_samples, *n_vars);
    fclose(fp);
    return (matrix);
}

int main() {
    size_t n_samples_x, n_samples_y, n_vars_x, n_vars_y;
    double** x = parse_csv("../test_data/x.csv", &n_samples_x, &n_vars_x);
    double** y = parse_csv("../test_data/y.csv", &n_samples_y, &n_vars_y);

    double* xvar;
    double evppi_res;
    double reference_evppi[3] = {7.1, 2.3, 9.9};
    for (unsigned char i = 0; i < 3; i++) {
        xvar = x[i];
        // double* y2 = get_col(y, n_samples_y, n_vars_y, 2);

        evppi_res = evppi(xvar, y, n_samples_x, n_vars_y);
        if (fabs(evppi_res - reference_evppi[i]) > 0.5) {
            printf("Wrong EVPI for variable %i: %f is not %f\n", i, evppi_res,
                   reference_evppi[i]);
            return 1;
        }
    }
    printf("Test passed.\n");

    return 0;
}
