#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>
#include "evpi.h"

#define MAXCHAR 100

unsigned int count_samples(FILE *fp)
{
    unsigned int samplecount = 0;
    for (char c = getc(fp); c != EOF; c = getc(fp))
        if (c == '\n') samplecount++;
    rewind(fp);
    return samplecount-1;
}

unsigned int count_vars(FILE *fp)
{
    unsigned int varcount = 0;
    for (char c = getc(fp); c != '\n'; c = getc(fp))
        if (c == ',') varcount++;
    rewind(fp);
    return varcount;
}

double** get_vals(FILE *fp, size_t n_samples, size_t n_vars)
{
    // set up 2d array as pointer of pointers
    double *data = malloc(n_samples * n_vars * sizeof(double));
    double **result = malloc(n_samples * sizeof(double*));
    for(size_t i=0; i<n_samples; i++) result[i] = data + i*n_vars;

    // parse csv into this array
    char row[MAXCHAR];
    char *token;
    size_t i, j;
    i = 0;
    fgets(row, MAXCHAR, fp); // skip first row (header)
    while (feof(fp) != true)
    {
        fgets(row, MAXCHAR, fp);
        token = strtok(row, ",");
        token = strtok(NULL, ","); // skip first col (index)
        j = 0;
        while(token != NULL)
        {
            result[i][j] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    return result;
}

double* get_col(double** matrix, size_t n_samples, size_t n_vars, size_t col_idx)
{
    double* column = malloc(n_samples * sizeof(double));
    for(size_t i=0; i<n_samples; i++){
        column[i] = matrix[i][col_idx];
    }
    return column;
}

double** parse_csv(char* path, size_t* n_samples, size_t* n_vars)
{
    FILE *fp;
    fp = fopen(path,"r");
    *n_samples = count_samples(fp);
    *n_vars = count_vars(fp);
    double** matrix = get_vals(fp, *n_samples, *n_vars);
    fclose(fp);
    return(matrix);
}

int main()
{
    size_t n_samples_x, n_samples_y, n_vars_x, n_vars_y;
    double** x = parse_csv("../test_data/x.csv", &n_samples_x, &n_vars_x);
    double** y = parse_csv("../test_data/y.csv", &n_samples_y, &n_vars_y);
    double* x1 = get_col(x, n_samples_x, n_vars_x, 1);
    // double* y2 = get_col(y, n_samples_y, n_vars_y, 2);
    
    double evpi_res = evppi(x1, y, n_samples_x, n_vars_y);
    printf("%f\n", evpi_res);
    
    return 0;
}
