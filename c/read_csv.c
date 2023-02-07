#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>

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

double** parse_csv(char* path)
{
    FILE *fp;
    fp = fopen(path,"r");
    size_t n_samples = count_samples(fp);
    size_t n_vars = count_vars(fp);
    double** res = get_vals(fp, n_samples, n_vars);
    fclose(fp);
    return(res);
}

int main()
{
    double** x = parse_csv("../test_data/x.csv");
    double** y = parse_csv("../test_data/y.csv");
    
    return 0;
}
