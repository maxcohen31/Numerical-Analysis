/*
 *  LU decomposition - Doolitle's algorithm implementation with partial pivoting
 *
 *  { Ly=b
 *  { Ux=y
 *
 *  Spring 2025 - Author: Abbate Emanuele
 *
 *  References:  https://www.cs.gordon.edu/courses/mat342/handouts/gauss.pdf 
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
void swapRows(double **a, int r1, int r2)
{
    /* swap two rows of a given matrix */
    double *temp = a[r1];
    a[r1] = a[r2];
    a[r2] = temp;
}

void partialPivoting(double **a, double **l, double **p, int n)
{
    // partial pivoting subroutine
    for (int k = 0; k < n; ++k)
    {
        // find the row with the max absolute value in the column k 
        int pivotRow = k; // let's say the row with the maximum pivot is k 
        double pivot = fabs(a[k][k]); // we suppose the pivot is the matrix diagonal element
        for (int i = k; i < n; ++i)
        {
            if (fabs(a[i][k]) > pivot)
            {
                pivot = fabs(a[i][k]); // take the absoulte value of the max element
                pivotRow = i; // update the row
            }
        }
        // swap rows if the index of the column k and the row index are different
        // we make this swap since we want the max absolute value element in the main diagonal
        if (pivotRow != k)
        {
            swapRows(a, k, pivotRow); // update A
            swapRows(p, k, pivotRow); // update P
        }
        if (k > 0) // k = 0 -> we did not calculate L yet
        {
            // update L
            for (int i = 0; i < k; ++i)
            {
                double temp = l[k][i];
                l[k][i] = l[pivotRow][i];
                l[pivotRow][i] = temp;
            }
        }    
    }
}

void showMatrix(double **m, int d)
{
    for (int i = 0; i < d; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            printf("%8.4f ", m[i][j]);
        }
        printf("\n");
    }
}

void deallocateMemory(double **a, double **l, double **u, int d)
{
    for (int i = 0; i < d; ++i)
    {
        free(a[i]);
        free(l[i]);
        free(u[i]);
    }
    free(a);
    free(l);
    free(u);
}

int main(int argc, char **argv)
{
    
    srand(time(NULL));
    printf("Insert a dimension: ");
    int dim;
    if (scanf("%d", &dim) != 1)
    {
        fprintf(stderr, "Error scanf\n");
        exit(EXIT_FAILURE);
    }

    double **A = malloc(dim * sizeof(double*));
    double **L = malloc(dim * sizeof(double*));
    double **U = malloc(dim * sizeof(double*));
    if (!A || !L || !U)
    {
        printf("Error allocating memory for the matrix\n");
        exit(1);
    }
    for (int i = 0; i < dim; ++i)
    {
        A[i] = malloc(dim * sizeof(double));
        L[i] = malloc(dim * sizeof(double));
        U[i] = malloc(dim * sizeof(double));
        for (int j = 0; j < dim; ++j)
        {
            A[i][j] = 20000.0 * (double)(rand() / (double)RAND_MAX - 0.5); /* [-10000.0, 10000.0] */
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }
    return 0;
}
