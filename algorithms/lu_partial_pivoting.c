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
#include <string.h>

#define TOL 1e-12

/* declaration of functions */
double **makeCopy(double **a, int dim);
void swapRows(double **a, int r1, int r2);
void showMatrix(double **a, int d);
void partPivLU(double **a, double **l, double **u, double **p, int n);
void deallocateMemory(double **a, double **l, double **u, double **p, int d);
void matrixMultiply(double **a, double **b, double **result, int n) ;
bool checkPALU(double **a, double **p, double **l, double **u, int dim);
double maxDiff(double **pa, double **lu, int n);

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
    double **P = malloc(dim * sizeof(double*));
    if (!A || !L || !U || !P)
    {
        printf("Error allocating memory for the matrices\n");
        exit(1);
    }
    for (int i = 0; i < dim; ++i)
    {
        A[i] = malloc(dim * sizeof(double));
        L[i] = malloc(dim * sizeof(double));
        U[i] = malloc(dim * sizeof(double));
        P[i] = malloc(dim * sizeof(double));
        for (int j = 0; j < dim; ++j)
        {
            A[i][j] = 100.0 * (double)(rand() / (double)RAND_MAX - 0.5); /* [-50.0, 50.0] */
            L[i][j] = 0.0;
            P[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }
    
    double **originalA = makeCopy(A, dim); 
    partPivLU(A, L, U, P, dim);
    int res = checkPALU(originalA, P, L, U, dim);
    printf("PA = LU: %s\n", res ? "true" : "false");
    for (int i = 0; i < dim; ++i)
    {
        free(originalA[i]);
    }
    free(originalA);
    deallocateMemory(A, L, U, P, dim);

    return 0;
}

double **makeCopy(double **a, int dim)
{
    /* makes a copy of an 2D array */
    double **c = malloc(dim * sizeof(double*));
    for (int i = 0; i < dim; ++i)
    {
        c[i] = malloc(dim * sizeof(double));
        memcpy(c[i], a[i], dim * sizeof(double));
    }
    return c;
}
void swapRows(double **a, int r1, int r2)
{
    /* swap two rows of a given matrix */
    double *temp = a[r1];
    a[r1] = a[r2];
    a[r2] = temp;
}

void showMatrix(double **m, int d)
{
    /* print a matrix in a human readable format */
    for (int i = 0; i < d; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            printf("%8.4f ", m[i][j]);
        }
        printf("\n");
    }
}

void partPivLU(double **a, double **l, double **u, double **p, int n)
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

    /* LU decomposition */
    for (int i = 0; i < n; ++i)
    {
        for (int j = i; j < n; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum += l[i][k] * u[k][j];
            }
            u[i][j] = a[i][j] - sum; /* Compute U */
        }
        for (int j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum += l[j][k] * u[k][i];
            }
            if (i == j)
            {
                l[i][j] = 1.0; /* main diagonal set to one */
            }
            else
            {
                if (fabs(u[i][i]) < TOL) /* Since the matrix is randomly generated we don't know if it's singular or not */
                {
                    fprintf(stderr, "Pivot near 0 encountered\n");
                    exit(EXIT_FAILURE);
                }
                l[j][i] = (a[j][i] - sum) / u[i][i]; /* compute L */
                
            }
        }
    }

    printf("Matrix A: \n");
    showMatrix(a, n);
    printf("L: \n");
    showMatrix(l, n);
    printf("U: \n");
    showMatrix(u, n);
    printf("Permutation matrix after LU: \n");
    showMatrix(p, n);
    
}

void deallocateMemory(double **a, double **l, double **u, double **p, int d)
{
    for (int i = 0; i < d; ++i)
    {
        free(a[i]);
        free(l[i]);
        free(u[i]);
        free(p[i]);
    }
    free(a);
    free(l);
    free(u);
    free(p);
}

void matrixMultiply(double **a, double **b, double **result, int n) 
{
    for (int i = 0; i < n; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            result[i][j] = 0.0;
        }
    }    
    for (int i = 0; i < n; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            for (int k = 0; k < n; ++k) 
            {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

bool checkPALU(double **a, double **p, double **l, double **u, int n)
{
    double **pa = malloc(n * sizeof(double*));
    double **lu = malloc(n * sizeof(double*));
    for (int i = 0; i < n; ++i) 
    {
        pa[i] = malloc(n * sizeof(double));
        lu[i] = malloc(n * sizeof(double));
    }

    matrixMultiply(p, a, pa, n); /* performs PA */
    matrixMultiply(l, u, lu, n); /* performs LU */
    /* check the maximum difference between PA and LU */
    double d = maxDiff(pa, lu, n);
    printf("Max. difference between PA and LU is: %.12f\n", d);
    /* we check if the absolute value of the distance between PA and LU is lesser than a certain tollerance */
    /* if this happens we can consider the two matrices equal */
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (fabs(pa[i][j] - lu[i][j]) > TOL)
            {
                for (int k = 0; k < n; ++k) 
                {
                    free(pa[k]);
                    free(lu[k]);
                }
                free(pa);
                free(lu);
                return false;
            }
        }
    }
    /* free the resources */
    for (int i = 0; i < n; i++) 
    {
        free(pa[i]);
        free(lu[i]);
    }
    free(pa);
    free(lu);

    return true;
}

double maxDiff(double **pa, double **lu, int n)
{
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            max_diff = fmax(max_diff, fabs(pa[i][j] - lu[i][j]));
        }
    }
    return max_diff;
}
