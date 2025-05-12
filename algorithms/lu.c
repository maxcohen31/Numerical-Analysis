/*
 *  LU decomposition - Doolitle's algorithm implementation with partial pivoting
 *
 *  Spring 2025 - Author: Abbate Emanuele
 *
 *  References:  https://www.cs.gordon.edu/courses/mat342/handouts/gauss.pdf 
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <string.h>


// Sarrus's rule for a 3x3 matrix
// int determinant(int **m)
// {
//     int det = 0;
//     for (int i = 0; i < dim; ++i)
//     {
//         det += (m[0][i] * m[1][(i+1)%dim] * m[2][(i+2)%dim]) - (m[2][i] * m[1][(i+1)%dim] * m[0][(i+2)%dim]); 
//     }
//     return det;
// }


double **makeCopy(double **m, int d)
{
    // make a copy of a 2D array
    double **matrixCopy = malloc(d * sizeof(double*));
    for (int i = 0; i < d; ++i)
    {
        matrixCopy[i] = malloc(d * sizeof(double));
        memcpy(matrixCopy[i], m[i], d * sizeof(double));
    }
    return matrixCopy;
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

void deallocateMemory(double **a, int d)
{
    for (int i = 0; i < d; ++i)
    {
        free(a[i]);
    }
    free(a);
}

void swapRows(double **a, int r1, int r2)
{
    double *temp = a[r1];
    a[r1] = a[r2];
    a[r2] = temp;
}

void luDecomposition(double **a, int d)
{
    double **l = malloc(d * sizeof(double*));
    double **u = malloc(d * sizeof(double*));
    double **p = malloc(d * sizeof(double*));
    if (!l || !u || !p)
    {
        fprintf(stderr, "Error allocating memory\n");
        exit(EXIT_FAILURE);
    }
    
    // U = A 
    // L = I
    // P = I
    double **copy = makeCopy(a, d);

    // initialize P as the identity matrix 
    for (int i = 0; i < d; ++i)
    {
        p[i] = malloc(d * sizeof(double));
        for (int j = 0; j < d; ++j)
        {
            p[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    // initialize L as the identity matrix and U as zero 
    for (int i = 0; i < d; ++i)
    {
        l[i] = malloc(d * sizeof(double));
        u[i] = malloc(d * sizeof(double));
        for (int j = 0; j < d; ++j)
        {
            l[i][j] = (i == j) ? 1.0 : 0.0;
            u[i][j] = 0.0;
        }
    }

    // partial pivoting subroutine
    for (int k = 0; k < d; ++k)
    {
        // find the row with the max absolute value in the column k 
        int pivotRow = k; // let's say the max row is k 
        double pivot = fabs(copy[k][k]); // we suppose the pivot is the matrix diagonal element
        for (int i = k; i < d; ++i)
        {
            if (fabs(copy[i][k]) > pivot)
            {
                pivot = fabs(copy[i][k]); // take the absoulte value of the max element
                pivotRow = i;
            }
        }
        // swap rows if the index of the column k and the row index are different
        // we make this swap since we want the max absolute element value in the main diagonal
        if (pivotRow != k)
        {
            swapRows(copy, k, pivotRow);
            swapRows(p, k, pivotRow);
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

    // making the U matrix - backward substitution
     for (int i = 0; i < d; ++i)
     {
         for (int j = i; j < d; ++j)
         {
             double s = 0.0;
             for (int k = 0; k < i; ++k)
             {
                 s += l[i][k] * u[k][j];
             }
             // compute U(i, j)
             u[i][j] = copy[i][j] - s;
         }

         // making the L matrix - forward substitution
         for (int j = i; j < d; ++j)
         {
             // ones on the main diagonal
             if (i == j)
             {
                 l[i][j] = 1.0;
             }
             else
             {
                 double s = 0.0;
                 for (int t = 0; t < i; ++t)
                 {
                     s += l[j][t] * u[t][i];
                 }
                 // compute L(i,j)
                 l[j][i] = (copy[j][i] - s) / u[i][i];
             }
         }
     }

    printf("L is equal to:\n");
    showMatrix(l, d);
    printf("\n");
    printf("U is equal to: \n");
    showMatrix(u, d); 
    printf("P is equal to: \n");
    showMatrix(p, d);
    deallocateMemory(l, d);
    deallocateMemory(u, d);
    deallocateMemory(p, d);
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

    double **matrix = malloc(dim * sizeof(double*));
    if (matrix == NULL)
    {
        printf("Error allocating memory for the matrix\n");
        exit(1);
    }

    for (int i = 0; i < dim; ++i)
    {
        matrix[i] = malloc(dim * sizeof(double));
        if (matrix[i] == NULL)
        {
            printf("Error filling the matrix\n");
            exit(2);
        }
        for (int j = 0; j < dim; ++j)
        {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }

    printf("Matrix: \n");
    showMatrix(matrix, dim);
    luDecomposition(matrix, dim);
    deallocateMemory(matrix, dim);
    return 0;
}


