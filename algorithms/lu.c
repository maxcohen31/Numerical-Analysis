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
#include <pthread.h>
#include <unistd.h>

#define NUM_THREAD 3

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

/* thread structure */
typedef struct 
{
    double **a;
    double **l;
    double **u;
    int n;
    int startingRow;
    int endingRow;
} threadArg;

void *parallelLU(void *args)
{
    /* performing the LU decomposition without a permutation matrix */
    threadArg *t = (threadArg*)args;
    double **aMatrix = t->a;
    double **lMatrix = t->l;
    double **uMatrix = t->u;
    int start = t->startingRow;
    int end = t->endingRow;
    int d = t->n;

    printf("LU decomposition started!\n");
    for (int i = start; i < end; ++i)
    {
        printf("[%d] Thread -> building U\n", gettid());
        for (int j = i; j < d; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < d; ++k)
            {
                s += lMatrix[i][k] * uMatrix[k][j];
            }
            uMatrix[i][j] = aMatrix[i][j] - s;
        }
        /* lMatrix[i][i] = 1.0; // main diagonal of L set to 1 */
        printf("[%d] Thread -> building L\n", gettid());
        for (int j = i; j < d; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < d; ++k)
            {
                s += lMatrix[j][k] * uMatrix[k][i];
            }
            lMatrix[j][i] = (aMatrix[i][j] - s) / uMatrix[i][i];
        } 
    }

    return NULL;
}

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
            A[i][j] = rand() % RAND_MAX;
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = 0.0;
        }
    }

    pthread_t threads[NUM_THREAD]; /* array of threads */
    threadArg threadData[NUM_THREAD];
    /* assigning thr rows to the threads */
    int threadRow = dim / NUM_THREAD;
    
    for (int i = 0; i < NUM_THREAD; ++i)
    {
        threadData[i].a = A;
        threadData[i].l = L;
        threadData[i].u = U;
        threadData[i].n = dim;
        threadData[i].startingRow = i * threadRow;
        /* if threadRow is not divisible by NUM_THREAD */
        threadData[i].endingRow = (i == NUM_THREAD - 1) ? dim : (i + 1) * threadRow;
    }
    // TODO: create e join. Possibile implementazione di P


    return 0;
}


