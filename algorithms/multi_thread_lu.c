/*
 *  Multi-threaded LU decomposition - Doolitle's algorithm implementation
 *   
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
#include <pthread.h>
#include <unistd.h>

#define NUM_THREAD 3 /* number of thread */

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
    double **a; /* A */
    double **l; /* L */ 
    double **u; /* U */
    int n; /* dimension */
    int startingRow; /* the row the thread starts from */
    int endingRow; /* the row the thread finishes its job */
    int *currentRow; /* keeping track of the current row */
    pthread_mutex_t *mu; /* mutex */
    pthread_cond_t *cv; /* cond. variable */
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
            for (int k = 0; k < i; ++k)
            {
                s += lMatrix[i][k] * uMatrix[k][j];
            }
            uMatrix[i][j] = aMatrix[i][j] - s; /* compute U */
        }

        /* critical section: a thread can calculate a row in a wrong order */
        pthread_mutex_lock(t->mu);
        while ((*t->currentRow) != i)
        {
            pthread_cond_wait(t->cv, t->mu);
        }
        pthread_mutex_unlock(t->mu);
        printf("[%d] Thread -> building L\n", gettid());
        for (int j = i; j < d; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < i; ++k)
            {
                s += lMatrix[j][k] * uMatrix[k][i];
            }
            if (uMatrix[i][i] == 0)
            {
                fprintf(stderr, "Division by zero. Exit...\n");
                return NULL;
            }
            if (i == j)
            {
                lMatrix[i][j] = 1.0; /* set the main diagonal to one */
            }
            else 
            {
                lMatrix[j][i] = (aMatrix[j][i] - s) / uMatrix[i][i]; /* compute L */
            }
        } 

        pthread_mutex_lock(t->mu);
        (*(t->currentRow))++;
        pthread_cond_broadcast(t->cv);
        pthread_mutex_unlock(t->mu);
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
    /* free all the memory allocated for the matrices */
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
    
    srand(time(NULL)); /* generate a random seed */
    printf("Insert a dimension: ");
    int dim;
    if (scanf("%d", &dim) != 1)
    {
        fprintf(stderr, "Error scanf\n");
        exit(EXIT_FAILURE);
    }

    /* creation of the three matrices */
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
        /* allocate memory for each rows */
        A[i] = malloc(dim * sizeof(double));
        L[i] = malloc(dim * sizeof(double));
        U[i] = malloc(dim * sizeof(double));
        for (int j = 0; j < dim; ++j)
        {
            A[i][j] = 20000.0 * (double)(rand() / (double)RAND_MAX - 0.5); /* [-10000.0, 10000.0] */
            L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
    }

    pthread_t threads[NUM_THREAD]; /* array of threads */
    threadArg threadData[NUM_THREAD]; /* array of struct */
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    /* assigning thr rows to the threads */
    int threadRow = dim / NUM_THREAD;
    int row = 0;
    
    /* creation of the thread's generics */
    for (int i = 0; i < NUM_THREAD; ++i)
    {
        threadData[i].a = A;
        threadData[i].l = L;
        threadData[i].u = U;
        threadData[i].n = dim;
        threadData[i].startingRow = i * threadRow;
        /* if threadRow is not divisible by NUM_THREAD we assign the last row to the last thread */
        threadData[i].endingRow = (i == NUM_THREAD - 1) ? dim : (i + 1) * threadRow;
        threadData[i].mu = &mutex;
        threadData[i].cv = &cond;
        threadData[i].currentRow = &row;
    }
    for (int i = 0; i < NUM_THREAD; ++i)
    {
        /* check if the creation of the threads goes well */
        if (pthread_create(&threads[i], NULL, parallelLU, (void*)&threadData[i]) != 0)
        {
            perror("Error creating threads\n");
            exit(EXIT_FAILURE);
        }
    }
    
    for (int i = 0; i < NUM_THREAD; ++i)
    {
        /* check if the join ends up well */
        if (pthread_join(threads[i], NULL) != 0)
        {
            perror("Error joining threads\n");
            exit(EXIT_FAILURE);
        }
    }

    /* deallocate the memory and destroy mutexes and cond. variables */
    printf("Starting matrix:\n");
    showMatrix(A, dim);
    printf("L:\n");
    showMatrix(L, dim);
    printf("U:\n");
    showMatrix(U, dim);
    deallocateMemory(A, L, U, dim);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    return 0;
}


