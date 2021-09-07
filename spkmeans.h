#ifndef FINAL_PROJECT_SPKMEANS_H
#define FINAL_PROJECT_SPKMEANS_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>

#define MAX_KMEANS_ITER 300
#define SIZE_OF_VOID_2PTR sizeof(void **)

/* TODO ifndef */

/* Avoid -0.0000 representation */
#define NegZero(value) (value) < 0.0 && (value) > -0.00005 ? -(value) : (value)
/* Free macros */
#define MyFree(block) myFree(block); block = NULL
#define MyRecycleMatFree(block) freeUsedMem = *block; block = NULL

/* Enum macros */
#define FOREACH_GOAL(GOAL) \
GOAL(jacobi) \
GOAL(wam)  \
GOAL(ddg)   \
GOAL(lnorm)  \
GOAL(spk)
#define GENERATE_ENUM(ENUM) ENUM,
#define GENERATE_STRING(STRING) #STRING,

typedef enum {
    FOREACH_GOAL(GENERATE_ENUM)
    NUM_OF_GOALS
} GOAL;
static const char *GOAL_STRING[] = {FOREACH_GOAL(GENERATE_STRING)};

/* Global memory variables */
void **headOfMemList;
void *freeUsedMem;

/* Public function declaration */
double **dataAdjustmentMatrices(double **datapointsArray, GOAL goal, int *k, int dimension, int numOfDatapoints);
double **kMeans(double **vectorsArray, int numOfVectors, int dimension, int k, const int *firstCentralIndexes, int maxIter);
double **jacobiAlgorithm(double **matrix, int n);
GOAL str2enum(char *str);
void myFree(void *effectiveBlockMem);
void *myAlloc(void *effectiveUsedMem, size_t size);
void **alloc2DArray(int rows, int cols, size_t basicSize, size_t basicPtrSize, void *recycleMemBlock);
void freeAllMemory(); /* Free the allocated memory */
#endif //FINAL_PROJECT_SPKMEANS_H
