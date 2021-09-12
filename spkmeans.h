#ifndef FINAL_PROJECT_SPKMEANS_H
#define FINAL_PROJECT_SPKMEANS_H
/* This header contains macros, constants and functions used by the C API
 * as a public interface */

/*******************************************************************************
********************************** Imports *************************************
*******************************************************************************/
#include <stdio.h>

/*******************************************************************************
********************************* Constants ************************************
*******************************************************************************/
#define MAX_KMEANS_ITER 300
#define SIZE_OF_VOID_2PTR sizeof(void **)

/*******************************************************************************
********************************* Macros ***************************************
*******************************************************************************/
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

/*******************************************************************************
********************************* Struct ***************************************
*******************************************************************************/
typedef enum {
    FOREACH_GOAL(GENERATE_ENUM)
    NUM_OF_GOALS
} GOAL;

/*******************************************************************************
******************************** Globals ***************************************
*******************************************************************************/
/* Global memory variables */
void **headOfMemList;
void *freeUsedMem;

/*******************************************************************************
**************************** Functions Declaration *****************************
*******************************************************************************/

/**
 * The function runs spk algorithm steps and stop at the desired goal.
 * The function returns the relevant matrix depended on the GOAL.
 * The function also calculates and assign K if not provided.
 * @param datapointsArray Original data to adjust
 * @param goal Desired goal
 * @param k number of clusters (for kmeans)
 * @param dimension datapoints' number of features
 * @param numOfDatapoints number of datapoints
 * @return Matrix: 'spk' - T, 'wam' - W, 'ddg' - D, 'lnorm' - Lnorm, NULL on failure.
 */
double **dataAdjustmentMatrices(double **datapointsArray, GOAL goal, int *k,
                                int dimension, int numOfDatapoints);

/**
 * This function runs the main KMeans clustering algorithm.
 * @param vectorsArray Vectors array to be clustered
 * @param numOfVectors Number of vectors
 * @param dimension Vectors' dimension
 * @param k Number of desired clusters
 * @param firstCentralIndexes First vectors indexes to be the initial clusters'
 *          centroids (for kmeans++ only), NULL for kmeans
 * @param maxIter Maximum number of kmeans iterations till convergence
 * @return Final clusters centroids and vector to cluster labeling as one matrix
 */
double **kMeans(double **vectorsArray, int numOfVectors, int dimension, int k,
                const int *firstCentralIndexes, int maxIter);

/**
 * This function performs Jacobi's diagonal method on a symmetric matrix.
 * @param matrix A symmetric matrix
 * @param n matrix's dimension
 * @return Transposed eigenvectors matrix (V^T), NULL on failure
 */
double **jacobiAlgorithm(double **matrix, int n);

/**
 * The function allocates memory for any dynamic memory needed.
 * If use new memory space, add it to the list of memory blocks and update the pointers.
 * @param effectiveUsedMem Block of allocated memory - without list's pointers
 * @param size Size of block in bytes
 * @return Pointer to head of effective block of memory, NULL on failure
 */
void *myAlloc(void *effectiveUsedMem, size_t size);

/**
 * The function builds a 2 dimension array (matrix) using "myAlloc" function.
 * @param rows Matrxi's number of rows
 * @param cols Matrxi's number of columns
 * @param basicSize sizeof(type) in bytes
 * @param basicPtrSize sizeof(type *) in bytes
 * @param recycleMemBlock Free used memory block pointer, NULL for new allocation
 * @return Pointer to a matrix array
 */
void **alloc2DArray(int rows, int cols, size_t basicSize, size_t basicPtrSize,
                    void *recycleMemBlock);

/**
 * This function free unnecessary memory and keep the order of the memory list.
 * @param effectiveBlockMem Block of allocated memory - without list's pointers
 */
void myFree(void *effectiveBlockMem);

/**
 * This function free all memory allocated at runtime.
 * The function uses the memory list to support unexpected exit of program/errors.
 */
void freeAllMemory();

/**
 * This function convert String to enum representation.
 * @param str Enum as string
 * @return GOAL enum, special value NUM_OF_GOALS on failure
 */
GOAL str2enum(char *str);

#endif /*FINAL_PROJECT_SPKMEANS_H */
