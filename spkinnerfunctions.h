#ifndef FINAL_PROJECT_SPKINNERFUNCTIONS_H
#define FINAL_PROJECT_SPKINNERFUNCTIONS_H
/* This header contains macros, constants and functions used by the C spk inner
 * implementation. Other public functions are at spkmeansmodule.h*/

/*******************************************************************************
********************************** Imports *************************************
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>

/*******************************************************************************
********************************* Constants ************************************
*******************************************************************************/
#define EPSILON 1.0E-15
#define MAX_JACOBI_ITER 100
#define MAX_FEATURES 10
#define COMMA_CHAR ','
#define REQUIRED_NUM_OF_ARGUMENTS 4
#define K_ARGUMENT 1
#define GOAL_ARGUMENT 2
#define MAX_DATAPOINTS 50
#define END_OF_STRING '\0'
#define PRINT_FORMAT "%.4f"
#define ERROR_MSG "An Error Has Occured\n"
#define INVALID_INPUT_MSG "Invalid Input!\n"

/*******************************************************************************
********************************* Macros ***************************************
*******************************************************************************/
/* x^2 macro */
#define SQ(x) ((x)*(x))

/* Custom logical assert macro - print error, free memory and exit program */
#define MyAssert(exp)       \
if (!(exp)) {               \
fprintf(stderr, ERROR_MSG); \
freeAllMemory();            \
exit(EXIT_FAILURE);         \
}

/* Enum macros */
#define GENERATE_STRING(STRING) #STRING,
static const char *GOAL_STRING[] = {FOREACH_GOAL(GENERATE_STRING)};

/*******************************************************************************
********************************* Struct ***************************************
*******************************************************************************/
/* Cluster type for the kmeans algorithm */
typedef struct {
    double *prevCentroid;
    double *currCentroid;
    int counter; /* Number of vectors (datapoints) in cluster */
} Cluster;

/* Eigenvalue type for the jacobi algorithm */
typedef struct {
    double value;
    int vector;
} Eigenvalue;

/*******************************************************************************
**************************** Functions Declaration *****************************
*******************************************************************************/

/************************ Spectral Clustering Functions ***********************/

/**
 * This function form the Weighted Adjacency Matrix out of vectors list.
 * @param vectorsArray Vectors as a matrix
 * @param numOfVectors number of vectors
 * @param dimension vectors' dimension
 * @return Weighted Adjacency Matrix as matrix (2D double array), NULL on failure
 */
double **weightedMatrix(double **vectorsArray, int numOfVectors, int dimension);

/**
 * This function form the Diagonal Degree Matrix of Weighted Adjacency Matrix.
 * @param wMatrix Weighted Adjacency Matrix
 * @param n W's dimension
 * @return Diagonal matrix, NULL on failure
 */
double **dMatrix(double **wMatrix, int n);

/**
 * This function form the Normalized Graph Laplacian matrix in a given D + W matrix.
 * Overwrite W matrix to be Lnorm.
 * @param wMatrix Weighted Adjacency Matrix
 * @param dMatrix Diagonal Degree Matrix
 * @param numOfVectors W/D's dimension
 * @return Lnorm matrix
 */
double **laplacian(double **wMatrix, double **dMatrix, int numOfVectors);

/**
 * This function form T matrix from Lnorm eigenvalues, eigenvectors and k.
 * @param eigenvalues Lnorm's eigenvalues sorted
 * @param eigenvectorsMat Lnorm's eigenvectors sorted
 * @param n Lnorm's dimension
 * @param k Number of clusters for the kmeans
 * @return T matrix, NULL on failure
 */
double **initTMatrix(Eigenvalue *eigenvalues, double **eigenvectorsMat, int n, int k);

/**
 * This function calculate the optimum k using Eigengap Heuristic method.
 * @param eigenvalues Lnorm's eigenvalues sorted
 * @param n Lnorm's dimension > 1
 * @return Optimum k (number of clusters) for the kmeans
 */
int eigengapHeuristicKCalc(Eigenvalue *eigenvalues, int n);

/****************************** KMeans Functions ******************************/

/**
 * This function initialize the clusters array.
 * @param vectorsArray Vectors to be clustered
 * @param k Number of desired clusters
 * @param dimension vectors' dimension
 * @param firstCentralIndexes First vectors indexes to be the initial clusters'
 *          centroids (for kmeans++ only), NULL for kmeans
 * @return Initialized Clusters array
 */
Cluster *initClusters(double **vectorsArray, int k, int dimension,
                      const int *firstCentralIndexes);

/**
 * This function assign the closest cluster for each vector.
 * The function also cont the number of vectors for each cluster
 *      and sum the vectors components for later use.
 * @param vectorsArray Vectors to be clustered
 * @param clustersArray Clusters array
 * @param vecToClusterLabeling Vector to cluster labeling array
 * @param k Number of clusters
 * @param numOfVectors Number of vectors
 * @param dimension Vectors' dimension
 */
void assignVectorsToClusters(double **vectorsArray, Cluster *clustersArray,
                             double *vecToClusterLabeling, int k,
                             int numOfVectors, int dimension);

/**
 * This function finds vector's closest cluster (in terms of euclidean norm).
 * @param vec Vector to be clustered
 * @param clustersArray Clusters array
 * @param k Number of clusters
 * @param dimension Vector's dimension
 * @return Vector's closest cluster index
 */
int findMyCluster(double *vec, Cluster *clustersArray, int k, int dimension);

/**
 * This function calculates the squared euclidean norm between two vectors.
 * @param vec1 First vector
 * @param vec2 Second vector
 * @param dimension vectors' dimension
 * @return The squared euclidean norm between the two vectors
 */
double vectorsSqNorm(const double *vec1, const double *vec2, int dimension);

/**
 * This function recalculates clusters centroids after one kmeans iteration.
 * @param clustersArray Clusters array
 * @param k Number of clusters
 * @param dimension Vectors' dimension
 * @return Number of clusters' components changed during last iteration
 */
int recalcCentroids(Cluster *clustersArray, int k, int dimension);

/**
 * This function organize clusters array for the next iteration:
 *      Reset counter and current centroid (to be zero vector)
 *      Update previous centroids to be current centroids
 * @param clustersArray Clusters array
 * @param k Number of clusters
 * @param dimension Vectors' dimension
 */
void initCurrCentroidAndCounter(Cluster *clustersArray, int k, int dimension);

/**
 * This function organize KMeans result into a matrix:
 *      First k rows - Clusters centroids
 *      Last row (Could be from different length) vectors to clusters labeling
 * @param clustersArray Clusters array
 * @param vecToClusterLabeling Vector to cluster labeling array
 * @param k
 * @param dimension
 * @return
 */
double **buildFinalCentroidsMat(Cluster *clustersArray, double *vecToClusterLabeling,
                                int k, int dimension);

/******************************** Jacobi Functions ****************************/

/**
 * This function performs a single jacobi rotation.
 * @param a A symmetric matrix to perform the rotation on
 * @param v The cumulative eigenvectors matrix
 * @param n a's dimension
 * @param i Pivot row index
 * @param j Pivot column index
 * @return Off-diag Frobenius norm delta
 */
double jacobiRotate(double **a, double **v, int n, int i, int j);

/** This function chooses the pivot index for the jacobi rotation
 *      - the max abs off diagonal element > 0.
 * If the matrix is already diagonal - assign pivotRow with special value EOF
 * @param matrix Symmetric matrix for the jacobi rotation
 * @param n matrix's dimension
 * @param pivotRow To assign pivot's row index
 * @param pivotCol to assing pivot's column index
 */
void pivotIndex(double **matrix, int n, int *pivotRow, int *pivotCol);

/**
 * Build an n * n identity matrix.
 * @param n matrix's dimension
 * @return Identity matrix, NULL on failure
 */
double **initIdentityMatrix(int n);

/**
 * Sorting eigenvalues using qsort and comparator (makes it stable).
 * @param a Diagonal matrix (after jacobi's algorithm)
 * @param n a's dimension
 * @return Sorted eigenvalues array
 */
Eigenvalue *sortEigenvalues(double **a, int n);

/**
 * Comparator function for the eigenvalues qsort.
 * @param p1 First element
 * @param p2 Second element
 * @return result > 0 if p1 < p2, result < 0 if p1 > p2
 */
int cmpEigenvalues (const void *p1, const void *p2);

/* Print functions */
/**
 * This function print matrix in csv format.
 * @param matrix Matrix to be printed
 * @param rows Number of matrix's rows
 * @param cols Number of matrix's columns
 */
void printMatrix(double **matrix, int rows, int cols);

/**
 * The function prints the jacobi result in csv format:
 *      first line - eigenvalues
 *      Remain lines - eigenvectors as rows
 * @param a The diagonal matrix - eigenvalues
 * @param v The eigenvectors matrix
 * @param n a/v's dimension
 */
void printJacobi(double **a, double **v, int n);

/*************************** Auxiliary Functions ******************************/

/**
 * This function read cmd-line arguments, validate and assign them the matching variables.
 * @param argc Number of cmd-line arguments
 * @param argv cmd-line arguments as array of strings
 * @param k K to be assigned
 * @param goal Goal to be assigned
 * @param filenamePtr filename ptr to be assigned
 */
void validateAndAssignInput(int argc, char **argv, int *k, GOAL *goal, char **filenamePtr);

/**
 * The function read from csv format file (extension .txt/.csv) into matrix.
 * @param rows To be assigned with matrix's number of rows
 * @param cols To be assigned with matrix's number of columns
 * @param fileName Filename of .csv/.txt file in csv format
 * @param goal SPK desired goal
 * @return File content as a matrix
 */
double **readDataFromFile(int *rows, int *cols, char *fileName, GOAL goal);

/**
 * This function calculates and assign the Data's number of features,
 *      while reading the first line of the file.
 * @param dimension To be assigned with number of features
 * @param file The opened file pointer
 * @param firstLine To be assigned with the first line in the file
 */
void calcDim(int *dimension, FILE *file, double *firstLine);

#endif /* FINAL_PROJECT_SPKINNERFUNCTIONS_H */