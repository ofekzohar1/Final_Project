#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>

#define SQ(x) ((x)*(x))
#define EPSILON 1.0E-15
#define MAX_JACOBI_ITER 100
#define MAX_FEATURES 10
#define COMMA_CHAR ','
#define REQUIRED_NUM_OF_ARGUMENTS 4
#define K_ARGUMENT 1
#define GOAL_ARGUMENT 2
#define MAX_KMEANS_ITER 300
#define MAX_DATAPOINTS 50
#define END_OF_STRING '\0'
#define PRINT_FORMAT "%.4f"
#define ERROR_MSG "An Error Has Occurred\n"
#define INVALID_INPUT_MSG "Invalid Input!\n"
#define SIZE_OF_VOID_2PTR sizeof(void **)

#define MyAssert(exp) \
if (!(exp)) {      \
fprintf(stderr, ERROR_MSG); \
freeAllMemory();                      \
assert (exp);          \
exit(EXIT_FAILURE);     \
}

/* Avoid -0.0000 representation */
#define NegZero(value) (value) < 0.0 && (value) > -0.00005 ? -(value) : (value)

#define MyFree(block) myFree(block); block = NULL

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

typedef struct {
    double *prevCentroid;
    double *currCentroid;
    int counter; /* Number of vectors (datapoints) in cluster */
} Cluster;

typedef struct {
    double value;
    int vector;
} Eigenvalue;

double **fitClusteringToOriginData(int k, int dimension, int numOfVectors, double **vectorsArray, double **originData, Cluster *clustersArray);
Cluster *kMeans(int k, int maxIter, int dimension, int numOfVectors, double **vectorsArray, int *firstCentralIndexes);
Cluster *initClusters(double **vectorsArray, int k, int dimension, int *firstCentralIndexes);
double vectorsNorm(const double *vec1, const double *vec2, const int *dimension); /* Calculate the norm between 2 vectors */
int findMyCluster(double *vec, Cluster *clustersArray, const int *k, const int *dimension); /* Return the vector's closest cluster (in terms of norm) */
void assignVectorsToClusters(double **vectorsArray, Cluster *clustersArray, const int *k, const int *numOfVectors, const int *dimension); /* For any vector assign to his closest cluster */
int recalcCentroids(Cluster *clustersArray, const int *k, const int *dimension); /* Recalculate clusters' centroids, return number of changes */
void initCurrCentroidAndCounter(Cluster *clustersArray, const int *k, const int *dimension); /* Set curr centroid to prev centroid and reset the counter */
void freeAllMemory(); /* Free the allocated memory */
void jacobiAlgorithm(double **matrix, double **eigenvectorsMat, int n);
void initIdentityMatrix(double **matrix, int n);
int eigengapHeuristicKCalc(Eigenvalue *eigenvalues, int n);
void **alloc2DArray(int rows, int cols, size_t basicSize, size_t basicPtrSize, void *freeUsedMem);
void *myAlloc(void *usedMem, size_t size);
void printFinalCentroids(Cluster *clustersArray, int k, int dimension);
void printMatrix(double **matrix, int rows, int cols);
void validateAndAssignInput(int argc, char **argv, int *k, GOAL *goal, char **filenamePtr);
double **readDataFromFile(int *rows, int *cols, char *fileName, GOAL goal);
double **weightedMatrix(double **wMatrix, double **vectorsArray, int numOfVectors, int dimension);
double *dMatrix(double **wMatrix, int dim);
double **laplacian(double **wMatrix, double *dMatrix, int numOfVectors);
void calcDim(int *dimension, FILE *file, double *firstLine);
double **initTMatrix(Eigenvalue *eigenvalues, double **eigenvectorsMat, void *freeUsedMem, int n, int k);
Eigenvalue *sortEigenvalues(double **a, int n);
int cmpEigenvalues (const void *p1, const void *p2);
void pivotIndex(double **matrix, int n, int *pivotRow, int *pivotCol);
void printJacobi(double **a, double **v, int n);
void printDiagMat(double *matrix, int n);
GOAL str2enum(char *str);
void myFree(void *blockMem);

void **headOfMemList;

/**********************************
*********** Main ******************
**********************************/

int main(int argc, char *argv[]) {
    int k, dimension, numOfDatapoints;
    GOAL goal;
    char *filename;
    double **datapointsArray, **tMat, **wMat, **lnormMat, **eigenvectorsMat, *ddgMat, **freeUsedMem;
    Eigenvalue *eigenvalues;
    Cluster *clustersArray;
    headOfMemList = NULL;
    /* int firstIndex[] = {44,46,68,64}; */

    validateAndAssignInput(argc, argv, &k, &goal, &filename);
    datapointsArray = readDataFromFile(&numOfDatapoints, &dimension, filename, goal);
    if (goal == spk && k >= numOfDatapoints) {
        printf(INVALID_INPUT_MSG);
        goto end;
    }

    freeUsedMem = (double **) alloc2DArray(numOfDatapoints, numOfDatapoints, sizeof(double), sizeof(double *), NULL);
    if (goal == jacobi) {
        eigenvectorsMat = freeUsedMem;
        jacobiAlgorithm(datapointsArray, eigenvectorsMat, numOfDatapoints);
        printJacobi(datapointsArray, eigenvectorsMat, numOfDatapoints);
        goto end;
    }

    wMat = weightedMatrix(freeUsedMem, datapointsArray, numOfDatapoints, dimension);
    if (goal==wam){
        printMatrix(wMat, numOfDatapoints, numOfDatapoints);
        goto end;
    }

    ddgMat = dMatrix(wMat, numOfDatapoints);
    if (goal==ddg){
        printDiagMat(ddgMat, numOfDatapoints);
        MyFree(ddgMat);
        goto end;
    }

    lnormMat = laplacian(wMat, ddgMat, numOfDatapoints);
    if(goal==lnorm){
        printMatrix(lnormMat, numOfDatapoints, numOfDatapoints);
        goto end;
    }

    eigenvectorsMat = (double **) alloc2DArray(numOfDatapoints, numOfDatapoints, sizeof(double), sizeof(double *), NULL);
    jacobiAlgorithm(lnormMat, eigenvectorsMat, numOfDatapoints);
    eigenvalues = sortEigenvalues(lnormMat, numOfDatapoints);
    freeUsedMem = lnormMat;

    if (k == 0)
        k = eigengapHeuristicKCalc(eigenvalues, numOfDatapoints);
    tMat = initTMatrix(eigenvalues, eigenvectorsMat, *freeUsedMem, numOfDatapoints, k);

    clustersArray = kMeans(k, MAX_KMEANS_ITER, k, numOfDatapoints, tMat, NULL);
    printFinalCentroids(clustersArray, k, k);
    freeUsedMem = tMat;

    end:
    freeAllMemory();
    return 0;
}

void printJacobi(double **a, double **v, int n) {
    int i;
    double value;

    for (i = 0; i < n; ++i) {
        if (i != 0)
            printf("%c", COMMA_CHAR);
        value = a[i][i];
        value = NegZero(value);
        printf(PRINT_FORMAT, value);
    }
    printf("\n");
    printMatrix(v, n, n);
}

/**********************************
******** Spectral Clustering ******
**********************************/

double **dataClustering(int k, int maxIter, int dimension, int numOfVectors, double **vectorsArray, double **originData, int *firstCentralIndexes) {
    Cluster *clustersArray;
    double **originDataCentroids;

    clustersArray = kMeans(k, maxIter, k, numOfVectors, vectorsArray, firstCentralIndexes);
    originDataCentroids = fitClusteringToOriginData(k, dimension, numOfVectors, vectorsArray, originData, clustersArray);

    freeAllMemory(vectorsArray, originData, clustersArray, k);
    return originDataCentroids;
}

double **fitClusteringToOriginData(int k, int dimension, int numOfVectors, double **vectorsArray, double **originData, Cluster *clustersArray) {
    int i, j, clusterIndex, clusterSize;
    double **originDataCentroids;
    double *dataPoint;

    originDataCentroids = (double **) alloc2DArray(k, dimension, sizeof(double), sizeof(double *), NULL);
    for (i = 0; i < k; ++i) {
        for (j = 0; j < dimension; ++j) {
            originDataCentroids[i][j] = 0.0;
        }
    }

    for (i = 0; i < numOfVectors; ++i) {
        /* Set vectors final cluster to the corresponding original data */
        dataPoint = originData[i];
        clusterIndex = (int) vectorsArray[i][dimension];
        dataPoint[dimension] = clusterIndex;
        for (j = 0; j < dimension; ++j) {
            originDataCentroids[clusterIndex][j] += dataPoint[j];
        }
    }

    for (i = 0; i < k; ++i) {
        clusterSize = clustersArray[i].counter;
        for (j = 0; j < dimension; ++j) {
            originDataCentroids[i][j] /= clusterSize; /* Mean calc */
        }
    }
    return originDataCentroids;
}

void **alloc2DArray(int rows, int cols, size_t basicSize, size_t basicPtrSize, void *freeUsedMem) {
    int i;
    void *blockMem, **matrix;
    /* Reallocate block of memory */
    blockMem = myAlloc(freeUsedMem, rows * cols * basicSize + rows * basicPtrSize);
    matrix = (void **) ((char *)blockMem + rows * cols * basicSize);

    for (i = 0; i < rows; ++i) {
        /* Set matrix to point to 2nd dimension array */
        *((void **)((char *)matrix + i * basicPtrSize)) = (void *) (((char *) blockMem) + i * cols * basicSize);
    }
    return matrix;
}

void *myAlloc(void *usedMem, size_t size) {
    void *effectiveUsedMem = usedMem != NULL ? (void *)((char *)usedMem - SIZE_OF_VOID_2PTR * 2) : NULL;
    void *blockMem, **blockMemPlusPtr = (void **)realloc(effectiveUsedMem, size + SIZE_OF_VOID_2PTR * 2);
    MyAssert(blockMemPlusPtr != NULL);
    blockMem = (void *)((char *)blockMemPlusPtr + SIZE_OF_VOID_2PTR * 2);
    if (usedMem == NULL) { /* New Allocation */
        /* Set ptr to the prev/next dynamic allocated memory block */
        blockMemPlusPtr[0] = NULL;
        if (headOfMemList != NULL) {
            blockMemPlusPtr[1] = headOfMemList;
            ((void **)headOfMemList)[0] = blockMemPlusPtr;
        } else
            blockMemPlusPtr[1] = NULL;
        headOfMemList = blockMemPlusPtr;
    } else {
       if (effectiveUsedMem != blockMemPlusPtr) {
           if (blockMemPlusPtr[0] != NULL)
               ((void **)blockMemPlusPtr[0])[1] = blockMemPlusPtr;
           else
               headOfMemList = blockMemPlusPtr;
           if(blockMemPlusPtr[1] != NULL)
               ((void **)blockMemPlusPtr[1])[0] = blockMemPlusPtr;
       }
    }
    return blockMem;
}

void myFree(void *blockMem) {
    void **effectiveBlockMem;
    if (blockMem == NULL)
        return;
    effectiveBlockMem = (void **)((char *)blockMem - SIZE_OF_VOID_2PTR * 2);
    if(effectiveBlockMem[0] != NULL) {
        ((void **)effectiveBlockMem[0])[1] = effectiveBlockMem[1]; /* Set prev's next to current next */
    } else {
        headOfMemList = effectiveBlockMem[1];
    }
    if(effectiveBlockMem[1] != NULL) {
        ((void **)effectiveBlockMem[1])[0] = effectiveBlockMem[0]; /* Set next's prev to current prev */
    }
    free(effectiveBlockMem);
}

void freeAllMemory() {
    void **cuurBlock = headOfMemList, **nextBlock;

    while (cuurBlock != NULL) {
        nextBlock = cuurBlock[1];
        free(cuurBlock);
        cuurBlock = nextBlock;
    }
    headOfMemList = NULL;
}

/*void freeAllMemory(double **vectorsArray, double **originData, Cluster *clustersArray, int k) {
    int i;
    *//* Free clusters *//*
    if (clustersArray != NULL) {
        for (i = 0; i < k; ++i) {
            MyFree(clustersArray[i].currCentroid);
            MyFree(clustersArray[i].prevCentroid);
        }
    }
    MyFree(clustersArray);
    if (vectorsArray != NULL)
        MyFree(*vectorsArray);
    if (originData != NULL)
        MyFree(*originData);
}*/

void printMatrix(double **matrix, int rows, int cols) {
    int i, j;
    double value;

    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            if (j > 0)
                printf("%c", COMMA_CHAR);
            value = matrix[i][j];
            value = NegZero(value);
            printf(PRINT_FORMAT, value); /* Print with an accuracy of 4 digits after the dot */
        }
        printf("\n");
    }
}

void printDiagMat(double *matrix, int n) {
    int i, j;
    double value;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            if (j > 0)
                printf("%c", COMMA_CHAR);
            if (i != j)
                value = 0.0;
            else {
                value = matrix[i];
                value = NegZero(value);
            }
            printf(PRINT_FORMAT, value); /* Print with an accuracy of 4 digits after the dot */
        }
        printf("\n");
    }
}

GOAL str2enum(char *str) {
    int j;
    /* Str to lowercase */
    for (j = 0; str[j] != END_OF_STRING; ++j){
        str[j] = (char) tolower(str[j]);
    }

    for (j = 0; j < NUM_OF_GOALS; ++j) {
        if (!strcmp(str, GOAL_STRING[j]))
            return j;
    }
    return NUM_OF_GOALS; /* Invalid str to enum convert */
}

void printFinalCentroids(Cluster *clustersArray, int k, int dimension) {
    int i, j;
    double value;

    for (i = 0; i < k; ++i) {
        for (j = 0; j < dimension; ++j) {
            if (j > 0)
                printf("%c", COMMA_CHAR);
            value = clustersArray[i].currCentroid[j];
            value = NegZero(value);
            printf(PRINT_FORMAT, value);
        }
        printf("\n");
    }
}

/*************************************
*********** kMeans Algorithm *********
*************************************/

Cluster *kMeans(int k, int maxIter, int dimension, int numOfVectors, double **vectorsArray, int *firstCentralIndexes) {
    int i, changes;
    Cluster *clustersArray;

    /* Initialize clusters arrays */
    clustersArray = initClusters(vectorsArray, k, dimension, firstCentralIndexes);

    for (i = 0; i < maxIter; ++i) {
        initCurrCentroidAndCounter(clustersArray, &k, &dimension); /* Update curr centroid to prev centroid and reset the counter */
        assignVectorsToClusters(vectorsArray, clustersArray, &k, &numOfVectors, &dimension);
        changes = recalcCentroids(clustersArray, &k, &dimension); /* Calculate new centroids */
        if (changes == 0) { /* Centroids stay unchanged in the current iteration */
            break;
        }
    }
    return clustersArray;
}

Cluster *initClusters(double **vectorsArray, int k, int dimension, int *firstCentralIndexes) {
    int i, j;
    Cluster *clustersArray;
    /* Allocate memory for clustersArray */
    clustersArray = (Cluster *) myAlloc(NULL, k * sizeof(Cluster));

    for (i = 0; i < k; ++i) {
        clustersArray[i].counter = 0;
        clustersArray[i].prevCentroid = (double *) myAlloc(NULL, dimension * sizeof(double));
        clustersArray[i].currCentroid = (double *) myAlloc(NULL, dimension * sizeof(double));

        if (firstCentralIndexes == NULL) { /* Kmeans */
            for (j = 0; j < dimension; ++j) {
                /* Assign the first k vectors to their corresponding clusters */
                clustersArray[i].currCentroid[j] = vectorsArray[i][j];
            }
        } else { /* kMeans++ */
            /* Assign the initial k vectors to their corresponding clusters according to the ones calculated in python */
            for (j = 0; j < dimension; ++j) {
                clustersArray[i].currCentroid[j] = vectorsArray[firstCentralIndexes[i]][j];
            }
        }
    }
    myFree(firstCentralIndexes); /* No longer necessary */
    return clustersArray;
}

double vectorsNorm(const double *vec1, const double *vec2, const int *dimension) {
    double norm = 0;
    int i;
    for (i = 0; i < *dimension; ++i) {
        norm += SQ(vec1[i] - vec2[i]);
    }
    return norm;
}

int findMyCluster(double *vec, Cluster *clustersArray, const int *k, const int *dimension) {
    int myCluster, j;
    double minNorm, norm;

    myCluster = 0;
    minNorm = vectorsNorm(vec, clustersArray[0].prevCentroid, dimension);
    for (j = 1; j < *k; ++j) { /* Find the min norm == the closest cluster */
        norm = vectorsNorm(vec, clustersArray[j].prevCentroid, dimension);
        if (norm < minNorm) {
            myCluster = j;
            minNorm = norm;
        }
    }
    return myCluster;
}

void assignVectorsToClusters(double **vectorsArray, Cluster *clustersArray, const int *k, const int *numOfVectors,
                             const int *dimension) {
    int i, j, myCluster;
    double *vec;
    for (i = 0; i < *numOfVectors; ++i) {
        vec = vectorsArray[i];
        myCluster = findMyCluster(vectorsArray[i], clustersArray, k, dimension);
        vec[*dimension] = myCluster; /* Set vector's cluster to his closest */
        for (j = 0; j < *dimension; ++j) {
            clustersArray[myCluster].currCentroid[j] += vec[j]; /* Summation of the vectors Components */
        }
        clustersArray[myCluster].counter++; /* Count the number of vectors for each cluster */
    }
}

int recalcCentroids(Cluster *clustersArray, const int *k, const int *dimension) {
    int i, j, changes = 0;
    Cluster cluster;
    for (i = 0; i < *k; ++i) {
        cluster = clustersArray[i];
        for (j = 0; j < *dimension; ++j) {
            cluster.currCentroid[j] /= cluster.counter; /* Calc the mean value */
            changes += cluster.prevCentroid[j] != cluster.currCentroid[j] ? 1
                    : 0; /* Count the number of changed centroids' components */
        }
    }
    return changes;
}

void initCurrCentroidAndCounter(Cluster *clustersArray, const int *k, const int *dimension) {
    int i, j;
    for (i = 0; i < *k; ++i) {
        for (j = 0; j < *dimension; ++j) {
            clustersArray[i].prevCentroid[j] = clustersArray[i].currCentroid[j]; /* Set prev centroid to curr centroid */
            clustersArray[i].currCentroid[j] = 0; /* Reset curr centroid */
        }
        clustersArray[i].counter = 0; /* Reset counter */
    }
}

/*************************************
*********** Jacobi Algorithm *********
*************************************/

void pivotIndex(double **matrix, int n, int *pivotRow, int *pivotCol) {
    int i, j;
    double maxAbs = -1, tempValue;
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            tempValue = fabs(matrix[i][j]);
            if (maxAbs < tempValue) {
                maxAbs = tempValue;
                *pivotRow = i;
                *pivotCol = j;
            }
        }
    }
    if (maxAbs == 0.0)
        *pivotRow = EOF; /* Matrix is diagonal */
}

void initIdentityMatrix(double **matrix, int n) {
    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            matrix[i][j] = i == j ? 1.0 : 0.0;
        }
    }
}

double jacobiRotate(double **a, double **v, int n, int i, int j) {
    double theta, t, c, s;
    double ij, ii, jj, ir, jr;
    int r;

    theta = a[j][j] - a[i][i];
    theta /= (2 * a[i][j]);
    t = 1.0 / (fabs(theta) + sqrt(SQ(theta) + 1.0));
    t = theta < 0.0 ? -t : t;
    c = 1.0 / sqrt(SQ(t) + 1.0);
    s = t * c;

    ii = a[i][i];
    jj = a[j][j];
    ij = a[i][j];
    a[i][j] = 0.0;
    a[j][i] = 0.0;
    for (r = 0; r < n; r++) {
        if (r == i)
            /* c^2 * Aii + s^2 * Ajj - 2scAij */
            a[i][i] = SQ(c) * ii + SQ(s) * jj - 2 * s * c * ij;
        else if (r == j)
            /* s^2 * Aii + c^2 * Ajj + 2scAij */
            a[j][j] = SQ(s) * ii + SQ(c) * jj + 2 * s * c * ij;
        else { /* r != j, i */
            ir = a[i][r];
            jr = a[j][r];
            a[i][r] = c * ir - s * jr;
            a[j][r] = c * jr + s * ir;
            a[r][i] = a[i][r] ;
            a[r][j] = a[j][r] ;

        }

        /* Update the eigenvector matrix */
        ir = v[i][r];
        jr = v[j][r];
        v[i][r] = c * ir - s * jr;
        v[j][r] = c * jr + s * ir;
    }

    return 2 * SQ(ij); /* offNormDiff: Off(A)^2 - Off(A')^2 = 2 * Aij^2 */
}

Eigenvalue *sortEigenvalues(double **a, int n) {
    int i;
    Eigenvalue *eigenvalues = myAlloc(NULL, n * sizeof(Eigenvalue));

    for(i = 0; i < n; ++i) {
        eigenvalues[i].value = a[i][i];
        eigenvalues[i].vector = i;
    }

    qsort(eigenvalues, n, sizeof(Eigenvalue), cmpEigenvalues);
    return eigenvalues;
}

int cmpEigenvalues (const void *p1, const void *p2) {
    const Eigenvalue *q1 = p1, *q2 = p2;

    if (q1->value > q2->value)
        return 1;
    else if (q1->value < q2->value)
        return -1;
    return (q1->vector - q2->vector); /* Keeps qsort comparator stable */
}

void jacobiAlgorithm(double **matrix, double **eigenvectorsMat, int n) {
    double diffOffNorm;
    int jacobiIterCounter, pivotRow, pivotCol;

    initIdentityMatrix(eigenvectorsMat, n);

    jacobiIterCounter = 0;
    do {
        pivotIndex(matrix, n, &pivotRow, &pivotCol);
        if (pivotRow == EOF) /* Matrix is already diagonal */
            break;
        diffOffNorm = jacobiRotate(matrix, eigenvectorsMat, n, pivotRow, pivotCol);
        jacobiIterCounter++;
    } while (jacobiIterCounter < MAX_JACOBI_ITER && diffOffNorm >= EPSILON);
}

int eigengapHeuristicKCalc(Eigenvalue *eigenvalues, int n) {
    int i, maxIndex, m;
    double maxDelta, delta;

    m = n / 2;
    maxDelta = -1;
    maxIndex = -1;
    for (i = 0; i < m; i++) {
        delta = eigenvalues[i + 1].value - eigenvalues[i].value;
        if (maxDelta < delta) {
            maxDelta = delta;
            maxIndex = i;
        }
    }

    return maxIndex + 1;
}

double **initTMatrix(Eigenvalue *eigenvalues, double **eigenvectorsMat, void *freeUsedMem, int n, int k) {
    int i, j;
    double sumSqRow, value;
    double **tMat = (double **) alloc2DArray(n, k + 1, sizeof(double), sizeof(double *), freeUsedMem);

    for (i = 0; i < n; ++i) {
        sumSqRow = 0.0;
        tMat[i][k] = 0.0; /* Cluster ID */
        for (j = 0; j < k; ++j) {
            value = eigenvectorsMat[eigenvalues[j].vector][i];
            tMat[i][j] = value;
            sumSqRow += SQ(value);
        }
        if (sumSqRow != 0.0) { /* TODO check about zero line */
            sumSqRow = 1.0 / sqrt(sumSqRow);
            for (j = 0; j < k; ++j) {
                tMat[i][j] *= sumSqRow;
            }
        }
    }
    /* Free memory */
    MyFree(eigenvalues);
    if (eigenvectorsMat != NULL) {
        myFree(*eigenvectorsMat);
        eigenvectorsMat = NULL;
    }
    return tMat;
}

/********* Ben's part *********/

/*
 * Builds an n*n matrix while n is the number of vectors
 * Assign the weighted factor as requested
 * Param - vectorsArray matrix, number of vectors and dimensions
 * Using sqrt and norm functions
 * Returns the weighted Matrix
 */
double **weightedMatrix(double **wMatrix, double **vectorsArray, int numOfVectors, int dimension) {
    int i, j;
    double norm;

    for (i = 0; i < numOfVectors; i++) {
        wMatrix[i][i] = 0.0;
        for (j = i + 1; j < numOfVectors; j++) {
            norm = sqrt(vectorsNorm(vectorsArray[i], vectorsArray[j], &dimension));
            wMatrix[i][j] = exp(-0.5 * norm);
            wMatrix[j][i] = wMatrix[i][j];
        }
    }
    return wMatrix;
}

/*
 *Build the Diagonal Degree Matrix from the Weighted Adjacency Matrix
 *returns a pointer to the Diagonal Degree Matrix
 * */
double *dMatrix(double **wMatrix, int dim) {
    int i, j;
    double *dMatrix, sum;
    dMatrix = (double *) myAlloc(NULL, dim * sizeof(double));

    for (i = 0; i < dim; i++) {
        sum = 0.0;
        for (j = 0; j < dim; j++) {
            sum += wMatrix[i][j];
        }
        dMatrix[i] = sum;
    }
    return dMatrix;
}

/*
 * Builds The Normalized Graph Laplacian from the weighted matrix and the degree matrix
 * returns a pointer to the Normalized Graph Laplacian Matrix*/
double **laplacian(double **wMatrix, double *dMatrix, int numOfVectors) {
    int i, j;
    double **lMatrix = wMatrix;

    for (i = 0; i < numOfVectors; i++) {
        dMatrix[i] = 1 / sqrt(dMatrix[i]);
    }

    for (i = 0; i < numOfVectors; i++) {
        for (j = 0; j < numOfVectors; j++) {
            lMatrix[i][j] = -1.0 * dMatrix[i] * dMatrix[j] * wMatrix[i][j];
            if (i == j)
                lMatrix[i][j] += 1.0;
        }
    }
    MyFree(dMatrix); /* End of need */
    return lMatrix;
}

void calcDim(int *dimension, FILE *file, double *firstLine) {
    char c;
    double value;
    *dimension = 0;
    do {
        fscanf(file, "%lf%c", &value, &c);
        firstLine[(*dimension)++] = value;
    } while (c != '\n' && c != '\r');
}

double **readDataFromFile(int *rows, int *cols, char *fileName, GOAL goal) {
    int counter, maxLen;
    char c;
    double value;
    FILE *file;
    double **matrix, *dataBlock;

    maxLen = goal != jacobi ? MAX_FEATURES : MAX_DATAPOINTS;
    dataBlock = (double *) myAlloc(NULL, maxLen * sizeof(double));
    file = fopen(fileName, "r");
    MyAssert(file != NULL);
    calcDim(cols, file, dataBlock);

    maxLen = goal != jacobi ? MAX_DATAPOINTS : *cols;
    /* Reallocate memory to hold the data */
    dataBlock = (double *) myAlloc(dataBlock ,maxLen * (*cols) * sizeof(double));

    counter = *cols;
    while (fscanf(file, "%lf%c", &value, &c) != EOF) {
        dataBlock[counter++] = value;
    }
    MyAssert(fclose(file) != EOF);

    *rows = counter / *cols;
    matrix = (double **) alloc2DArray(*rows, *cols, sizeof(double), sizeof(double *), dataBlock);
    return matrix;
}

void validateAndAssignInput(int argc, char **argv, int *k, GOAL *goal, char **filenamePtr) {
    char *nextCh;

    if (argc >= REQUIRED_NUM_OF_ARGUMENTS) {
        *goal = str2enum(argv[GOAL_ARGUMENT]);
        *filenamePtr = argv[REQUIRED_NUM_OF_ARGUMENTS - 1];
        if (*goal < NUM_OF_GOALS) {
            if (*goal != spk) {
                *k = 0;
                return;
            } else {
                /* k greater than zero and the conversion succeeded, valid goal */
                *k = strtol(argv[K_ARGUMENT], &nextCh, 10);
                if (*k >= 0 && *nextCh == END_OF_STRING)
                    return;
            }
        }
    }
    printf(INVALID_INPUT_MSG);
    exit(0);
}
