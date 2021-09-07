#include "spkmeans.h"

#define SQ(x) ((x)*(x))
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

#define MyAssert(exp) \
if (!(exp)) {      \
fprintf(stderr, ERROR_MSG); \
freeAllMemory();                      \
assert (0);          \
exit(EXIT_FAILURE);     \
}

typedef struct {
    double *prevCentroid;
    double *currCentroid;
    int counter; /* Number of vectors (datapoints) in cluster */
} Cluster;

typedef struct {
    double value;
    int vector;
} Eigenvalue;

Cluster *initClusters(double **vectorsArray, int k, int dimension, const int *firstCentralIndexes);
double **buildFinalCentroidsMat(Cluster *clustersArray, double *vecToClusterLabeling, int k, int dimension);
double vectorsNorm(const double *vec1, const double *vec2, int dimension); /* Calculate the norm between 2 vectors */
int findMyCluster(double *vec, Cluster *clustersArray, int k, int dimension); /* Return the vector's closest cluster (in terms of norm) */
void assignVectorsToClusters(double **vectorsArray, Cluster *clustersArray, double *vecToClusterLabeling, int k, int numOfVectors, int dimension);  /* For any vector assign to his closest cluster */
int recalcCentroids(Cluster *clustersArray, int k, int dimension); /* Recalculate clusters' centroids, return number of changes */
void initCurrCentroidAndCounter(Cluster *clustersArray, int k, int dimension); /* Set curr centroid to prev centroid and reset the counter */
double jacobiRotate(double **a, double **v, int n, int i, int j);
double **initIdentityMatrix(int n);
int eigengapHeuristicKCalc(Eigenvalue *eigenvalues, int n);
void printMatrix(double **matrix, int rows, int cols);
void validateAndAssignInput(int argc, char **argv, int *k, GOAL *goal, char **filenamePtr);
double **readDataFromFile(int *rows, int *cols, char *fileName, GOAL goal);
double **weightedMatrix(double **vectorsArray, int numOfVectors, int dimension);
double **dMatrix(double **wMatrix, int n);
double **laplacian(double **wMatrix, double **dMatrix, int numOfVectors);
void calcDim(int *dimension, FILE *file, double *firstLine);
double **initTMatrix(Eigenvalue *eigenvalues, double **eigenvectorsMat, int n, int k);
Eigenvalue *sortEigenvalues(double **a, int n);
int cmpEigenvalues (const void *p1, const void *p2);
void pivotIndex(double **matrix, int n, int *pivotRow, int *pivotCol);
void printJacobi(double **a, double **v, int n);

/**********************************
*********** Main ******************
**********************************/

int main(int argc, char *argv[]) {
    int k, dimension, numOfDatapoints;
    GOAL goal;
    char *filename;
    double **datapointsArray, **calcMat;
    headOfMemList = NULL, freeUsedMem = NULL;

    validateAndAssignInput(argc, argv, &k, &goal, &filename);
    datapointsArray = readDataFromFile(&numOfDatapoints, &dimension, filename, goal);
    if (goal == spk && k >= numOfDatapoints) {
        printf(INVALID_INPUT_MSG);
    } else {
        if (goal == jacobi) {
            calcMat = jacobiAlgorithm(datapointsArray, numOfDatapoints);
        } else {
            calcMat = dataAdjustmentMatrices(datapointsArray, goal, &k, dimension, numOfDatapoints);
            MyRecycleMatFree(datapointsArray);
        }

        /* Print results */
        switch (goal) {
            case jacobi:
                printJacobi(datapointsArray, calcMat, numOfDatapoints);
                break;
            case wam:
            case ddg:
            case lnorm:
                printMatrix(calcMat, numOfDatapoints, numOfDatapoints);
                break;
            case spk:
                calcMat = kMeans(calcMat, numOfDatapoints, k, k, NULL, MAX_KMEANS_ITER);
                printMatrix(calcMat, k, k);
                break;
            default:
                break; /* TODO exit prog? */
        }
    }

    freeAllMemory();
    return 0;
}

/**********************************
******** Spectral Clustering ******
**********************************/

double **dataAdjustmentMatrices(double **datapointsArray, GOAL goal, int *k, int dimension, int numOfDatapoints) {
    double **tMat, **wMat, **lnormMat, **eigenvectorsMat, **ddgMat;
    GOAL task;
    Eigenvalue *eigenvalues;

    task = wam;
    /* The Weighted Adjacency Matrix - step 1.1.1 */
    wMat = weightedMatrix(datapointsArray, numOfDatapoints, dimension);
    if (task++ == goal)
        return wMat;
    /* The Diagonal Degree Matrix - step 1.1.2 */\
    ddgMat = dMatrix(wMat, numOfDatapoints);
    if (task++ == goal)
        return ddgMat;
    /* The Normalized Graph Laplacian - step 2 */
    lnormMat = laplacian(wMat, ddgMat, numOfDatapoints);
    if (task == goal)
        return lnormMat;
    MyRecycleMatFree(ddgMat);
    /* Determine k and obtain the first k eigenvectors using Jacobi algorithm - step 3 */
    eigenvectorsMat = jacobiAlgorithm(lnormMat, numOfDatapoints);
    eigenvalues = sortEigenvalues(lnormMat, numOfDatapoints);
    MyRecycleMatFree(lnormMat);
    if (*k == 0)
        *k = eigengapHeuristicKCalc(eigenvalues, numOfDatapoints);
    /* Form the matrix T (from U) - step 4 + 5 */
    tMat = initTMatrix(eigenvalues, eigenvectorsMat, numOfDatapoints, *k);
    MyRecycleMatFree(eigenvectorsMat);
    MyFree(eigenvalues);
    return tMat;
}

/*
 * Builds an n*n matrix while n is the number of vectors
 * Assign the weighted factor as requested
 * Param - vectorsArray matrix, number of vectors and dimensions
 * Using sqrt and norm functions
 * Returns the weighted Matrix
 */
double **weightedMatrix(double **vectorsArray, int numOfVectors, int dimension) {
    int i, j;
    double norm, **wMatrix = (double **) alloc2DArray(numOfVectors, numOfVectors, sizeof(double), sizeof(double *), freeUsedMem);

    for (i = 0; i < numOfVectors; i++) {
        wMatrix[i][i] = 0.0;
        for (j = i + 1; j < numOfVectors; j++) {
            norm = sqrt(vectorsNorm(vectorsArray[i], vectorsArray[j], dimension));
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
double **dMatrix(double **wMatrix, int n) {
    int i, j;
    double **dMatrix, sum;
    dMatrix = (double **) alloc2DArray(n, n, sizeof(double), sizeof(double *), freeUsedMem);

    for (i = 0; i < n; i++) {
        sum = 0.0;
        for (j = 0; j < n; j++) {
            dMatrix[i][j] = 0.0;
            sum += wMatrix[i][j];
        }
        dMatrix[i][i] = sum;
    }
    return dMatrix;
}

/*
 * Builds The Normalized Graph Laplacian from the weighted matrix and the degree matrix
 * returns a pointer to the Normalized Graph Laplacian Matrix*/
double **laplacian(double **wMatrix, double **dMatrix, int numOfVectors) {
    int i, j;
    double **lMatrix = wMatrix;

    for (i = 0; i < numOfVectors; i++) {
        dMatrix[i][i] = 1 / sqrt(dMatrix[i][i]);
    }

    for (i = 0; i < numOfVectors; i++) {
        for (j = 0; j < numOfVectors; j++) {
            lMatrix[i][j] = -1.0 * dMatrix[i][i] * dMatrix[j][j] * wMatrix[i][j];
            if (i == j)
                lMatrix[i][j] += 1.0;
        }
    }
    return lMatrix;
}

double **initTMatrix(Eigenvalue *eigenvalues, double **eigenvectorsMat, int n, int k) {
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
    return tMat;
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

/**********************************
******** Memory Allocation ********
**********************************/

void *myAlloc(void *effectiveUsedMem, size_t size) {
    void *usedMem = effectiveUsedMem != NULL ? (void *)((char *)effectiveUsedMem - SIZE_OF_VOID_2PTR * 2) : NULL;
    void *blockMem, **blockMemPlusPtr = (void **)realloc(usedMem, size + SIZE_OF_VOID_2PTR * 2);
    MyAssert(blockMemPlusPtr != NULL);
    blockMem = (void *)((char *)blockMemPlusPtr + SIZE_OF_VOID_2PTR * 2);
    if (effectiveUsedMem == NULL) { /* New Allocation */
        /* Set ptr to the prev/next dynamic allocated memory block */
        blockMemPlusPtr[0] = NULL;
        if (headOfMemList != NULL) {
            blockMemPlusPtr[1] = headOfMemList;
            ((void **)headOfMemList)[0] = blockMemPlusPtr;
        } else
            blockMemPlusPtr[1] = NULL;
        headOfMemList = blockMemPlusPtr;
    } else {
        if (usedMem != blockMemPlusPtr) {
            if (blockMemPlusPtr[0] != NULL)
                ((void **)blockMemPlusPtr[0])[1] = blockMemPlusPtr;
            else
                headOfMemList = blockMemPlusPtr;
            if(blockMemPlusPtr[1] != NULL)
                ((void **)blockMemPlusPtr[1])[0] = blockMemPlusPtr;
        }
        if (freeUsedMem == effectiveUsedMem)
            freeUsedMem = NULL; /* Unfree the memory - used again */
    }
    return blockMem;
}

void **alloc2DArray(int rows, int cols, size_t basicSize, size_t basicPtrSize, void *recycleMemBlock) {
    int i;
    void *blockMem, **matrix;
    /* Reallocate block of memory */
    blockMem = myAlloc(recycleMemBlock, rows * cols * basicSize + rows * basicPtrSize);
    matrix = (void **) ((char *)blockMem + rows * cols * basicSize);

    for (i = 0; i < rows; ++i) {
        /* Set matrix to point to 2nd dimension array */
        *((void **)((char *)matrix + i * basicPtrSize)) = (void *) (((char *) blockMem) + i * cols * basicSize);
    }
    return matrix;
}

void myFree(void *effectiveBlockMem) {
    void **blockMem;
    if (effectiveBlockMem == NULL)
        return;
    blockMem = (void **)((char *)effectiveBlockMem - SIZE_OF_VOID_2PTR * 2);
    if(blockMem[0] != NULL) {
        ((void **)blockMem[0])[1] = blockMem[1]; /* Set prev's next to current next */
    } else {
        headOfMemList = blockMem[1];
    }
    if(blockMem[1] != NULL) {
        ((void **)blockMem[1])[0] = blockMem[0]; /* Set next's prev to current prev */
    }
    free(blockMem);
}

void freeAllMemory() {
    void **currBlock = headOfMemList, **nextBlock;

    while (currBlock != NULL) {
        nextBlock = currBlock[1];
        free(currBlock);
        currBlock = nextBlock;
    }
    headOfMemList = NULL;
}

/**********************************
******** Printing results *********
**********************************/

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

/*************************************
*********** kMeans Algorithm *********
*************************************/

double **kMeans(double **vectorsArray, int numOfVectors, int dimension, int k, const int *firstCentralIndexes, int maxIter) {
    int i, changes;
    Cluster *clustersArray;
    double *vecToClusterLabeling, **finalCentroidsAndVecLabeling;

    /* Initialize clusters arrays */
    clustersArray = initClusters(vectorsArray, k, dimension, firstCentralIndexes);
    vecToClusterLabeling = (double *) myAlloc(freeUsedMem, numOfVectors * sizeof(double));

    for (i = 0; i < maxIter; ++i) {
        initCurrCentroidAndCounter(clustersArray, k, dimension); /* Update curr centroid to prev centroid and reset the counter */
        assignVectorsToClusters(vectorsArray, clustersArray, vecToClusterLabeling, k, numOfVectors, dimension);
        changes = recalcCentroids(clustersArray, k, dimension); /* Calculate new centroids */
        if (changes == 0) { /* Centroids stay unchanged in the current iteration */
            break;
        }
    }
    finalCentroidsAndVecLabeling = buildFinalCentroidsMat(clustersArray, vecToClusterLabeling, k, dimension);
    MyFree(clustersArray);
    return finalCentroidsAndVecLabeling;
}

Cluster *initClusters(double **vectorsArray, int k, int dimension, const int *firstCentralIndexes) {
    int i, j;
    Cluster *clustersArray;
    double **centroidMat;
    /* Allocate memory for clustersArray */
    clustersArray = (Cluster *) myAlloc(NULL, k * sizeof(Cluster));
    centroidMat = (double **) alloc2DArray(k + 1, dimension * 2, sizeof(double), sizeof(double *), freeUsedMem);

    for (i = 0; i < k; ++i) {
        clustersArray[i].counter = 0;
        clustersArray[i].currCentroid = centroidMat[i];
        clustersArray[i].prevCentroid = centroidMat[i] + dimension;

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
    return clustersArray;
}

void assignVectorsToClusters(double **vectorsArray, Cluster *clustersArray, double *vecToClusterLabeling, int k, int numOfVectors, int dimension) {
    int i, j, myCluster;
    double *vec;
    for (i = 0; i < numOfVectors; ++i) {
        vec = vectorsArray[i];
        myCluster = findMyCluster(vec, clustersArray, k, dimension);
        vecToClusterLabeling[i] = myCluster; /* Set vector's cluster to his closest */
        for (j = 0; j < dimension; ++j) {
            clustersArray[myCluster].currCentroid[j] += vec[j]; /* Summation of the vectors Components */
        }
        clustersArray[myCluster].counter++; /* Count the number of vectors for each cluster */
    }
}

int findMyCluster(double *vec, Cluster *clustersArray, int k, int dimension) {
    int myCluster, j;
    double minNorm, norm;

    myCluster = 0;
    minNorm = vectorsNorm(vec, clustersArray[0].prevCentroid, dimension);
    for (j = 1; j < k; ++j) { /* Find the min norm == the closest cluster */
        norm = vectorsNorm(vec, clustersArray[j].prevCentroid, dimension);
        if (norm < minNorm) {
            myCluster = j;
            minNorm = norm;
        }
    }
    return myCluster;
}

double vectorsNorm(const double *vec1, const double *vec2, int dimension) {
    double norm = 0;
    int i;
    for (i = 0; i < dimension; ++i) {
        norm += SQ(vec1[i] - vec2[i]);
    }
    return norm;
}

int recalcCentroids(Cluster *clustersArray, int k, int dimension) {
    int i, j, changes = 0;
    Cluster cluster;
    for (i = 0; i < k; ++i) {
        cluster = clustersArray[i];
        for (j = 0; j < dimension; ++j) {
            cluster.currCentroid[j] /= cluster.counter; /* Calc the mean value */
            /* Count the number of changed centroids' components */
            changes += cluster.prevCentroid[j] != cluster.currCentroid[j] ? 1: 0;
        }
    }
    return changes;
}

void initCurrCentroidAndCounter(Cluster *clustersArray, int k, int dimension) {
    int i, j;
    for (i = 0; i < k; ++i) {
        for (j = 0; j < dimension; ++j) {
            clustersArray[i].prevCentroid[j] = clustersArray[i].currCentroid[j]; /* Set prev centroid to curr centroid */
            clustersArray[i].currCentroid[j] = 0; /* Reset curr centroid */
        }
        clustersArray[i].counter = 0; /* Reset counter */
    }
}

double **buildFinalCentroidsMat(Cluster *clustersArray, double *vecToClusterLabeling, int k, int dimension) {
    double **matrix = (double **)(clustersArray[k - 1].prevCentroid + dimension * 3);
    matrix[k] = vecToClusterLabeling;
    return matrix;
}

/*************************************
*********** Jacobi Algorithm *********
*************************************/

double **jacobiAlgorithm(double **matrix, int n) {
    double diffOffNorm, **eigenvectorsMat;
    int jacobiIterCounter, pivotRow, pivotCol;

    eigenvectorsMat = initIdentityMatrix(n);

    jacobiIterCounter = 0;
    do {
        pivotIndex(matrix, n, &pivotRow, &pivotCol);
        if (pivotRow == EOF) /* Matrix is already diagonal */
            break;
        diffOffNorm = jacobiRotate(matrix, eigenvectorsMat, n, pivotRow, pivotCol);
        jacobiIterCounter++;
    } while (jacobiIterCounter < MAX_JACOBI_ITER && diffOffNorm >= EPSILON);

    return eigenvectorsMat;
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

double **initIdentityMatrix(int n) {
    int i, j;
    double **matrix = (double **) alloc2DArray(n, n, sizeof(double), sizeof(double *), freeUsedMem);

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            matrix[i][j] = i == j ? 1.0 : 0.0;
        }
    }
    return matrix;
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

/*************************************
********* Auxiliary Functions ********
*************************************/

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

void calcDim(int *dimension, FILE *file, double *firstLine) {
    char c;
    double value;
    *dimension = 0;
    do {
        fscanf(file, "%lf%c", &value, &c);
        firstLine[(*dimension)++] = value;
    } while (c != '\n' && c != '\r');
}
