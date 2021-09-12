#include "spkmeans.h"
#include "spkinnerfunctions.h"
/* This file implements all C functions - SPK, KMEANS, JACOBI and others */

/*******************************************************************************
********************************** Main ****************************************
*******************************************************************************/

/**
 * Main spectral clustering program.
 * Print the result according to the user's goal.
 * @param argv - User's arguments: k, goal, filename
 */
int main(int argc, char *argv[]) {
    int k, dimension, numOfDatapoints;
    GOAL goal;
    char *filename;
    double **datapointsArray, **calcMat;
    headOfMemList = NULL, freeUsedMem = NULL; /* Init C memory containers */

    /* Validate and read user's input */
    validateAndAssignInput(argc, argv, &k, &goal, &filename);
    datapointsArray = readDataFromFile(&numOfDatapoints, &dimension, filename, goal);
    if (goal == spk && k >= numOfDatapoints) {
        printf(INVALID_INPUT_MSG);
    } else { /* SPK algorithm */
        if (goal == jacobi) {
            calcMat = jacobiAlgorithm(datapointsArray, numOfDatapoints);
        } else { /* Get T/W/D/Lnorm matrix */
            calcMat = dataAdjustmentMatrices(datapointsArray, goal, &k, dimension,
                                             numOfDatapoints);
            MyRecycleMatFree(datapointsArray);
        }
        MyAssert(calcMat != NULL);

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
                /* Run kmeans on T matrix */
                calcMat = kMeans(calcMat, numOfDatapoints, k, k, NULL, MAX_KMEANS_ITER);
                MyAssert(calcMat != NULL);
                printMatrix(calcMat, k, k);
                break;
            default:
                MyAssert(0); /* Unexpected goal error */
        }
    }

    freeAllMemory();
    return 0;
}

/*******************************************************************************
***************************** Spectral Clustering ******************************
*******************************************************************************/

/* The function runs spk algorithm steps and stop at the desired goal.
 * The function returns the relevant matrix depended on the GOAL. */
double **dataAdjustmentMatrices(double **datapointsArray, GOAL goal, int *k,
                                int dimension, int numOfDatapoints) {
    double **tMat, **wMat, **lnormMat, **eigenvectorsMat, **ddgMat;
    GOAL task;
    Eigenvalue *eigenvalues;

    task = wam; /* Start from the first step */
    /* The Weighted Adjacency Matrix - step 1.1.1 */
    wMat = weightedMatrix(datapointsArray, numOfDatapoints, dimension);
    if (task++ == goal || wMat == NULL)
        return wMat;
    /* The Diagonal Degree Matrix - step 1.1.2 */
    ddgMat = dMatrix(wMat, numOfDatapoints);
    if (task++ == goal || ddgMat == NULL)
        return ddgMat;
    /* The Normalized Graph Laplacian - step 2 */
    lnormMat = laplacian(wMat, ddgMat, numOfDatapoints);
    if (task == goal || lnormMat == NULL)
        return lnormMat;
    MyRecycleMatFree(ddgMat);
    /* Determine k and obtain the first k eigenvectors using Jacobi algorithm - step 3 */
    eigenvectorsMat = jacobiAlgorithm(lnormMat, numOfDatapoints);
    eigenvalues = sortEigenvalues(lnormMat, numOfDatapoints);
    if (eigenvectorsMat == NULL || eigenvalues == NULL) return NULL;
    MyRecycleMatFree(lnormMat);

    if (*k == 0) /* If k not provided */
        *k = eigengapHeuristicKCalc(eigenvalues, numOfDatapoints);
    /* Form the matrix T (from U) - step 4 + 5 */
    tMat = initTMatrix(eigenvalues, eigenvectorsMat, numOfDatapoints, *k);
    MyRecycleMatFree(eigenvectorsMat);
    MyFree(eigenvalues);
    return tMat;
}

/* This function form The Weighted Adjacency Matrix out of vectors list. */
double **weightedMatrix(double **vectorsArray, int numOfVectors, int dimension) {
    int i, j;
    double norm;
    double **wMatrix = (double **) alloc2DArray(numOfVectors, numOfVectors,
                                                sizeof(double), sizeof(double *), freeUsedMem);

    if (wMatrix != NULL) { /* Memory allocation fail */
        for (i = 0; i < numOfVectors; i++) {
            wMatrix[i][i] = 0.0; /* No loops allowed */
            for (j = i + 1; j < numOfVectors; j++) {
                norm = sqrt(vectorsSqNorm(vectorsArray[i], vectorsArray[j],
                                          dimension));
                wMatrix[i][j] = exp(-0.5 * norm);
                wMatrix[j][i] = wMatrix[i][j]; /* Symmetry */
            }
        }
    }
    return wMatrix;
}

/* This function form the Diagonal Degree Matrix of Weighted Adjacency Matrix. */
double **dMatrix(double **wMatrix, int n) {
    int i, j;
    double **dMatrix, sum;
    dMatrix = (double **) alloc2DArray(n, n, sizeof(double), sizeof(double *),
                                       freeUsedMem);

    if (dMatrix != NULL) { /* Memory allocation fail */
        for (i = 0; i < n; i++) {
            sum = 0.0;
            for (j = 0; j < n; j++) {
                dMatrix[i][j] = 0.0; /* Off-diag set to zero */
                sum += wMatrix[i][j]; /* Sum W's i row */
            }
            dMatrix[i][i] = sum;
        }
    }
    return dMatrix;
}

/* This function form the Normalized Graph Laplacian matrix in a given D + W matrix. */
double **laplacian(double **wMatrix, double **dMatrix, int numOfVectors) {
    int i, j;
    double **lMatrix = wMatrix;

    /* Calc D^-1/2 */
    for (i = 0; i < numOfVectors; i++) {
        dMatrix[i][i] = 1 / sqrt(dMatrix[i][i]);
    }

    /* Lnorm = I - D^-1/2 * W * D^-1/2 */
    for (i = 0; i < numOfVectors; i++) {
        for (j = 0; j < numOfVectors; j++) {
            lMatrix[i][j] = -1.0 * dMatrix[i][i] * dMatrix[j][j] * wMatrix[i][j];
            if (i == j) /* Identity matrix: Add 1 to the primary diagonal */
                lMatrix[i][j] += 1.0;
        }
    }
    return lMatrix;
}

/* This function form T matrix from Lnorm eigenvalues, eigenvectors and k. */
double **initTMatrix(Eigenvalue *eigenvalues, double **eigenvectorsMat, int n, int k) {
    int i, j;
    double sumSqRow, value;
    double **tMat = (double **) alloc2DArray(n, k, sizeof(double),
                                             sizeof(double *), freeUsedMem);

    if (tMat != NULL) { /* Memory allocation fail */
        for (i = 0; i < n; ++i) {
            sumSqRow = 0.0;
            /* Form U matrix */
            for (j = 0; j < k; ++j) {
                value = eigenvectorsMat[eigenvalues[j].vector][i];
                tMat[i][j] = value;
                sumSqRow += SQ(value);
            }
            if (sumSqRow == 0.0) /* Zero line */
                return NULL;
            /* Normalize U rows == T */
            sumSqRow = 1.0 / sqrt(sumSqRow);
            for (j = 0; j < k; ++j) {
                tMat[i][j] *= sumSqRow;
            }
        }
    }
    return tMat;
}

/* This function calculate the optimum k using Eigengap Heuristic method. */
int eigengapHeuristicKCalc(Eigenvalue *eigenvalues, int n) {
    int i, maxIndex, m;
    double maxDelta, delta;

    m = n / 2; /* floor(n/2) */
    maxDelta = -1;
    maxIndex = -1;
    for (i = 0; i < m; i++) {
        delta = eigenvalues[i + 1].value - eigenvalues[i].value;
        if (maxDelta < delta) {
            maxDelta = delta;
            maxIndex = i;
        }
    }

    return maxIndex + 1; /* Index starts from 0 */
}

/*******************************************************************************
********************************** KMeans **************************************
*******************************************************************************/

/* This function runs the main KMeans clustering algorithm. */
double **kMeans(double **vectorsArray, int numOfVectors, int dimension, int k,
                const int *firstCentralIndexes, int maxIter) {
    int i, changes;
    Cluster *clustersArray;
    double *vecToClusterLabeling, **finalCentroidsAndVecLabeling;

    /* Initialize clusters arrays */
    clustersArray = initClusters(vectorsArray, k, dimension, firstCentralIndexes);
    vecToClusterLabeling = (double *) myAlloc(freeUsedMem, numOfVectors * sizeof(double));
    if (vecToClusterLabeling == NULL || clustersArray == NULL) return NULL;

    for (i = 0; i < maxIter; ++i) {
        /* Update curr centroid to prev centroid and reset the counter */
        initCurrCentroidAndCounter(clustersArray, k, dimension);
        assignVectorsToClusters(vectorsArray, clustersArray, vecToClusterLabeling,
                                k, numOfVectors, dimension);
        /* Calculate new centroids */
        changes = recalcCentroids(clustersArray, k, dimension);
        if (changes == 0) {
            /* Centroids stay unchanged in the current iteration == convergence */
            break;
        }
    }
    /* Organize the results as a matrix */
    finalCentroidsAndVecLabeling = buildFinalCentroidsMat(clustersArray, vecToClusterLabeling,
                                                          k, dimension);
    MyFree(clustersArray);
    return finalCentroidsAndVecLabeling;
}

/* This function initialize the clusters array. */
Cluster *initClusters(double **vectorsArray, int k, int dimension,
                      const int *firstCentralIndexes) {
    int i, j;
    Cluster *clustersArray;
    double **centroidMat;

    /* Allocate memory for clustersArray */
    clustersArray = (Cluster *) myAlloc(NULL, k * sizeof(Cluster));
    centroidMat = (double **) alloc2DArray(k + 1, dimension * 2,
                                           sizeof(double), sizeof(double *), freeUsedMem);
    if (clustersArray == NULL || centroidMat == NULL) return NULL;

    for (i = 0; i < k; ++i) {
        clustersArray[i].counter = 0;
        clustersArray[i].currCentroid = centroidMat[i];
        clustersArray[i].prevCentroid = centroidMat[i] + dimension;

        if (firstCentralIndexes == NULL) { /* KMeans */
            for (j = 0; j < dimension; ++j) {
                /* Assign the first k vectors to their corresponding clusters */
                clustersArray[i].currCentroid[j] = vectorsArray[i][j];
            }
        } else { /* KMeans++ */
            /* Assign the initial k vectors to their corresponding clusters
             * according to the ones calculated in python */
            for (j = 0; j < dimension; ++j) {
                clustersArray[i].currCentroid[j] = vectorsArray[firstCentralIndexes[i]][j];
            }
        }
    }
    return clustersArray;
}

/* This function assign the closest cluster for each vector. */
void assignVectorsToClusters(double **vectorsArray, Cluster *clustersArray,
                             double *vecToClusterLabeling, int k,
                             int numOfVectors, int dimension) {
    int i, j, myCluster;
    double *vec;

    for (i = 0; i < numOfVectors; ++i) {
        vec = vectorsArray[i];
        /* Set vector's cluster to his closest */
        myCluster = findMyCluster(vec, clustersArray, k, dimension);
        vecToClusterLabeling[i] = myCluster;

        for (j = 0; j < dimension; ++j) {
            /* Summation of the vectors Components */
            clustersArray[myCluster].currCentroid[j] += vec[j];
        }
        /* Count the number of vectors for each cluster */
        clustersArray[myCluster].counter++;
    }
}

/* This function finds vector's closest cluster (in terms of euclidean norm). */
int findMyCluster(double *vec, Cluster *clustersArray, int k, int dimension) {
    int myCluster, j;
    double minNorm, norm;

    myCluster = 0;
    minNorm = vectorsSqNorm(vec, clustersArray[0].prevCentroid, dimension);
    for (j = 1; j < k; ++j) { /* Find the min norm == the closest cluster */
        norm = vectorsSqNorm(vec, clustersArray[j].prevCentroid, dimension);
        if (norm < minNorm) {
            myCluster = j;
            minNorm = norm;
        }
    }
    return myCluster;
}

/* This function calculates the squared euclidean norm between two vectors. */
double vectorsSqNorm(const double *vec1, const double *vec2, int dimension) {
    double sqNorm = 0;
    int i;

    for (i = 0; i < dimension; ++i) {
        sqNorm += SQ(vec1[i] - vec2[i]);
    }
    return sqNorm;
}

/* This function recalculates clusters centroids after one kmeans iteration. */
int recalcCentroids(Cluster *clustersArray, int k, int dimension) {
    Cluster cluster;
    int i, j, changes = 0;

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

/* This function organize clusters array for the next iteration:
 *      Reset counter and current centroid (to be zero vector)
 *      Update previous centroids to be current centroids */
void initCurrCentroidAndCounter(Cluster *clustersArray, int k, int dimension) {
    int i, j;
    for (i = 0; i < k; ++i) {
        for (j = 0; j < dimension; ++j) {
            /* Set prev centroid to curr centroid */
            clustersArray[i].prevCentroid[j] = clustersArray[i].currCentroid[j];
            clustersArray[i].currCentroid[j] = 0; /* Reset curr centroid */
        }
        clustersArray[i].counter = 0; /* Reset counter */
    }
}

/* This function organize KMeans result into a matrix:
 *      First k rows - Clusters centroids
 *      Last row (Could be from different length) vectors to clusters labeling */
double **buildFinalCentroidsMat(Cluster *clustersArray, double *vecToClusterLabeling,
                                int k, int dimension) {
    /* Restore first row pointer of centroid matrix from "initClusters" */
    double **matrix = (double **)(clustersArray[k - 1].prevCentroid + dimension * 3);
    /* Assign last row to point to vectors labeling array */
    matrix[k] = vecToClusterLabeling;
    return matrix;
}

/*******************************************************************************
****************************** Jacobi Algorithm ********************************
*******************************************************************************/

/* This function performs Jacobi's diagonal method on a symmetric matrix. */
double **jacobiAlgorithm(double **matrix, int n) {
    double diffOffNorm, **eigenvectorsMat;
    int jacobiIterCounter, pivotRow, pivotCol;

    eigenvectorsMat = initIdentityMatrix(n); /* Init the eigenvectors matrix */

    if (eigenvectorsMat != NULL) { /* Memory allocation fail */
        jacobiIterCounter = 0;
        do {
            pivotIndex(matrix, n, &pivotRow, &pivotCol); /* Choose pivot index */
            if (pivotRow == EOF) /* Matrix is already diagonal */
                break;
            /* perform rotation */
            diffOffNorm = jacobiRotate(matrix, eigenvectorsMat, n, pivotRow, pivotCol);
            jacobiIterCounter++;
        } while (jacobiIterCounter < MAX_JACOBI_ITER && diffOffNorm > EPSILON);
    }
    return eigenvectorsMat;
}

/* This function performs a single jacobi rotation. */
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
            /* Symmetry */
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

/* This function chooses the pivot index for the jacobi rotation
 *      - the max abs off diagonal element > 0. */
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

/* Build an n * n identity matrix. */
double **initIdentityMatrix(int n) {
    int i, j;
    double **matrix = (double **) alloc2DArray(n, n, sizeof(double),
                                               sizeof(double *), freeUsedMem);

    if (matrix != NULL) { /* Memory allocation fail */
        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                matrix[i][j] = i == j ? 1.0 : 0.0;
            }
        }
    }
    return matrix;
}

/* Sorting eigenvalues using qsort and comparator (makes it stable). */
Eigenvalue *sortEigenvalues(double **a, int n) {
    int i;
    Eigenvalue *eigenvalues = myAlloc(NULL, n * sizeof(Eigenvalue));

    if (eigenvalues != NULL) { /* Memory allocation fail */
        for (i = 0; i < n; ++i) {
            eigenvalues[i].value = a[i][i];
            eigenvalues[i].vector = i; /* The original order after the jacobi algorithm */
        }

        qsort(eigenvalues, n, sizeof(Eigenvalue), cmpEigenvalues);
    }
    return eigenvalues;
}

/* Comparator function for the eigenvalues qsort. */
int cmpEigenvalues (const void *p1, const void *p2) {
    const Eigenvalue *q1 = p1, *q2 = p2;

    if (q1->value > q2->value)
        return 1;
    else if (q1->value < q2->value)
        return -1;
    return (q1->vector - q2->vector); /* Keeps qsort comparator stable */
}

/*******************************************************************************
***************************** Memory Allocation ********************************
*******************************************************************************/

/* The function allocates memory for any dynamic memory needed. */
void *myAlloc(void *effectiveUsedMem, size_t size) {
    /* Get the "real" head of Block - with the pointers, if not NULL */
    void *usedMem = effectiveUsedMem != NULL ?
                    (void *)((char *)effectiveUsedMem - SIZE_OF_VOID_2PTR * 2) : NULL;
    void *blockMem;
    void **blockMemPlusPtr = (void **)realloc(usedMem, size + SIZE_OF_VOID_2PTR * 2);
    if(blockMemPlusPtr == NULL) return NULL; /* Memory allocation fail */

    /* blockMemPlusPtr[0] == prev block pointer, blockMemPlusPtr[1] == next pointer */
    blockMem = (void *)((char *)blockMemPlusPtr + SIZE_OF_VOID_2PTR * 2);
    if (effectiveUsedMem == NULL) { /* New Allocation */
        /* Set ptr to the prev/next dynamic allocated memory block */
        blockMemPlusPtr[0] = NULL;
        if (headOfMemList != NULL) { /* Not empty list */
            blockMemPlusPtr[1] = headOfMemList;
            ((void **)headOfMemList)[0] = blockMemPlusPtr;
        } else
            blockMemPlusPtr[1] = NULL;
        headOfMemList = blockMemPlusPtr; /* Update head of memory list */
    } else { /* Reallloc */
        /* Update pointers */
        if (usedMem != blockMemPlusPtr) { /* Block changed location in memory */
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

/* The function builds a 2 dimension array (matrix) using "myAlloc" function. */
void **alloc2DArray(int rows, int cols, size_t basicSize, size_t basicPtrSize,
                    void *recycleMemBlock) {
    int i;
    void *blockMem, **matrix;
    /* Reallocate block of memory - use extra space at the end for row pointers */
    blockMem = myAlloc(recycleMemBlock, rows * cols * basicSize + rows * basicPtrSize);
    if (blockMem == NULL) return NULL; /* Memory allocation fail */
    matrix = (void **) ((char *)blockMem + rows * cols * basicSize);

    for (i = 0; i < rows; ++i) {
        /* Set matrix to point to head of rows */
        *((void **)((char *)matrix + i * basicPtrSize)) =
                (void *) (((char *) blockMem) + i * cols * basicSize);
    }
    return matrix;
}

/* This function free unnecessary memory and keep the order of the memory list. */
void myFree(void *effectiveBlockMem) {
    void **blockMem;
    if (effectiveBlockMem == NULL) /* NULL pointer - Do nothing */
        return;

    /* Get the "real" head of Block - with the pointers */
    blockMem = (void **)((char *)effectiveBlockMem - SIZE_OF_VOID_2PTR * 2);
    /* Unlink/delete from the list - update pointers */
    if(blockMem[0] != NULL) {
        /* Set prev's next to current next */
        ((void **)blockMem[0])[1] = blockMem[1];
    } else {
        headOfMemList = blockMem[1];
    }
    if(blockMem[1] != NULL) {
        /* Set next's prev to current prev */
        ((void **)blockMem[1])[0] = blockMem[0];
    }
    free(blockMem);
}

/* This function free all memory allocated at runtime. */
void freeAllMemory() {
    void **currBlock = headOfMemList, **nextBlock;

    while (currBlock != NULL) {
        nextBlock = currBlock[1];
        free(currBlock);
        currBlock = nextBlock;
    }
    headOfMemList = NULL; /* Empty list */
}

/*******************************************************************************
**************************** Printing results **********************************
*******************************************************************************/

/* This function print matrix in csv format. */
void printMatrix(double **matrix, int rows, int cols) {
    int i, j;
    double value;

    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            if (j > 0)
                printf("%c", COMMA_CHAR);
            value = matrix[i][j];
            value = NegZero(value); /* Avoid -0.0000 presentation */
            /* Print with an accuracy of desired digits after the decimal point */
            printf(PRINT_FORMAT, value);
        }
        printf("\n");
    }
}

/* The function prints the jacobi result in csv format */
void printJacobi(double **a, double **v, int n) {
    int i;
    double value;

    for (i = 0; i < n; ++i) {
        if (i != 0)
            printf("%c", COMMA_CHAR);
        value = a[i][i];
        value = NegZero(value); /* Avoid -0.0000 presentation */
        /* Print with an accuracy of desired digits after the decimal point */
        printf(PRINT_FORMAT, value);
    }
    printf("\n");
    printMatrix(v, n, n); /* Print eigenvectors matrix v == V^T */
}

/*******************************************************************************
************************* Auxiliary Functions **********************************
*******************************************************************************/

/* This function read cmd-line arguments, validate and assign them the matching variables. */
void validateAndAssignInput(int argc, char **argv, int *k, GOAL *goal, char **filenamePtr) {
    char *nextCh;

    if (argc >= REQUIRED_NUM_OF_ARGUMENTS) {
        *goal = str2enum(argv[GOAL_ARGUMENT]);
        *filenamePtr = argv[REQUIRED_NUM_OF_ARGUMENTS - 1];
        if (*goal < NUM_OF_GOALS) {
            if (*goal != spk) {
                *k = 0; /* K is unnecessary */
                return;
            } else {
                /* k greater than zero and the conversion succeeded, valid goal */
                *k = strtol(argv[K_ARGUMENT], &nextCh, 10);
                if (*k >= 0 && *nextCh == END_OF_STRING)
                    return;
            }
        }
    }
    /* Invalid input from user */
    printf(INVALID_INPUT_MSG);
    exit(0);
}

/* This function convert String to enum representation. */
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

/* The function read from csv format file (extension .txt/.csv) into matrix. */
double **readDataFromFile(int *rows, int *cols, char *fileName, GOAL goal) {
    int counter, maxLen;
    char c;
    double value;
    FILE *file;
    double **matrix, *dataBlock;

    maxLen = goal != jacobi ? MAX_FEATURES : MAX_DATAPOINTS;
    dataBlock = (double *) myAlloc(NULL, maxLen * sizeof(double));
    MyAssert(dataBlock != NULL); /* Memory allocation fail */
    file = fopen(fileName, "r");
    MyAssert(file != NULL); /* File opened successfully */
    calcDim(cols, file, dataBlock);

    maxLen = goal != jacobi ? MAX_DATAPOINTS : *cols;
    /* Reallocate memory to hold the data */
    dataBlock = (double *) myAlloc(dataBlock ,maxLen * (*cols) * sizeof(double));
    MyAssert(dataBlock != NULL); /* Memory allocation fail */

    counter = *cols;
    while (fscanf(file, "%lf%c", &value, &c) != EOF) {
        dataBlock[counter++] = value;
    }
    MyAssert(fclose(file) != EOF); /* File closed successfully */

    *rows = counter / *cols;
    /* Make it 2D array */
    matrix = (double **) alloc2DArray(*rows, *cols, sizeof(double),
                                      sizeof(double *), dataBlock);
    MyAssert(matrix != NULL); /* Memory allocation fail */
    return matrix;
}

/* This function calculates and assign the Data's number of features,
 *      while reading the first line of the file. */
void calcDim(int *dimension, FILE *file, double *firstLine) {
    char c;
    double value;
    *dimension = 0;
    do {
        fscanf(file, "%lf%c", &value, &c);
        firstLine[(*dimension)++] = value;
    } while (c == COMMA_CHAR);
}