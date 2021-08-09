#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define SQ(x) ((x)*(x))
#define EPSILON 0.001
#define MAX_JACOBI_ITER 100
/* Custom made python type error */
#define MyPy_TypeErr(x, y) \
PyErr_Format(PyExc_TypeError, "%s type is required (got type %s)", x ,Py_TYPE(y)->tp_name) \

typedef struct {
    double *prevCentroid;
    double *currCentroid;
    int counter; /* Number of vectors (datapoints) in cluster */
} Cluster;

int initVectorsArray(double ***vectorsArrayPtr, const int *numOfVectors, const int *dimension, PyObject *pyVectorsList); /* Insert vectors into an array */
int initClusters(Cluster **clustersArrayPtr, double **vectorsArray, const int *k, const int *dimension, const int *firstCentralIndexes); /* Initialize empty clusters array */
double vectorsNorm(const double *vec1, const double *vec2, const int *dimension); /* Calculate the norm between 2 vectors */
int findMyCluster(double *vec, Cluster *clustersArray, const int *k, const int *dimension); /* Return the vector's closest cluster (in terms of norm) */
void assignVectorsToClusters(double **vectorsArray, Cluster *clustersArray, const int *k, const int *numOfVectors, const int *dimension); /* For any vector assign to his closest cluster */
int recalcCentroids(Cluster *clustersArray, const int *k, const int *dimension); /* Recalculate clusters' centroids, return number of changes */
void initCurrCentroidAndCounter(Cluster *clustersArray, const int *k, const int *dimension); /* Set curr centroid to prev centroid and reset the counter */
PyObject *buildPyListCentroids(Cluster *clustersArray, const int *k, const int *dimension); /* Print clusters' final centroids */
void freeMemoryVectorsClusters(double **vectorsArray, Cluster *clustersArray, const int *k, int *firstCentralIndexes); /* Free the allocated memory */
void jacobiAlgorithm(double *matrix, double *eigenvectorsMat, int n);
void initIdentityMatrix(double *matrix, int n);
int EigengapHeuristicKCalc (double *a, int n);

static PyObject* fit(int k, int maxIter, int dimension, int numOfVectors, double **vectorsArray, int* firstCentralIndexes, Cluster **clustersArrayPtr) {
    int i, changes;

    /* Initialize clusters arrays */
    if (initClusters(clustersArrayPtr, vectorsArray, &k, &dimension, firstCentralIndexes)) /* 0 (false) on success, true on malloc fail */
        return PyErr_NoMemory();

        for (i = 0; i < maxIter; ++i) {
            initCurrCentroidAndCounter(*clustersArrayPtr, &k, &dimension); /* Update curr centroid to prev centroid and reset the counter */
            assignVectorsToClusters(vectorsArray, *clustersArrayPtr, &k, &numOfVectors, &dimension);
            changes = recalcCentroids(*clustersArrayPtr, &k, &dimension); /* Calculate new centroids */
            if (changes == 0) { /* Centroids stay unchanged in the current iteration */
                break;
            }
        }

        return buildPyListCentroids(*clustersArrayPtr, &k, &dimension);
}

/*
 * This actually defines the fit function using a wrapper C API function
 * The wrapping function needs a PyObject* self argument.
 * This is a requirement for all functions and methods in the C API.
 * It has input PyObject *args from Python.
 */
static PyObject* fit_connect(PyObject *self, PyObject *args) {
    Py_ssize_t i;
    PyObject *pyCentralsList, *pyVectorsList, *resault = NULL;
    int k, maxIter, dimension, numOfVectors, *firstCentralIndexes;
    double **vectorsArray = NULL; /* Default value - not allocated */
    Cluster *clustersArray = NULL; /* Default value - not allocated */

    if (!PyArg_ParseTuple(args, "iiiiOO", &k, &maxIter, &dimension, &numOfVectors, &pyCentralsList, &pyVectorsList))
        return NULL; /* Type error - not in correct format */
        if (!PyList_Check(pyCentralsList)) {
            MyPy_TypeErr("List", pyCentralsList);
            return NULL; /* Type error - not a python List */
        }
        if (!PyList_Check(pyVectorsList))
        {
            MyPy_TypeErr("List", pyVectorsList);
            return NULL; /* Type error - not a python List */
        }

        firstCentralIndexes = (int *) malloc(k * sizeof(int));
        if (firstCentralIndexes == NULL) {
            resault = PyErr_NoMemory(); /* Memory allocation error */
            goto end;
        }
        for (i = 0; i < k; i++) {
            firstCentralIndexes[i] = (int)PyLong_AsLong(PyList_GetItem(pyCentralsList, i));
            if (PyErr_Occurred()) {
                goto end; /* Casting error to int */
            }
        }
        if (initVectorsArray(&vectorsArray, &numOfVectors, &dimension, pyVectorsList)) /* return 0 (false) on success, true on error */
            goto end; /* On any error from initVectorsArray() */
            resault = fit(k, maxIter, dimension, numOfVectors, vectorsArray, firstCentralIndexes, &clustersArray);

            end:
    freeMemoryVectorsClusters(vectorsArray, clustersArray, &k, firstCentralIndexes); /* Free memory */
    return resault;
}

/*
 * This array tells Python what methods this module has.
 * We will use it in the next structure
 */
static PyMethodDef _method[] = {
        {"fit",                      /* the Python method name that will be used */
         (PyCFunction) fit_connect, /* the C-function that implements the Python function and returns static PyObject*  */
         METH_VARARGS,   /* flags indicating parametersaccepted for this function */
         NULL},      /*  The docstring for the function (PyDoc_STR("")) */
         {NULL, NULL, 0, NULL}        /* The is a sentinel. Python looks for this entry to know that all
                                       of the functions for the module have been defined. */
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
        _method /* the PyMethodDef array from before containing the methods of the extension */
};

/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the module’s initialization function.
 * The initialization function must be named PyInit_name(), where name is the name of the module and should match
 * what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file
 */
PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&moduledef);
}

int initVectorsArray(double ***vectorsArrayPtr, const int *numOfVectors, const int *dimension, PyObject *pyVectorsList) {
    Py_ssize_t i, j;
    double *matrix;
    PyObject *vector, *comp;
    /* Allocate memory for vectorsArrayPtr */
    matrix = (double *) malloc((*numOfVectors) * ((*dimension) + 1) * sizeof(double));
    *vectorsArrayPtr = malloc((*numOfVectors) * sizeof(double *));
    if (matrix == NULL || (*vectorsArrayPtr) == NULL){
        if (matrix != NULL)
            free(matrix); /* Free matrix if exist */
            PyErr_SetNone(PyExc_MemoryError);
            return 1; /* Memory allocation error */
    }

    for (i = 0; i < *numOfVectors; ++i) {
        (*vectorsArrayPtr)[i] = matrix + i * ((*dimension) + 1); /* Set VectorsArray to point to 2nd dimension array */
        vector = PyList_GetItem(pyVectorsList, i);
        if (!PyList_Check(vector)) {
            MyPy_TypeErr("List", vector);
            free(*vectorsArrayPtr);
            free(matrix);
            *vectorsArrayPtr = NULL;
            return 1; /* Type error - not a python List */
        }
        for (j = 0; j < *dimension; ++j) {
            comp = PyList_GetItem(vector, j);
            (*vectorsArrayPtr)[i][j] = PyFloat_AsDouble(comp);
            if (PyErr_Occurred()) {
                free(*vectorsArrayPtr);
                free(matrix);
                *vectorsArrayPtr = NULL;
                return 1; /* Cast error to double */
            }
        }
    }
    return 0; /* Success */
}

int initClusters(Cluster **clustersArrayPtr, double **vectorsArray, const int *k, const int *dimension, const int *firstCentralIndexes) {
    int i, j, mallocFails = 0;
    /* Allocate memory for clustersArrayPtr */
    *clustersArrayPtr = (Cluster *) malloc((*k) * sizeof(Cluster));
    if ((*clustersArrayPtr) != NULL) {
        for (i = 0; i < *k; ++i) {
            (*clustersArrayPtr)[i].counter = 0;
            (*clustersArrayPtr)[i].prevCentroid = (double *) malloc((*dimension) * sizeof(double));
            (*clustersArrayPtr)[i].currCentroid = (double *) malloc((*dimension) * sizeof(double));
            if ((*clustersArrayPtr)[i].prevCentroid == NULL || (*clustersArrayPtr)[i].currCentroid == NULL) {
                mallocFails++;
                continue;
            }

            /* Assign the initial k vectors to their corresponding clusters according to the ones calculated in python */
            for (j = 0; j < *dimension && mallocFails == 0; ++j) {
                (*clustersArrayPtr)[i].currCentroid[j] = vectorsArray[firstCentralIndexes[i]][j];
            }
        }
    } else {
        mallocFails++;
    }
    return mallocFails; /* 0 on success, greater than 0 on fail */
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
    for (j = 1; j < *k; ++j) { /* Find the min norm == closest cluster */
        norm = vectorsNorm(vec, clustersArray[j].prevCentroid, dimension);
        if (norm < minNorm) {
            myCluster = j;
            minNorm = norm;
        }
    }
    return myCluster;
}

void assignVectorsToClusters(double **vectorsArray, Cluster *clustersArray, const int *k, const int *numOfVectors, const int *dimension) {
    int j, i, myCluster;
    double *vec;
    for (j = 0; j < *numOfVectors; ++j) {
        vec = vectorsArray[j];
        myCluster = findMyCluster(vectorsArray[j], clustersArray, k, dimension);
        vec[*dimension] = myCluster; /* Set vector's cluster to his closest */
        for (i = 0; i < *dimension; ++i) {
            clustersArray[myCluster].currCentroid[i] += vec[i]; /* Summation of the vectors Components */
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
            changes += cluster.prevCentroid[j] != cluster.currCentroid[j] ? 1 : 0; /* Count the number of changed centroids' components */
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

PyObject *buildPyListCentroids(Cluster *clustersArray, const int *k, const int *dimension) {
    Py_ssize_t i, j;
    PyObject *listOfCentrals, *central, *comp;
    listOfCentrals = PyList_New(*k);
    if (listOfCentrals != NULL) { /* If NULL alloc fail */
        for (i = 0; i < *k; ++i) {
            central = PyList_New(*dimension);
            if (central == NULL) {
                Py_DecRef(listOfCentrals);
                return NULL; /* If NULL alloc fail */
            }
            for (j = 0; j < *dimension; ++j) {
                comp = PyFloat_FromDouble(clustersArray[i].currCentroid[j]);
                if (comp == NULL || PyList_SetItem(central, j, comp)) {
                    Py_DecRef(listOfCentrals);
                    Py_DecRef(central);
                    Py_XDECREF(comp);
                    return NULL; /* Set error */
                }
            }
            if (PyList_SetItem(listOfCentrals, i, central)) {
                Py_DecRef(listOfCentrals);
                Py_DecRef(central);
                return NULL; /* Set error */
            }
        }
    }
    return listOfCentrals;
}

void freeMemoryVectorsClusters(double **vectorsArray, Cluster *clustersArray, const int *k, int *firstCentralIndexes) {
    int i;
    /* Free clusters */
    if (clustersArray != NULL) {
        for (i = 0; i < *k; ++i) {
            free(clustersArray[i].currCentroid);
            free(clustersArray[i].prevCentroid);
        }
    }
    free(clustersArray);
    free(firstCentralIndexes);

    /* Free vectors */
    if (vectorsArray != NULL)
        free(*vectorsArray);
    free(vectorsArray);
}

/*************************************
*********** Jacobi Algorithm *********
*************************************/

void pivotIndex(double *matrix, int n, int *pivotRow, int *pivotCol) {
    int i, j;
    double maxAbs = -1, tempValue;
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n ; ++j) {
            tempValue = fabs(matrix[j + i * n]);
            if (maxAbs < tempValue) {
                maxAbs = tempValue;
                *pivotRow = i;
                *pivotCol = j;
            }
        }
    }
}

void initIdentityMatrix(double *matrix, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        matrix[i + i * n] = 1.0;
    }
}

double jacobiRotate(double *a, double *v, int n, int i, int j) {
    double theta, t, c, s;
    double ij, ii, jj, ri, rj;
    int r;

    theta = a[j + j * n] - a[i + i * n]; // Ajj - Aii
    theta /= (2 * a[j + i * n]);
    t = 1.0 / (fabs(theta) + sqrt(SQ(theta) + 1.0));
    t = theta < 0.0 ? -t : t;
    c = 1.0 / sqrt(SQ(t) + 1.0);
    s = t * c;

    ii = a[i + i * n];
    jj = a[j + j * n];
    ij = a[j + i * n];
    a[j + i * n] = 0.0; // Aij = 0
    a[i + j * n] = 0.0;
    for (r = 0; r < n; r++) {
        if (r == i)
            // c^2 * Aii + s^2 * Ajj - 2scAij
            a[i + i * n] = SQ(c) * ii + SQ(s) * jj - 2 * s * c * ij;
        else if (r == j)
            // s^2 * Aii + c^2 * Ajj + 2scAij
            a[j + j * n] = SQ(s) * ii + SQ(c) * jj + 2 * s * c * ij;
        else { // r != j, i
            ri = a[i + r * n];
            rj = a[j + r * n];
            a[i + r * n] = c * ri - s * rj;
            a[j + r * n] = c * rj + s * ri;
            a[r + i * n] = a[i + r * n];
            a[r + j * n] = a[j + r * n];
        }

        // Update the eigenvector matrix
        ri = v[i + r * n];
        rj = v[j + r * n];
        v[i + r * n] = c * ri - s * rj;
        v[j + r * n] = c * rj + s * ri;
    }

    return 2 * SQ(ij); // offNormDiff: Off(A)^2 - Off(A')^2 = 2 * Aij^2
}

void sortEigenvaluesAndVectors(double *a, double *v, int n) {
    int i, j, minIndex;
    double temp, min;

    for (i = 0; i < n - 1; i++) {
        minIndex = i;
        min = a[i + i * n];
        for (j = i + 1; j < n; j++) {
            if (a[j + j * n] < min) {
                min = a[j + j * n];
                minIndex = j;
            }
        }

        if (i != minIndex) {
            a[minIndex + minIndex * n] = a[i + i * n];
            a[i + i * n] = min;
            for (j = 0; j < n; j++) {
                temp = v[i + j * n];
                v[i + j * n] = v[minIndex + j * n];
                v[minIndex + j * n] = temp;
            }
        }
    }
}

void jacobiAlgorithm(double *matrix, double *eigenvectorsMat, int n) {
    double diffOffNorm;
    int jacobiIterCounter, pivotRow, pivotCol;

    initIdentityMatrix(eigenvectorsMat, n);

    jacobiIterCounter = 0;
    do {
        pivotIndex(matrix, n, &pivotRow, &pivotCol);
        diffOffNorm = jacobiRotate(matrix, eigenvectorsMat, n, pivotRow, pivotCol);
        jacobiIterCounter++;
    } while (jacobiIterCounter < MAX_JACOBI_ITER && diffOffNorm >= EPSILON);

    sortEigenvaluesAndVectors(matrix, eigenvectorsMat, n);
}

int EigengapHeuristicKCalc (double *a, int n) {
    int i, maxIndex, m;
    double maxDelta, delta;

    m = n / 2;
    maxDelta = -1;
    maxIndex = -1;
    for (i = 0; i < m; i++) {
        delta = a[(i+1) + (i+1) * n] - a[i + i * n];
        if (maxDelta < delta) {
            maxDelta = delta;
            maxIndex = i;
        }
    }

    return maxIndex + 1;
}

void initTMatrix(double *v, double **t, int n, int k) {
    int i, j;
    double sumSqRow;

    for (i = 0; i < n; ++i) {
        sumSqRow = 0.0;
        for (j = 0; j < k; ++j) {
            sumSqRow += SQ(v[j + i * n]);
        }
        sumSqRow = 1.0 / sqrt(sumSqRow);
        for (j = 0; j < k; ++j) {
            t[i][j] = v[j + i * n] * sumSqRow;
        }
    }
}

/********* Ben's part *********/

/*
 * Builds a n*n matrix while n is the number of vectors
 * Assign the weighted factor as requested
 * Param - vectorsArray matrix, number of vectors and dimensions
 * Using sqrt and norm functions
 * Returns the weighted Matrix
 */
double **weightedMatrix(double **vectorsArray,const int *numOfVectors, const int *dimension)
{
    int i, j;
    double *matrix, **wMatrix, norm;
    matrix = (double *) malloc((*numOfVectors) * (*numOfVectors) * sizeof(double));
    assert(matrix != NULL);
    wMatrix = malloc((*numOfVectors) * sizeof(double *));
    assert(wMatrix != NULL);
    for (i = 0; i < *numOfVectors; ++i) {
        wMatrix[i] = matrix + i * ((*numOfVectors));
    }
    for (i=0;i<*numOfVectors;i++){
        wMatrix[i][i]=0;
        for (j = (i + 1);j<*numOfVectors;j++){
            norm = sqrt(vectorsNorm(vectorsArray[i], vectorsArray[j], dimension));
            wMatrix[i][j] = exp(-1*0.5*norm); /*need to switch exp*/
            wMatrix[j][i] = wMatrix[i][j];
        }
    }
    return wMatrix;
}
/*
 *Build the Diagonal Degree Matrix from the Weighted Adjacency Matrix
 *returns a pointer to the Diagonal Degree Matrix
 * */
double *dMatrix(double **wMatrix,const int *numOfVectors, const int *dimension)
{
    int i, j;
    double  *dMatrix, sum;
    dmatrix = (double *) malloc((*numOfVectors) * sizeof(double));
    assert(dMatrix != NULL);
    for (i=0;i<*numOfVectors;i++){
        sum=0;
        for (j=0;j<*numOfVectors;j++){
            sum+= wMatrix[i][j];
        }
        dMatrix[i] = sum;
    }
    return dMatrix;
}

/*
 * Builds The Normalized Graph Laplacian from the weighted matrix and the degree matrix
 * returns a pointer to the Normalized Graph Laplacian Matrix*/
double **lPlacian(double **wMatrix,double *dMatrix, const int *numOfVectors)
{
    int i,j;
    double *lMatrix;
    lMatrix = (double *) malloc((*numOfVectors) * (*numOfVectors) * sizeof(double));
    assert(lMatrix != NULL);
    for (i=0;i<*numOfVectors;i++){
        dMatrix[i] = 1/sqrt(dMatrix[i]);
    }
    for (i=0;i<*numOfVectors;i++){
        for (j=0;j<*numOfVectors;j++){
            if(i==j){
                lMatrix[j+i*n] = 1 - dMatrix[i]*dMatrix[j]*wMatrix[i][j];
            }
            else{
                lMatrix[j+i*n] = -1*dMatrix[i]*dMatrix[j]*wMatrix[i][j];
            }
        }
    }
    return lMatrix;
}