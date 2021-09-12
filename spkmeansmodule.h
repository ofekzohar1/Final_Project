#ifndef FINAL_PROJECT_SPKMEANSMODULE_H
#define FINAL_PROJECT_SPKMEANSMODULE_H
/* This header contains macros, constants and functions used to link between
 * C to python */

/*******************************************************************************
********************************* Macros ***************************************
*******************************************************************************/
/* Custom logical assert macro - free memory and return NULL (error in python) */
#define MyAssert(exp) \
if (!(exp)) {         \
freeAllMemory();        \
if (!PyErr_Occurred()) { PyErr_NoMemory(); }  \
/* If none exception reported, raise memory exception */  \
return NULL;            \
}

/* Custom typeErr raising */
#define MyPy_TypeErr(x, y) \
PyErr_Format(PyExc_TypeError, "%s type is required (got type %s)", x ,Py_TYPE(y)->tp_name)

/*******************************************************************************
**************************** Functions Declaration *****************************
*******************************************************************************/

/** The C-function that implements the Python function calc_mat.
 * Gets vectors list as matrix and return matrix calculated according to the
 *      goal provided using 'dataAdjustmentMatrices' C function in "spkmeans.h".
 * @param args - Arguments from python:
 *      vectors list, goal, n_clusters (k), n_features, n_vectors (N)
 * @return Matrix (python list of lists): 'spk' - T, 'wam' - W, 'ddg' - D, 'lnorm' - Lnorm
 */
static PyObject *calc_mat_connect(PyObject *self, PyObject *args);

/** The C-function that implements the Python function kmeans.
 * Gets vectors list as matrix and initial centroids list, runs kmeans clustering
 *      using 'kMeans' C function in "spkmeans.h".
 * @param args - Arguments from python:
 *      vectors list (matrix), n_vectors (N), n_features, n_clusters (k),
 *          list of indexes to be the initial clusters centroids
 * @return Final clusters' centroids (python list of lists) and vectors labeling
 *      (vector to cluster, list) as tuple
 */
static PyObject *kmeans_connect(PyObject *self, PyObject *args);

/** The C-function that implements the Python function jacobi.
 * Gets symmetrical matrix, runs jacobi diagonalizing algorithm using
 *      'jacobiAlgorithm' C function in "spkmeans.h".
 * @param args - Arguments from python: symmetrical matrix, dimension (n)
 * @return The transposed eigenvectors matrix (P^T as List of lists) and
 *      eigenvalues list packed in a tuple.
 */
static PyObject *jacobi_connect(PyObject *self, PyObject *args);

/*
 * This function Gets python type list of lists (float) and convert it to C double matrix.
 * If an error occur return NULL.
 */
double **pyLOLToCMat(PyObject *pyListOfLists, int rows, int cols);

/*
 * This function Gets python int type list and convert it to C array.
 * If an error occurs return NULL.
 */
int *pyIntListToCArray(PyObject *pyIntList, int len);

/*
 * This function Gets C double array, build and return python type list (float).
 * If an error occur return NULL.
 */
PyObject *cArrToPythonList(double *array, int len);

/*
 * This function Gets C double array, build and return python type list (float).
 * If an error occur return NULL.
 */
PyObject *cMatToPyLOL(double **matrix, int rows, int cols);

/*
 * This function pack kmeans results into python tuple.
 * If an error occur return NULL.
 */
PyObject *kmeansResToPyObject(double **matrix, int rows, int cols, int numOfDatapoints);

/*
 * This function pack jacobi results into python tuple.
 * If an error occur return NULL.
 */
PyObject *jacobiResToPyObject(double **eigenvectorsMat, double *eigenvalues, int n);

#endif /* FINAL_PROJECT_SPKMEANSMODULE_H */
