#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"

#define MyAssert(exp) \
if (!(exp)) {         \
freeAllMemory();                      \
return NULL;   \
}

/* Functions declaration */
static PyObject *calc_mat_connect(PyObject *self, PyObject *args);
static PyObject *kmeans_connect(PyObject *self, PyObject *args);
static PyObject *jacobi_connect(PyObject *self, PyObject *args);
PyObject *cMatToPyLOL(double **matrix, int rows, int cols);
double **pyLOLToCMat(PyObject *pyListOfLists, int rows, int cols);
int *pyIntListToCArray(PyObject *pyIntList, int len);
PyObject *cArrToPythonList(double *array, int len);
PyObject *kmeansResToPyObject(double **matrix, int rows, int cols, int numOfDatapoints);
PyObject *jacobiResToPyObject(double **eigenvectorsMat, double *eigenvalues, int n);

/*
 * This array tells Python what methods this module has.
 * We will use it in the next structure
 */
static PyMethodDef _method[] = {
        {"calc_mat",                      /* the Python method name that will be used */
         (PyCFunction) calc_mat_connect, /* the C-function that implements the Python function and returns static PyObject*  */
         METH_VARARGS,   /* flags indicating parametersaccepted for this function */
         PyDoc_STR("Return calculated matrix (wMat/ddgMat/Lnorm/tMat) according to the goal provided."
                   "\n Spk goal returns tMat.")},      /*  The docstring for the function (PyDoc_STR("")) */
        {"jacobi",                      /* the Python method name that will be used */
         (PyCFunction) jacobi_connect, /* the C-function that implements the Python function and returns static PyObject*  */
         METH_VARARGS,   /* flags indicating parametersaccepted for this function */
         PyDoc_STR("Run Jacobi's algorithm on a symmetric matrix."
                   "\nReturn the eigenvectors matrix and list of eigenvalues.")},      /*  The docstring for the function (PyDoc_STR("")) */
        {"kmeans",                      /* the Python method name that will be used */
         (PyCFunction) kmeans_connect, /* the C-function that implements the Python function and returns static PyObject*  */
         METH_VARARGS,   /* flags indicating parametersaccepted for this function */
         PyDoc_STR("Run KMeans's algorithm. Return the final centroids and vectors labeling.")},      /*  The docstring for the function (PyDoc_STR("")) */
         {NULL, NULL, 0, NULL}        /* The is a sentinel. Python looks for this entry to know that all
                                       of the functions for the module have been defined. */
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "spkmeans", /* name of module */
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
PyInit_spkmeans(void) {
    return PyModule_Create(&moduledef);
}

/*
* This actually defines the kMeans function using a wrapper C API function
* The wrapping function needs a PyObject* self argument.
* This is a requirement for all functions and methods in the C API.
* It has input PyObject *args from Python.
*/
static PyObject *calc_mat_connect(PyObject *self, PyObject *args) {
    PyObject *pyListOfLists, *pyResult;
    int k, dimension, numOfDatapoints, cols;
    double **datapointsArray, **calcMat;
    char *strGoal;
    GOAL goal;
    headOfMemList= NULL, freeUsedMem = NULL;

    MyAssert(PyArg_ParseTuple(args, "Osiii",&pyListOfLists, &strGoal, &k, &dimension, &numOfDatapoints));
    /* Type error - not in correct format */

    goal = str2enum(strGoal);
    MyAssert(goal != NUM_OF_GOALS); /* Valid goal */
    datapointsArray = pyLOLToCMat(pyListOfLists, numOfDatapoints, dimension);
    MyAssert(datapointsArray != NULL);
    calcMat = dataAdjustmentMatrices(datapointsArray, goal, &k, dimension, numOfDatapoints);
    MyAssert(calcMat != NULL);
    if (goal == spk)
        cols = k;
    else
        cols = numOfDatapoints;
    pyResult = cMatToPyLOL(calcMat, numOfDatapoints, cols);

    freeAllMemory();
    return pyResult;
}
/*
* This connects the kMeans function using a wrapper C API function
* The wrapping function needs a PyObject* self argument.
* It has input PyObject *args from Python.
*/
static PyObject *kmeans_connect(PyObject *self, PyObject *args) {
    PyObject *pyListOfLists, *pyResult, *pyListOfIndexes;
    int k, dimension, numOfDatapoints, *firstCentralIndexes;
    double **datapointsArray, **calcMat;
    headOfMemList= NULL, freeUsedMem = NULL;

    MyAssert(PyArg_ParseTuple(args, "OiiiO",&pyListOfLists, &numOfDatapoints, &dimension, &k, &pyListOfIndexes));
    /* Type error - not in correct format */

    datapointsArray = pyLOLToCMat(pyListOfLists, numOfDatapoints, dimension);
    firstCentralIndexes = pyIntListToCArray(pyListOfIndexes, k);
    MyAssert(datapointsArray != NULL && firstCentralIndexes != NULL);
    calcMat = kMeans(datapointsArray, numOfDatapoints, dimension, k, firstCentralIndexes, MAX_KMEANS_ITER);
    MyAssert(calcMat != NULL);
    pyResult = kmeansResToPyObject(calcMat, k, dimension, numOfDatapoints);

    freeAllMemory();
    return pyResult;
}

static PyObject *jacobi_connect(PyObject *self, PyObject *args) {
    PyObject *pyListOfLists, *pyResult;
    int i, n;
    double **eigenvectorsMat, **matrix;
    headOfMemList= NULL, freeUsedMem = NULL;

    MyAssert(PyArg_ParseTuple(args, "Oi", &pyListOfLists, &n));
    matrix = pyLOLToCMat(pyListOfLists, n, n);
    MyAssert(matrix != NULL);
    eigenvectorsMat = jacobiAlgorithm(matrix, n);
    MyAssert(eigenvectorsMat != NULL);
    for (i = 1; i < n; ++i) {
        matrix[0][i] = matrix[i][i];
    }
    pyResult = jacobiResToPyObject(eigenvectorsMat, matrix[0], n);

    freeAllMemory();
    return pyResult;
}

int *pyIntListToCArray(PyObject *pyIntList, int len) {
    Py_ssize_t i;
    int *array, value;

    array = (int *) myAlloc(freeUsedMem, len * sizeof(int));
    if (array != NULL) {
        for (i = 0; i < len; ++i) {
            value = (int) PyLong_AsLong(PyList_GetItem(pyIntList, i));
            if (PyErr_Occurred()) {
                return NULL; /* Casting error to int */
            }
            array[i] = value;
        }
    }
    return array;
}

PyObject *cArrToPythonList(double *array, int len) {
    Py_ssize_t i;
    PyObject *pyList, *pyValue;
    pyList = PyList_New(len);
    if(pyList != NULL) {
        for (i = 0; i < len; ++i) {
            pyValue = PyFloat_FromDouble(array[i]);
            if (pyValue == NULL || PyList_SetItem(pyList, i, pyValue)) {
                Py_DecRef(pyList);
                return NULL; /* Set error */
            }
        }
    }
    return pyList;
}

double **pyLOLToCMat(PyObject *pyListOfLists, int rows, int cols) {
    Py_ssize_t i, j;
    double **matrix, value;
    PyObject *pyList, *pyValue;

    MyAssert(PyList_Check(pyListOfLists)); /* Is a list */
    /* Allocate memory for matrix */
    matrix = (double **) alloc2DArray(rows, cols, sizeof(double), sizeof(double *), freeUsedMem);
    if (matrix != NULL) {
        for (i = 0; i < rows; ++i) {
            pyList = PyList_GetItem(pyListOfLists, i);
            MyAssert(PyList_Check(pyList)); /* Is a list */
            for (j = 0; j < cols; ++j) {
                pyValue = PyList_GetItem(pyList, j);
                value = PyFloat_AsDouble(pyValue);
                MyAssert(!PyErr_Occurred()); /* Check for an error */
                matrix[i][j] = value;
            }
        }
    }
    return matrix;
}

PyObject *cMatToPyLOL(double **matrix, int rows, int cols) {
    Py_ssize_t i, j;
    PyObject *pyLOL, *pyList, *pyValue;
    pyLOL = PyList_New(rows);
    if(pyLOL != NULL) {
        for (i = 0; i < rows; ++i) {
            pyList = PyList_New(cols);
            if (pyList == NULL) {
                Py_DecRef(pyLOL);
                return NULL; /* If NULL - alloc fail */
            }
            for (j = 0; j < cols; ++j) {
                pyValue = PyFloat_FromDouble(matrix[i][j]);
                if (pyValue == NULL || PyList_SetItem(pyList, j, pyValue)) {
                    Py_DecRef(pyLOL);
                    Py_DecRef(pyList);
                    return NULL; /* Set error */
                }
            }
            if (PyList_SetItem(pyLOL, i, pyList)) {
                Py_DecRef(pyLOL);
                return NULL; /* Set error */
            }
        }
    }
    return pyLOL;
}

PyObject *kmeansResToPyObject(double **matrix, int rows, int cols, int numOfDatapoints) {
    PyObject *pyCentroidsMat, *pyVecLabeling;

    pyCentroidsMat = cMatToPyLOL(matrix, rows, cols);
    pyVecLabeling = cArrToPythonList(matrix[rows], numOfDatapoints);
    if (pyCentroidsMat == NULL || pyVecLabeling == NULL)
        return NULL; /* Error */

    return PyTuple_Pack(2, pyCentroidsMat, pyVecLabeling);
}

PyObject *jacobiResToPyObject(double **eigenvectorsMat, double *eigenvalues, int n) {
    PyObject *pyEigenvectorsMat, *pyEigenvalues;

    pyEigenvectorsMat = cMatToPyLOL(eigenvectorsMat, n, n);
    pyEigenvalues = cArrToPythonList(eigenvalues, n);
    if (pyEigenvectorsMat == NULL || pyEigenvalues == NULL)
        return NULL; /* Error */

    return PyTuple_Pack(2, pyEigenvectorsMat, pyEigenvalues);
}