#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeansmodule.h" /* Macros and functions declarations */
#include "spkmeans.h" /* spk's public interface functions and macros */

/**********************************
********* Module settings *********
**********************************/

/* This array tells Python what methods this module has */
static PyMethodDef method[] = {
        {"calc_mat", /* the Python method name that will be used */
         (PyCFunction) calc_mat_connect,
         /* the C-function that implements the Python function and returns static PyObject*  */
         METH_VARARGS, /* flags indicating parameters are accepted for this function */
         /*  The docstring for the function (PyDoc_STR("")) */
         PyDoc_STR("Return calculated matrix (wMat/ddgMat/Lnorm/tMat) "
                   "according to the goal provided.\n Spk goal returns tMat.")},

        {"jacobi", (PyCFunction) jacobi_connect, METH_VARARGS,
         PyDoc_STR("Run Jacobi's algorithm on a symmetric matrix."
                   "\nReturn the eigenvectors matrix and list of eigenvalues.")},

        {"kmeans", (PyCFunction) kmeans_connect, METH_VARARGS,
         PyDoc_STR("Run KMeans algorithm. Return the final centroids and vectors labeling.")},

         {NULL, NULL, 0, NULL} /* This is a sentinel */
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "spkmeansmodule", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,  /* size of per-interpreter state of the module,
                    * or -1 if the module keeps state in global variables. */
        method /* the PyMethodDef array from before containing the methods of the extension */
};

/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the
 *      module’s initialization function.
 * The initialization function must be named PyInit_name(), where name is the
 *      name of the module and should match what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file
 */
PyMODINIT_FUNC
PyInit_spkmeansmodule(void) {
    return PyModule_Create(&moduledef);
}

/**********************************
******** Connect functions ********
**********************************/

/* The C-function that implements the Python function calc_mat. */
static PyObject *calc_mat_connect(PyObject *self, PyObject *args) {
    PyObject *pyListOfLists, *pyResult;
    int k, dimension, numOfDatapoints, cols;
    double **datapointsArray, **calcMat;
    char *strGoal;
    GOAL goal;
    headOfMemList= NULL, freeUsedMem = NULL; /* Init C memory containers */

    MyAssert(PyArg_ParseTuple(args, "Osiii", &pyListOfLists, &strGoal, &k,
                              &dimension, &numOfDatapoints));
    /* Assert fail == Type error - not in correct format */

    goal = str2enum(strGoal);
    if (goal == NUM_OF_GOALS) { /* Not Valid goal */
        PyErr_SetString(PyExc_ValueError, "Not valid goal.");
        return NULL;
    }
    /* Convert python matrix to C matrix */
    datapointsArray = pyLOLToCMat(pyListOfLists, numOfDatapoints, dimension);
    MyAssert(datapointsArray != NULL);
    /* Calc matrix according to the goal provided */
    calcMat = dataAdjustmentMatrices(datapointsArray, goal, &k, dimension,
                                     numOfDatapoints);
    MyAssert(calcMat != NULL);

    if (goal == spk)
        cols = k; /* T's dimensions - N x K */
    else
        cols = numOfDatapoints; /* Otherwise - N x N */
    /* Convert result back to python type List of lists */
    pyResult = cMatToPyLOL(calcMat, numOfDatapoints, cols);
    MyAssert(pyResult != NULL);

    freeAllMemory();
    return pyResult;
}

/* The C-function that implements the Python function kmeans. */
static PyObject *kmeans_connect(PyObject *self, PyObject *args) {
    PyObject *pyListOfLists, *pyResult, *pyListOfIndexes;
    int k, dimension, numOfDatapoints, *firstCentralIndexes;
    double **datapointsArray, **calcMat;
    headOfMemList= NULL, freeUsedMem = NULL; /* Init C memory containers */

    MyAssert(PyArg_ParseTuple(args, "OiiiO",&pyListOfLists, &numOfDatapoints,
                              &dimension, &k, &pyListOfIndexes));
    /* Assert fail == Type error - not in correct format */

    /* Convert python types to C types */
    datapointsArray = pyLOLToCMat(pyListOfLists, numOfDatapoints, dimension);
    firstCentralIndexes = pyIntListToCArray(pyListOfIndexes, k);
    MyAssert(datapointsArray != NULL && firstCentralIndexes != NULL);
    /* KMeans clustering using 'kmeans' implementation in C */
    calcMat = kMeans(datapointsArray, numOfDatapoints, dimension, k,
                     firstCentralIndexes, MAX_KMEANS_ITER);
    MyAssert(calcMat != NULL);
    /* Convert result back to python type - tuple (LOL, List) */
    pyResult = kmeansResToPyObject(calcMat, k, dimension, numOfDatapoints);
    MyAssert(pyResult != NULL);

    freeAllMemory();
    return pyResult;
}

/* The C-function that implements the Python function jacobi. */
static PyObject *jacobi_connect(PyObject *self, PyObject *args) {
    PyObject *pyListOfLists, *pyResult;
    int i, n;
    double **eigenvectorsMat, **matrix;
    headOfMemList= NULL, freeUsedMem = NULL; /* Init C memory containers */

    MyAssert(PyArg_ParseTuple(args, "Oi", &pyListOfLists, &n));
    /* Assert fail == Type error - not in correct format */

    /* Convert python types to C types */
    matrix = pyLOLToCMat(pyListOfLists, n, n);
    MyAssert(matrix != NULL);
    /* Jacobi algorithm using 'jacobiAlgorithm' implementation in C */
    eigenvectorsMat = jacobiAlgorithm(matrix, n);
    MyAssert(eigenvectorsMat != NULL);
    for (i = 1; i < n; ++i) {
        /* Order the eigenvalues list in the first row of the diag matrix */
        matrix[0][i] = matrix[i][i];
    }
    /* Convert result back to python type - tuple (LOL, List) */
    pyResult = jacobiResToPyObject(eigenvectorsMat, matrix[0], n);
    MyAssert(pyResult != NULL);

    freeAllMemory();
    return pyResult;
}

/***********************************
** C <-> python convert functions **
***********************************/

/* This function Gets python int type list and convert it to C array. */
int *pyIntListToCArray(PyObject *pyIntList, int len) {
    Py_ssize_t i;
    int *array, value;
    PyObject *pyValue;

    if (!PyList_Check(pyIntList)) { /* Not a list */
        MyPy_TypeErr("list", pyIntList);
        return NULL;
    }
    array = (int *) myAlloc(freeUsedMem, len * sizeof(int));
    if (array != NULL) { /* Memory allocation fail */
        for (i = 0; i < len; ++i) {
            pyValue = PyList_GetItem(pyIntList, i);
            value = (int) (pyValue != NULL ? PyLong_AsLong(pyValue) : EOF);
            if (PyErr_Occurred()) {
                return NULL; /* Casting error to int */
            }
            array[i] = value;
        }
    }
    return array;
}

/* This function Gets C double array, build and return python type list (float). */
PyObject *cArrToPythonList(double *array, int len) {
    Py_ssize_t i;
    PyObject *pyList, *pyValue;

    pyList = PyList_New(len);
    if(pyList != NULL) { /* Memory allocation fail */
        for (i = 0; i < len; ++i) {
            pyValue = PyFloat_FromDouble(array[i]);
            if (pyValue == NULL || PyList_SetItem(pyList, i, pyValue)) {
                Py_DecRef(pyList); /* Free the list */
                return NULL; /* Set error */
            }
        }
    }
    return pyList;
}

/* This function Gets python type list of lists (float) and convert into C double matrix. */
double **pyLOLToCMat(PyObject *pyListOfLists, int rows, int cols) {
    Py_ssize_t i, j;
    double **matrix, value;
    PyObject *pyList, *pyValue;

    if (!PyList_Check(pyListOfLists)) { /* Not a list */
        MyPy_TypeErr("list", pyListOfLists);
        return NULL;
    }
    /* Allocate memory for matrix */
    matrix = (double **) alloc2DArray(rows, cols, sizeof(double), sizeof(double *),
                                      freeUsedMem);
    if (matrix != NULL) { /* Memory allocation fail */
        for (i = 0; i < rows; ++i) {
            pyList = PyList_GetItem(pyListOfLists, i);
            if (PyErr_Occurred()) /* Check for an error */
                return NULL;
            if (!PyList_Check(pyList)) { /* Not a list */
                MyPy_TypeErr("list", pyList);
                return NULL;
            }
            for (j = 0; j < cols; ++j) {
                pyValue = PyList_GetItem(pyList, j);
                value = pyValue != NULL ? PyFloat_AsDouble(pyValue) : EOF;
                if (PyErr_Occurred()) /* Check for an error */
                    return NULL;
                matrix[i][j] = value;
            }
        }
    }
    return matrix;
}

/* This function Gets C double array, build and return python type list (float). */
PyObject *cMatToPyLOL(double **matrix, int rows, int cols) {
    Py_ssize_t i, j;
    PyObject *pyLOL, *pyList, *pyValue;

    pyLOL = PyList_New(rows);
    if(pyLOL != NULL) { /* Memory allocation fail */
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

/* This function pack kmeans results into python tuple. */
PyObject *kmeansResToPyObject(double **matrix, int rows, int cols, int numOfDatapoints) {
    PyObject *pyCentroidsMat, *pyVecLabeling;

    pyCentroidsMat = cMatToPyLOL(matrix, rows, cols);
    pyVecLabeling = cArrToPythonList(matrix[rows], numOfDatapoints);
    if (pyCentroidsMat == NULL || pyVecLabeling == NULL)
        return NULL; /* Error */

    /* Pack into tuple */
    return PyTuple_Pack(2, pyCentroidsMat, pyVecLabeling);
}

/* This function pack jacobi results into python tuple. */
PyObject *jacobiResToPyObject(double **eigenvectorsMat, double *eigenvalues, int n) {
    PyObject *pyEigenvectorsMat, *pyEigenvalues;

    pyEigenvectorsMat = cMatToPyLOL(eigenvectorsMat, n, n);
    pyEigenvalues = cArrToPythonList(eigenvalues, n);
    if (pyEigenvectorsMat == NULL || pyEigenvalues == NULL)
        return NULL; /* Error */

    /* Pack into tuple */
    return PyTuple_Pack(2, pyEigenvectorsMat, pyEigenvalues);
}