#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Custom made python type error */
#define MyPy_TypeErr(x, y) \
PyErr_Format(PyExc_TypeError, "%s type is required (got type %s)", x ,Py_TYPE(y)->tp_name) \

/* Print clusters' final centroids */
PyObject *buildPyListCentroids(Cluster *clustersArray, const int *k, const int *dimension);

/*
* This actually defines the kMeans function using a wrapper C API function
* The wrapping function needs a PyObject* self argument.
* This is a requirement for all functions and methods in the C API.
* It has input PyObject *args from Python.
*/
static PyObject *fit_connect(PyObject *self, PyObject *args) {
    Py_ssize_t i;
    PyObject * pyCentralsList, *pyVectorsList, *resault = NULL;
    int k, maxIter, dimension, numOfVectors, *firstCentralIndexes;
    double **vectorsArray = NULL; /* Default value - not allocated */
    Cluster *clustersArray = NULL; /* Default value - not allocated */

    if (!PyArg_ParseTuple(args, "iiiiOO", &k, &maxIter, &dimension, &numOfVectors, &pyCentralsList, &pyVectorsList))
        return NULL; /* Type error - not in correct format */
        if (!PyList_Check(pyCentralsList)) {
            MyPy_TypeErr("List", pyCentralsList);
            return NULL; /* Type error - not a python List */
        }
        if (!PyList_Check(pyVectorsList)) {
            MyPy_TypeErr("List", pyVectorsList);
            return NULL; /* Type error - not a python List */
        }

        firstCentralIndexes = (int *) malloc(k * sizeof(int));
        if (firstCentralIndexes == NULL) {
            resault = PyErr_NoMemory(); /* Memory allocation error */
            goto end;
        }
        for (i = 0; i < k; i++) {
            firstCentralIndexes[i] = (int) PyLong_AsLong(PyList_GetItem(pyCentralsList, i));
            if (PyErr_Occurred()) {
                goto end; /* Casting error to int */
            }
        }
        if (initVectorsArray(&vectorsArray, &numOfVectors, &dimension,
                             pyVectorsList)) /* return 0 (false) on success, true on error */
            goto end; /* On any error from initVectorsArray() */
            resault = kMeans(k, maxIter, dimension, numOfVectors, vectorsArray, firstCentralIndexes, &clustersArray);

            end:
    freeMemoryVectorsClusters(vectorsArray, clustersArray, &k, firstCentralIndexes); /* Free memory */
    return resault;
}

/*
 * This array tells Python what methods this module has.
 * We will use it in the next structure
 */
static PyMethodDef _method[] = {
        {"kMeans",                      /* the Python method name that will be used */
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
PyInit_mykmeanssp(void) {
    return PyModule_Create(&moduledef);
}

PyObject *buildPyListCentroids(Cluster *clustersArray, const int *k, const int *dimension) {
    Py_ssize_t i, j;
    PyObject * listOfCentrals, *central, *comp;
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