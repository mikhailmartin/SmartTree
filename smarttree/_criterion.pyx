cimport cython
from libc.math cimport log2

import numpy as np
cimport numpy as cnp

from ._dataset import Dataset


cnp.import_array()


cdef class ClassificationCriterion:

    def __cinit__(self, dataset: Dataset) -> None:
        self.y = dataset.y
        self.n_classes = len(dataset.classes)
        self.n_samples = len(dataset.y)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef long[:] distribution(self, cnp.npy_bool[:] mask):

        cdef Py_ssize_t i
        cdef long[:] result

        result = np.zeros(self.n_classes, dtype=np.int32)
        for i in range(self.n_samples):
            if mask[i]:
                result[self.y[i]] += 1

        return result


cdef class Gini(ClassificationCriterion):

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef double impurity(self, cnp.npy_bool[:] mask):

        cdef Py_ssize_t i
        cdef long[:] distribution
        cdef long N
        cdef double p_i, gini

        distribution = self.distribution(mask)
        N = 0
        for i in range(self.n_classes):
            N += distribution[i]

        gini = 1.0
        for i in range(self.n_classes):
            if distribution[i] > 0:
                p_i = <double>distribution[i] / <double>N
                gini -= p_i * p_i

        return gini


cdef class Entropy(ClassificationCriterion):

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef double impurity(self, cnp.npy_bool[:] mask):

        cdef Py_ssize_t i
        cdef long[:] distribution
        cdef long N
        cdef double p_i, gini

        distribution = self.distribution(mask)
        N = 0
        for i in range(self.n_classes):
            N += distribution[i]

        entropy = 0.0
        for i in range(self.n_classes):
            if distribution[i] > 0:
                p_i = <double>distribution[i] / <double>N
                entropy -= p_i * log2(p_i)

        return entropy
