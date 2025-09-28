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
    cpdef cnp.int64_t[:] distribution(self, cnp.npy_bool[:] mask):

        cdef Py_ssize_t i
        cdef cnp.int64_t[:] result
        cdef cnp.npy_bool mask_value
        cdef cnp.int64_t class_index

        result = np.zeros(self.n_classes, dtype=np.int64)
        for i in range(self.n_samples):
            mask_value = mask[i]
            if mask_value:
                class_index = self.y[i]
                result[class_index] += 1

        return result


cdef class Gini(ClassificationCriterion):

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef double impurity(self, cnp.npy_bool[:] mask):

        cdef Py_ssize_t i
        cdef cnp.int64_t[:] distribution
        cdef cnp.int64_t N, count
        cdef double p_i, gini

        distribution = self.distribution(mask)
        N = 0
        for i in range(self.n_classes):
            count = distribution[i]
            N += count

        gini = 1.0
        for i in range(self.n_classes):
            count = distribution[i]
            if count > 0:
                p_i = <double>count / <double>N
                gini -= p_i * p_i

        return gini


cdef class Entropy(ClassificationCriterion):

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef double impurity(self, cnp.npy_bool[:] mask):

        cdef Py_ssize_t i
        cdef cnp.int64_t[:] distribution
        cdef cnp.int64_t N, count
        cdef double p_i, gini

        distribution = self.distribution(mask)
        N = 0
        for i in range(self.n_classes):
            count = distribution[i]
            N += count

        entropy = 0.0
        for i in range(self.n_classes):
            count = distribution[i]
            if count > 0:
                p_i = <double>count / <double>N
                entropy -= p_i * log2(p_i)

        return entropy
