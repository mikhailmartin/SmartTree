cimport cython
from libc.math cimport log2
from libc.stdint cimport int8_t

import numpy as np

from ._dataset import Dataset


cdef class ClassificationCriterion:

    def __cinit__(self, dataset: Dataset) -> None:
        self.y = dataset.y
        self.n_classes = len(dataset.classes)
        self.n_samples = len(dataset.y)


cdef class Gini(ClassificationCriterion):

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef double impurity(self, int8_t[:] mask):

        cdef Py_ssize_t i
        cdef long[:] counts
        cdef long N
        cdef double p_i, gini

        counts = np.zeros(self.n_classes, dtype=np.int32)
        N = 0
        for i in range(self.n_samples):
            if mask[i]:
                N += 1
                counts[self.y[i]] += 1

        gini = 1.0
        for i in range(self.n_classes):
            if counts[i] > 0:
                p_i = <double>counts[i] / <double>N
                gini -= p_i * p_i

        return gini


cdef class Entropy(ClassificationCriterion):

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef double impurity(self, int8_t[:] mask):

        cdef Py_ssize_t i
        cdef long[:] counts
        cdef long N
        cdef double p_i, entropy

        counts = np.zeros(self.n_classes, dtype=np.int32)
        N = 0
        for i in range(self.n_samples):
            if mask[i]:
                N += 1
                counts[self.y[i]] += 1

        entropy = 0.0
        for i in range(self.n_classes):
            if counts[i] > 0:
                p_i = <double>counts[i] / <double>N
                entropy -= p_i * log2(p_i)

        return entropy
