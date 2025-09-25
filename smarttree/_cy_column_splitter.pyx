cimport cython
from libc.math cimport log2
from libc.stdint cimport int8_t

import numpy as np
import pandas as pd

from ._dataset import Dataset
from ._types import Criterion


cdef int CRITERION_GINI = 1


cdef class CyBaseColumnSplitter:

    def __cinit__(self, dataset: Dataset, criterion: Criterion) -> None:
        self.criterion = criterion.value
        self.y = dataset.y
        self.n_classes = len(dataset.classes)
        self.n_samples = len(dataset.y)

    def information_gain(
        self,
        parent_mask: pd.Series,
        child_masks: list[pd.Series],
        normalize: bool = False,
    ) -> float:

        cdef int8_t[:] parent_mask_arr, child_mask_arr
        cdef Py_ssize_t i, j, n_childs
        cdef long N, N_parent, N_childs, N_child_j
        cdef double impurity_parent, weighted_impurity_childs, impurity_child_i

        parent_mask_arr = parent_mask.values.astype(np.int8)
        child_mask_arrs = [
            child_mask.values.astype(np.int8) for child_mask in child_masks
        ]

        N = 0
        N_parent = 0
        for i in range(self.n_samples):
            N += 1
            if parent_mask_arr[i]:
                N_parent += 1

        impurity_parent = self.impurity(parent_mask_arr)

        N_childs = 0
        n_childs = len(child_mask_arrs)
        weighted_impurity_childs = 0.0
        for j in range(n_childs):
            N_child_j = 0
            child_mask_arr = child_mask_arrs[j]
            for i in range(self.n_samples):
                if child_mask_arr[i]:
                    N_child_j += 1
            N_childs += N_child_j
            impurity_child_i = self.impurity(child_mask_arr)
            weighted_impurity_childs += (<double>N_child_j / <double>N_parent) * impurity_child_i

        cdef double norm_coef
        if normalize:
            norm_coef = <double>N_parent / <double>N_childs
            weighted_impurity_childs *= norm_coef

        cdef double local_information_gain = impurity_parent - weighted_impurity_childs

        cdef double information_gain = (<double>N_parent / <double>N) * local_information_gain

        return information_gain

    cdef double impurity(self, int8_t[:] mask):
        if self.criterion == CRITERION_GINI:
            return self.gini_index(mask)
        else:
            return self.entropy(mask)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef double gini_index(self, int8_t[:] mask):

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

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cpdef double entropy(self, int8_t[:] mask):

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
