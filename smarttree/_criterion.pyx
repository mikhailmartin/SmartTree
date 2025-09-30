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

    cpdef double impurity_decrease(
        self,
        cnp.npy_bool[:] parent_mask,
        list[cnp.npy_bool[:]] child_masks,
        bint normalize,
    ):

        cdef Py_ssize_t i, j, n_childs
        cdef long N, N_parent, N_childs, N_child_j
        cdef double impurity_parent, weighted_impurity_childs, impurity_child_i, norm_coef, local_information_gain, information_gain

        N = 0
        N_parent = 0
        for i in range(self.n_samples):
            N += 1
            if parent_mask[i]:
                N_parent += 1

        impurity_parent = self.impurity(parent_mask)

        N_childs = 0
        n_childs = len(child_masks)
        weighted_impurity_childs = 0.0
        for j in range(n_childs):
            N_child_j = 0
            child_mask = child_masks[j]
            for i in range(self.n_samples):
                if child_mask[i]:
                    N_child_j += 1
            N_childs += N_child_j
            impurity_child_i = self.impurity(child_mask)
            weighted_impurity_childs += (<double>N_child_j / <double>N_parent) * impurity_child_i

        if normalize:
            norm_coef = <double>N_parent / <double>N_childs
            weighted_impurity_childs *= norm_coef

        local_information_gain = impurity_parent - weighted_impurity_childs

        information_gain = (<double>N_parent / <double>N) * local_information_gain

        return information_gain

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
