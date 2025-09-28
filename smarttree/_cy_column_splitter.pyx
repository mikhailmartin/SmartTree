cimport cython

import numpy as np
cimport numpy as cnp
import pandas as pd

from ._dataset import Dataset
from ._types import Criterion
from ._criterion cimport Entropy, Gini


cnp.import_array()

cdef int CRITERION_GINI = 1


cdef class CyBaseColumnSplitter:

    def __cinit__(self, dataset: Dataset, criterion: Criterion) -> None:
        self.y = dataset.y
        self.n_classes = len(dataset.classes)
        self.n_samples = len(dataset.y)
        if criterion.value == CRITERION_GINI:
            self.criterion = Gini(dataset)
        else:
            self.criterion = Entropy(dataset)

    def information_gain(
        self,
        parent_mask: pd.Series,
        child_masks: list[pd.Series],
        normalize: bool = False,
    ) -> float:

        cdef cnp.npy_bool[:] parent_mask_arr, child_mask_arr
        cdef Py_ssize_t i, j, n_childs
        cdef long N, N_parent, N_childs, N_child_j
        cdef double impurity_parent, weighted_impurity_childs, impurity_child_i

        parent_mask_arr = parent_mask.to_numpy()
        child_mask_arrs = [child_mask.to_numpy() for child_mask in child_masks]

        N = 0
        N_parent = 0
        for i in range(self.n_samples):
            N += 1
            if parent_mask_arr[i]:
                N_parent += 1

        impurity_parent = self.criterion.impurity(parent_mask_arr)

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
            impurity_child_i = self.criterion.impurity(child_mask_arr)
            weighted_impurity_childs += (<double>N_child_j / <double>N_parent) * impurity_child_i

        cdef double norm_coef
        if normalize:
            norm_coef = <double>N_parent / <double>N_childs
            weighted_impurity_childs *= norm_coef

        cdef double local_information_gain = impurity_parent - weighted_impurity_childs

        cdef double information_gain = (<double>N_parent / <double>N) * local_information_gain

        return information_gain
