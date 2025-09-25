cimport cython
from libc.math cimport log2
from libc.stdint cimport int8_t

import numpy as np
import pandas as pd

from ._dataset import Dataset
from ._types import Criterion


cdef int CRITERION_GINI = 1


cdef class CyBaseColumnSplitter:

    cdef int criterion
    cdef int[:] y
    cdef Py_ssize_t n_classes
    cdef Py_ssize_t n_samples

    def __cinit__(self, dataset: Dataset, criterion: Criterion) -> None:
        self.criterion = criterion.value
        self.y = dataset.y
        self.n_classes = len(dataset.classes)
        self.n_samples = len(dataset.y)

    cdef double impurity(self, int8_t[:] mask):
        if self.criterion == CRITERION_GINI:
            return self.gini_index(mask)
        else:
            return self.entropy(mask)

    def information_gain(
        self,
        parent_mask: pd.Series,
        child_masks: list[pd.Series],
        normalize: bool = False,
    ) -> float:
        r"""
        Calculates information gain of the split.

        Parameters:
            parent_mask: pd.Series
              boolean mask of parent node.
            child_masks: pd.Series
              list of boolean masks of child nodes.
            normalize: bool, default=False
              if True, normalizes information gain by split factor to handle
              unbalanced splits. Uses child node counts for normalization.

        Returns:
            float: information gain.

        Formula in LaTeX:
            \begin{align*}
            \text{Information Gain} =
            \frac{N_{\text{parent}}}{N} \cdot
            \Biggl( & \text{impurity}_{\text{parent}} - \\
            & \sum^C_{i=1} \frac{N_{\text{child}_i}}{N_{\text{parent}}}
            \cdot \text{impurity}_{\text{child}_i} \Biggr)
            \end{align*}
            where:
            \begin{itemize}
                \item $\text{Information Gain}$ — information gain;
                \item $N$ — number of samples in entire training set;
                \item $N_{\text{parent}}$ — number of samples in parent node;
                \item $\text{impurity}_{\text{parent}}$ — parent node impurity;
                \item $C$ — number of child nodes;
                \item $N_{\text{child}_i}$ — number of samples in child node;
                \item $\text{impurity}_{\text{child}_i}$ — child node impurity.
            \end{itemize}
        """
        cdef int8_t[:] parent_mask_arr, child_mask_arr
        parent_mask_arr = parent_mask.values.astype(np.int8)
        child_mask_arrs = [
            child_mask.values.astype(np.int8) for child_mask in child_masks
        ]

        cdef:
            Py_ssize_t i
            long N = 0
            long N_parent = 0
            int8_t parent_mask_value
        for i in range(self.n_samples):
            N += 1
            parent_mask_value = parent_mask_arr[i]
            if parent_mask_value:
                N_parent += 1

        cdef double impurity_parent = self.impurity(parent_mask_arr)

        cdef:
            Py_ssize_t j
            Py_ssize_t n_childs = len(child_mask_arrs)
            double weighted_impurity_childs = 0.0
            long N_childs = 0
            long N_child_j
            int8_t child_mask_value
            double impurity_child_i
        for j in range(n_childs):
            N_child_j = 0
            child_mask_arr = child_mask_arrs[j]
            for i in range(self.n_samples):
                child_mask_value = child_mask_arr[i]
                if child_mask_value:
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

    cpdef double gini_index(self, int8_t[:] mask):
        r"""
        Calculates Gini index in a tree node.

        Gini index formula in LaTeX:
            \text{Gini Index} = 1 - \sum^C_{i=1} p_i^2
            where
            C - total number of classes;
            p_i - the probability of choosing a sample with class i.
        """
        cdef:
            Py_ssize_t i
            int8_t mask_value
            long N = 0
        for i in range(self.n_samples):
            mask_value = mask[i]
            if mask_value:
                N += 1

        cdef:
            Py_ssize_t j
            long N_i
            int encoded_label
            double p_i = 0.0
            double gini_index = 1.0
        for j in range(self.n_classes):
            N_i = 0

            for i in range(self.n_samples):
                mask_value = mask[i]
                if mask_value:
                    encoded_label = self.y[i]
                    if encoded_label == j:
                        N_i += 1

            p_i = <double>N_i / <double>N
            gini_index -= p_i * p_i

        return gini_index

    cpdef double entropy(self, int8_t[:] mask):
        r"""
        Calculates entropy in a tree node.

        Entropy formula in LaTeX:
        H = \log{\overline{N}} = \sum^N_{i=1} p_i \log{(1/p_i)} = -\sum^N_{i=1} p_i \log{p_i}
        where
        H - entropy;
        \overline{N} - effective number of states;
        p_i - probability of the i-th system state.
        """
        cdef:
            Py_ssize_t i
            int8_t mask_value
            long N = 0
        for i in range(self.n_samples):
            mask_value = mask[i]
            if mask_value:
                N += 1

        cdef:
            Py_ssize_t j
            long N_i = 0
            int encoded_label
            double p_i = 0.0
            double entropy = 0.0
        for j in range(self.n_classes):
            N_i = 0

            for i in range(self.n_samples):
                mask_value = mask[i]
                if mask_value:
                    encoded_label = self.y[i]
                    if encoded_label == j:
                        N_i += 1

            if N_i != 0:
                p_i = <double>N_i / <double>N
                entropy -= p_i * log2(p_i)

        return entropy
