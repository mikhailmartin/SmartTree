cimport cython
from libc.math cimport log2
from libc.stdint cimport int8_t

import numpy as np
import pandas as pd

from ._dataset import Dataset


cdef class CyBaseColumnSplitter:

    cdef public str criterion
    cdef public object[:] y
    cdef public object[:] class_names

    def __cinit__(self, dataset: Dataset, criterion: str) -> None:
        self.criterion = criterion
        self.y = dataset.y.values
        self.class_names = dataset.class_names

    cdef double impurity(self, int8_t[:] mask):
        if self.criterion == "gini":
            return self.gini_index(mask)
        elif self.criterion in ("entropy", "log_loss"):
            return self.entropy(mask)
        else:
            assert False

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
            int i
            Py_ssize_t n = len(parent_mask_arr)
            long N = 0
            long N_parent = 0
            int8_t parent_mask_value
        for i in range(n):
            N += 1
            parent_mask_value = parent_mask_arr[i]
            if parent_mask_value:
                N_parent += 1

        cdef double impurity_parent = self.impurity(parent_mask_arr)

        cdef:
            int j
            double weighted_impurity_childs = 0.0
            long N_childs = 0
            long N_child_j
            int8_t child_mask_value
            double impurity_child_i
        for j in range(len(child_mask_arrs)):
            N_child_j = 0
            child_mask_arr = child_mask_arrs[j]
            for i in range(n):
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
            int i
            Py_ssize_t n = len(mask)
            int8_t mask_value
            long N = 0
        for i in range(n):
            mask_value = mask[i]
            if mask_value:
                N += 1

        cdef:
            int j
            cdef long N_i
            cdef object class_name, label
            double p_i = 0.0
            gini_index = 1.0
        for j in range(len(self.class_names)):
            N_i = 0
            class_name = self.class_names[j]

            for i in range(n):
                mask_value = mask[i]
                if mask_value:
                    label = self.y[i]
                    if label == class_name:
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
            int i
            Py_ssize_t n = len(mask)
            int8_t mask_value
            long N = 0
        for i in range(n):
            mask_value = mask[i]
            if mask_value:
                N += 1

        cdef:
            int j
            long N_i = 0
            object class_name, label
            double p_i = 0.0
            entropy = 0.0
        for j in range(len(self.class_names)):
            N_i = 0
            class_name = self.class_names[j]

            for i in range(n):
                mask_value = mask[i]
                if mask_value:
                    label = self.y[i]
                    if label == class_name:
                        N_i += 1

            if N_i != 0:
                p_i = <double>N_i / <double>N
                entropy -= p_i * log2(p_i)

        return entropy
