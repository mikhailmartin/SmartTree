cimport cython
from libc.math cimport log2
from libc.stdint cimport int8_t

import numpy as np


cdef class CyBaseColumnSplitter:

    def __init__(self, criterion):
        if criterion == "gini":
            self.impurity = self.gini_index
        elif criterion in ("entropy", "log_loss"):
            self.impurity = self.entropy

    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def gini_index(mask, y, class_names):

        cdef:
            int8_t[:] mask_arr = mask.values.astype(np.int8)
            object[:] y_arr = y.values

        cdef:
            int i
            Py_ssize_t n = len(mask)
            int8_t mask_value
            long N = 0
        for i in range(n):
            mask_value = mask_arr[i]
            if mask_value:
                N += 1

        cdef:
            int j
            long N_i = 0
            object class_name, label
            double p_i = 0.0
            gini_index = 1.0
        for j in range(len(class_names)):
            N_i = 0
            class_name = class_names[j]

            for i in range(n):
                mask_value = mask_arr[i]
                if mask_value:
                    label = y_arr[i]
                    if label == class_name:
                        N_i += 1

            p_i = <double>N_i / <double>N
            gini_index -= p_i * p_i

        return gini_index

    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def entropy(mask, y, class_names):

        cdef:
            int8_t[:] mask_arr = mask.values.astype(np.int8)
            object[:] y_arr = y.values

        cdef:
            int i
            Py_ssize_t n = len(mask)
            int8_t mask_value
            long N = 0
        for i in range(n):
            mask_value = mask_arr[i]
            if mask_value:
                N += 1

        cdef:
            int j
            long N_i = 0
            object class_name, label
            double p_i = 0.0
            entropy = 0.0
        for j in range(len(class_names)):
            N_i = 0
            class_name = class_names[j]

            for i in range(n):
                mask_value = mask_arr[i]
                if mask_value:
                    label = y_arr[i]
                    if label == class_name:
                        N_i += 1

            if N_i != 0:
                p_i = <double>N_i / <double>N
                entropy -= p_i * log2(p_i)

        return entropy
