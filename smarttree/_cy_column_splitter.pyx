cimport cython
from libc.stdint cimport int8_t
import numpy as np


cdef class CyBaseColumnSplitter:

    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def gini_index(mask, y, class_names):

        cdef int8_t[:] mask_arr = mask.values.astype(np.int8)
        cdef object[:] y_arr = y.values
        cdef long N = 0
        cdef long N_i = 0
        cdef double p_i = 0.0
        cdef double gini_index = 1.0
        cdef int i
        cdef int j
        cdef int n = len(mask)
        cdef int n_classes = len(class_names)
        cdef object class_name
        cdef object label
        cdef int8_t mask_value

        for i in range(n):
            mask_value = mask_arr[i]
            if mask_value:
                N += 1

        if N == 0:
            return 0.0

        for j in range(n_classes):
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
