from libc.stdint cimport int8_t

from ._criterion cimport ClassificationCriterion


cdef class CyBaseColumnSplitter:

    cdef ClassificationCriterion criterion
    cdef int[:] y
    cdef Py_ssize_t n_classes
    cdef Py_ssize_t n_samples
