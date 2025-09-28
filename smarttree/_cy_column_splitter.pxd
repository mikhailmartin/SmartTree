cimport numpy as cnp

from ._criterion cimport ClassificationCriterion


cdef class CyBaseColumnSplitter:

    cdef ClassificationCriterion criterion
    cdef cnp.int64_t[:] y
    cdef Py_ssize_t n_classes
    cdef Py_ssize_t n_samples
