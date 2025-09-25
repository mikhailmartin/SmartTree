from libc.stdint cimport int8_t


cdef class CyBaseColumnSplitter:

    cdef int criterion
    cdef int[:] y
    cdef Py_ssize_t n_classes
    cdef Py_ssize_t n_samples

    cdef double impurity(self, int8_t[:] mask)
    cpdef double gini_index(self, int8_t[:] mask)
    cpdef double entropy(self, int8_t[:] mask)
