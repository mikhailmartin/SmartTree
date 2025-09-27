from libc.stdint cimport int8_t


cdef class ClassificationCriterion:

    cdef int[:] y
    cdef Py_ssize_t n_classes
    cdef Py_ssize_t n_samples


cdef class Gini(ClassificationCriterion):
    cpdef double impurity(self, int8_t[:] mask)

cdef class Entropy(ClassificationCriterion):
    cpdef double impurity(self, int8_t[:] mask)
