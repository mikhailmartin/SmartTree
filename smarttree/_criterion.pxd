cimport numpy as cnp


cdef class ClassificationCriterion:

    cdef int[:] y
    cdef Py_ssize_t n_classes
    cdef Py_ssize_t n_samples

    cpdef long[:] distribution(self, cnp.npy_bool[:] mask)


cdef class Gini(ClassificationCriterion):
    cpdef double impurity(self, cnp.npy_bool[:] mask)

cdef class Entropy(ClassificationCriterion):
    cpdef double impurity(self, cnp.npy_bool[:] mask)
