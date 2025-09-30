cimport numpy as cnp


cdef class ClassificationCriterion:

    cdef cnp.int64_t[:] y
    cdef Py_ssize_t n_classes
    cdef Py_ssize_t n_samples

    cpdef double impurity_decrease(
        self,
        cnp.npy_bool[:] parent_mask,
        list[cnp.npy_bool[:]] child_masks,
        bint normalize,
    )
    cpdef cnp.int64_t[:] distribution(self, cnp.npy_bool[:] mask)


cdef class Gini(ClassificationCriterion):
    cpdef double impurity(self, cnp.npy_bool[:] mask)

cdef class Entropy(ClassificationCriterion):
    cpdef double impurity(self, cnp.npy_bool[:] mask)
