cimport numpy as cnp


cdef class Criterion:

    cdef cnp.int64_t[:] y
    cdef Py_ssize_t n_samples

    cpdef double impurity_decrease(
        self,
        cnp.npy_bool[:] parent_mask,
        list[cnp.npy_bool[:]] child_masks,
        bint normalize,
    )


cdef class ClassificationCriterion(Criterion):

    cdef Py_ssize_t n_classes

    cpdef cnp.int64_t[:] value(self, cnp.npy_bool[:] mask)


cdef class Gini(ClassificationCriterion):
    cpdef double impurity(self, cnp.npy_bool[:] mask)


cdef class Entropy(ClassificationCriterion):
    cpdef double impurity(self, cnp.npy_bool[:] mask)


class RegressionCriterion(Criterion):
    def value(self, mask) -> float: ...


class MSE(RegressionCriterion):
    def impurity(self, mask)  -> float: ...
