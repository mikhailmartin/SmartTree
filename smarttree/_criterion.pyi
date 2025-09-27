from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ._dataset import Dataset

class ClassificationCriterion(ABC):

    def __init__(self, dataset: Dataset) -> None:
        ...

    @abstractmethod
    def impurity(self, mask: NDArray[np.int8]) -> float:
        raise NotImplementedError

    def distribution(self, mask: NDArray[np.int8]) -> NDArray[np.int32]:
        ...


class Gini(ClassificationCriterion):

    def impurity(self, mask: NDArray[np.int8]) -> float:
        r"""
        Calculates Gini index in a tree node.

        Gini index formula in LaTeX:
            \text{Gini Index} = 1 - \sum^C_{i=1} p_i^2
            where
            C - total number of classes;
            p_i - the probability of choosing a sample with class i.
        """
        ...


class Entropy(ClassificationCriterion):

    def impurity(self, mask: NDArray[np.int8]) -> float:
        r"""
        Calculates entropy in a tree node.

        Entropy formula in LaTeX:
        H = \log{\overline{N}} = \sum^N_{i=1} p_i \log{(1/p_i)} = -\sum^N_{i=1} p_i \log{p_i}
        where
        H - entropy;
        \overline{N} - effective number of states;
        p_i - probability of the i-th system state.
        """
        ...
