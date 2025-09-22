import pandas as pd
from numpy.typing import NDArray

from ._types import ClassificationCriterionType

class CyBaseColumnSplitter:

    def __init__(self, criterion: ClassificationCriterionType) -> None:
        ...

    @staticmethod
    def gini_index(mask: pd.Series, y: pd.Series, class_names: NDArray) -> float:
        ...

    @staticmethod
    def entropy(mask: pd.Series, y: pd.Series, class_names: NDArray) -> float:
        ...
