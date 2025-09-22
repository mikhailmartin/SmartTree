import pandas as pd
from numpy.typing import NDArray

class CyBaseColumnSplitter:
    @staticmethod
    def gini_index(mask: pd.Series, y: pd.Series, class_names: NDArray) -> float:
        ...
