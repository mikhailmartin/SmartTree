import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Dataset:

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X
        self.y = y

        self.class_names: NDArray = np.sort(self.y.unique())
        self.has_na: dict[str, bool] = dict()
        self.mask_na: dict[str, pd.Series] = dict()
        for column in self.X.columns:
            mask_na = self.X[column].isna()
            has_na = mask_na.any()
            self.has_na[column] = has_na
            if has_na:
                self.mask_na[column] = mask_na

    @property
    def size(self) -> int:
        return self.X.shape[0]
