import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Dataset:

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:

        self.classes = np.sort(y.unique())
        self.y = np.searchsorted(self.classes, y.to_numpy()).astype(np.int64)
        self.has_na: dict[str, bool] = dict()
        self.mask_na: dict[str, NDArray] = dict()
        self.columns: dict[str, NDArray] = dict()
        for column in X.columns:
            mask_na = X[column].isna()
            has_na = mask_na.any()
            self.has_na[column] = has_na
            self.mask_na[column] = mask_na.to_numpy()
            self.columns[column] = X[column].to_numpy()
        self.n_samples, self.n_columns = X.shape

    def __getitem__(self, item) -> NDArray:
        return self.columns[item]
