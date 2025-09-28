import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X
        self.classes = np.sort(y.unique())
        self.y = np.searchsorted(self.classes, y.to_numpy()).astype(np.int64)
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
