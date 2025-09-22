from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Dataset:

    X: pd.DataFrame
    y: pd.Series
    class_names: NDArray = field(init=False)
    has_na: dict[str, bool] = field(init=False)
    mask_na: dict[str, pd.Series] = field(init=False)

    def __post_init__(self) -> None:
        self.class_names = np.sort(self.y.unique())
        self.has_na = dict()
        self.mask_na = dict()
        for column in self.X.columns:
            mask_na = self.X[column].isna()
            has_na = mask_na.any()
            self.has_na[column] = has_na
            if has_na:
                self.mask_na[column] = mask_na

    @property
    def size(self) -> int:
        return self.X.shape[0]
