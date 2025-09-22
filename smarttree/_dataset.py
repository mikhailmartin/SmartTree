from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Dataset:

    X: pd.DataFrame
    y: pd.Series

    def __post_init__(self) -> None:
        self.class_names: NDArray = np.sort(self.y.unique())
        self.mask_na = {column: self.X[column].isna() for column in self.X.columns}

    @property
    def size(self) -> int:
        return self.X.shape[0]
