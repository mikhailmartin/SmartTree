import pandas as pd
from numpy.typing import NDArray

def cgini_index(mask: pd.Series, y: pd.Series, class_names: NDArray) -> float: ...
