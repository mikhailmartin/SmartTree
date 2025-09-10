from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(slots=True)
class TreeNode:
    """Decision Tree Node."""

    number: int
    num_samples: int
    depth: int = field(repr=False)
    mask: pd.Series = field(repr=False)
    hierarchy: dict[str, str | list[str]] = field(repr=False)
    available_features: list[str] = field(repr=False)

    distribution: NDArray[np.integer]  # classification
    impurity: float
    label: str  # classification

    is_leaf: bool = field(init=False, repr=False)
    childs: list[Self] = field(init=False, repr=False)

    # set by NodeSplitter.find_best_split_for()
    information_gain: float = field(init=False, repr=False)
    split_type: str = field(init=False, repr=False)
    split_feature: str = field(init=False, repr=False)
    feature_values: list = field(init=False, repr=False)
    child_masks: list = field(init=False, repr=False)
    child_na_index: int = field(init=False, repr=False)

    # set by Builder.build()
    feature_value: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.is_leaf = True
        self.childs = []

        self.information_gain = float("-inf")
        self.split_type = ""
        self.split_feature = ""
        self.feature_values = []
        self.child_masks = []
        self.child_na_index = -1

        self.feature_value = []
