from typing import Self

import pandas as pd


class TreeNode:
    """Decision Tree Node."""
    def __init__(
        self,
        number: int,
        num_samples: int,
        depth: int,
        mask: pd.Series,
        hierarchy: dict[str: str | list[str]],
        available_feature_names: list[str],

        impurity: float,
        distribution: list[int],  # classification
        label: str,  # classification

        is_leaf: bool = True,
        childs: list[Self] | None = None,

        # set by NodeSplitter.find_best_split_for()
        information_gain: float | None = None,
        split_type: str | None = None,
        split_feature_name: str | None = None,
        feature_values: list | None = None,
        child_masks: list | None = None,

        # set by Builder.build()
        feature_value=None,  # TODO аннотация
    ) -> None:

        self.number = number
        self.num_samples = num_samples
        self.depth = depth
        self.mask = mask
        self.hierarchy = hierarchy
        self.available_feature_names = available_feature_names

        self.impurity = impurity
        self.distribution = distribution
        self.label = label

        self.is_leaf = is_leaf
        self.childs = [] if childs is None else childs

        self.information_gain = information_gain
        self.split_type = split_type
        self.split_feature_name = split_feature_name
        self.feature_values = feature_values
        self.child_masks = child_masks

        self.feature_value = feature_value

    def __repr__(self) -> str:
        representation = [
            f"node_number={self.number}",
            f"num_samples={self.num_samples}",
            f"distribution={self.distribution}",
            f"impurity={self.impurity}",
            f"label={self.label!r}",
        ]

        return f"{self.__class__.__name__}({', '.join(representation)})"
