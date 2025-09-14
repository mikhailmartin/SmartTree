from collections import defaultdict
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(slots=True)
class TreeNode:

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

    @classmethod
    def dummy(cls):
        return cls(
            number=-1,
            num_samples=-1,
            depth=-1,
            mask=pd.Series(),
            hierarchy={},
            available_features=[],
            distribution=np.array([]),
            impurity=0,
            label="",
        )


class Tree:
    def __init__(self) -> None:
        self.root: TreeNode = TreeNode.dummy()
        self.node_counter: int = 0
        self.leaf_counter: int = 0
        self.max_depth: int = 0
        self.feature_importances: defaultdict[str, float] = defaultdict(float)

    def create_node(
        self,
        mask: pd.Series,
        distribution: NDArray[np.integer],
        impurity: float,
        label: str,
        hierarchy: dict[str, str | list[str]],
        available_features: list[str],
        depth: int,
        is_root: bool = False,
    ) -> TreeNode:

        node = TreeNode(
            number=self.node_counter,
            num_samples=mask.sum(),
            distribution=distribution,
            impurity=impurity,
            label=label,
            depth=depth,
            mask=mask,
            hierarchy=hierarchy.copy(),
            available_features=available_features.copy(),
        )

        self.node_counter += 1
        self.leaf_counter += 1
        self.max_depth = max(self.max_depth, depth)

        if is_root:
            self.root = node

        return node

    def compute_feature_importances(self) -> dict[str, float]:

        amount = 0.0
        for importance in self.feature_importances.values():
            amount += importance

        normalized_feature_importances = dict()
        for feature, importance in self.feature_importances.items():
            normalized_feature_importances[feature] = importance / amount

        return normalized_feature_importances

