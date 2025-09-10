from typing import NamedTuple

import pandas as pd

from ._column_splitter import (
    CategoricalColumnSplitter,
    NumericalColumnSplitter,
    RankColumnSplitter,
)
from ._dataset import Dataset
from ._tree_node import TreeNode
from ._types import (
    CategoricalNaModeType,
    ClassificationCriterionType,
    NumericalNaModeType,
    SplitType,
)


NO_INFORMATION_GAIN = float("-inf")


class NodeSplitResult(NamedTuple):
    information_gain: float
    split_type: str
    split_feature: str
    feature_values: list[list[str]]
    child_masks: list[pd.Series]


class NodeSplitter:

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        criterion: ClassificationCriterionType,
        max_depth: int | float,
        min_samples_split: int,
        min_samples_leaf: int,
        max_leaf_nodes: int | float,
        min_impurity_decrease: float,
        max_childs: int | float,
        numerical_features: list[str],
        categorical_features: list[str],
        rank_features: dict[str, list],
        numerical_na_mode: NumericalNaModeType,
        categorical_na_mode: CategoricalNaModeType,
    ) -> None:

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.leaf_counter: int = 0

        self.feature_split_type: dict[str, SplitType] = dict()
        for feature in numerical_features:
            self.feature_split_type[feature] = "numerical"
        for feature in categorical_features:
            self.feature_split_type[feature] = "categorical"
        for feature in rank_features:
            self.feature_split_type[feature] = "rank"

        dataset = Dataset(X, y)
        self.num_col_splitter = NumericalColumnSplitter(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            na_mode=numerical_na_mode,
        )
        self.cat_col_splitter = CategoricalColumnSplitter(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            na_mode=categorical_na_mode,
            max_childs=max_childs,
        )
        self.rank_col_splitter = RankColumnSplitter(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            rank_features=rank_features,
        )

    def is_splittable(self, node: TreeNode) -> bool:
        """
        Checks whether a tree node can be split.

        If all conditions are met, the split information is stored in the node.
        """
        if node.depth >= self.max_depth:
            return False

        if node.num_samples < self.min_samples_split:
            return False

        split_result = self.find_best_split_for(node)
        if split_result.information_gain >= self.min_impurity_decrease:
            node.information_gain = split_result.information_gain
            node.split_type = split_result.split_type
            node.split_feature = split_result.split_feature
            node.feature_values = split_result.feature_values
            node.child_masks = split_result.child_masks
            return True
        else:
            return False

    def find_best_split_for(self, node: TreeNode) -> NodeSplitResult:

        best_split_result = NodeSplitResult(NO_INFORMATION_GAIN, "", "", [], [])
        for feature in node.available_features:
            split_type = self.feature_split_type[feature]
            match split_type:
                case "numerical":
                    split_result = self.num_col_splitter.split(node, feature)
                case "categorical":
                    split_result = self.cat_col_splitter.split(node, feature, self.leaf_counter)
                case "rank":
                    split_result = self.rank_col_splitter.split(node, feature)

            if best_split_result.information_gain < split_result.information_gain:
                best_split_result = NodeSplitResult(
                    split_result.information_gain,
                    split_type,
                    feature,
                    split_result.feature_values,
                    split_result.child_masks,
                )

        return best_split_result
