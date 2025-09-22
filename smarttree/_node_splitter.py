from typing import NamedTuple

import pandas as pd

from ._column_splitter import CatColumnSplitter, NumColumnSplitter, RankColumnSplitter
from ._dataset import Dataset
from ._tree import TreeNode
from ._types import ClassificationCriterionType, NaModeType, SplitType


class NodeSplitResult(NamedTuple):

    information_gain: float
    split_type: str
    split_feature: str
    feature_values: list[list[str]]
    child_masks: list[pd.Series]
    child_na_index: int = -1

    @classmethod
    def no_split(cls):
        return cls(
            information_gain=float("-inf"),
            split_type="",
            split_feature="",
            feature_values=[],
            child_masks=[],
        )


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
        num_features: list[str],
        cat_features: list[str],
        rank_features: dict[str, list],
        feature_na_mode: dict[str, NaModeType],
    ) -> None:

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease

        self.feature_split_type: dict[str, SplitType] = dict()
        for num_feature in num_features:
            self.feature_split_type[num_feature] = "numerical"
        for cat_feature in cat_features:
            self.feature_split_type[cat_feature] = "categorical"
        for rank_feature in rank_features:
            self.feature_split_type[rank_feature] = "rank"

        dataset = Dataset(X, y)
        self.num_col_splitter = NumColumnSplitter(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            feature_na_mode=feature_na_mode,
        )
        self.cat_col_splitter = CatColumnSplitter(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            feature_na_mode=feature_na_mode,
            max_childs=max_childs,
        )
        self.rank_col_splitter = RankColumnSplitter(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            rank_features=rank_features,
            feature_na_mode=feature_na_mode,
        )

    def is_splittable(self, node: TreeNode, leaf_counter: int) -> bool:
        """
        Checks whether a tree node can be split.

        If all conditions are met, the split information is stored in the node.
        """
        if node.depth >= self.max_depth:
            return False

        if node.num_samples < self.min_samples_split:
            return False

        split_result = self.find_best_split_for(node, leaf_counter)
        if split_result.information_gain >= self.min_impurity_decrease:
            node.information_gain = split_result.information_gain
            node.split_type = split_result.split_type
            node.split_feature = split_result.split_feature
            node.feature_values = split_result.feature_values
            node.child_masks = split_result.child_masks
            return True
        else:
            return False

    def find_best_split_for(self, node: TreeNode, leaf_counter: int) -> NodeSplitResult:

        best_split_result = NodeSplitResult.no_split()
        for feature in node.available_features:
            split_type = self.feature_split_type[feature]
            match split_type:
                case "numerical":
                    split_result = self.num_col_splitter.split(node, feature)
                case "categorical":
                    split_result = self.cat_col_splitter.split(node, feature, leaf_counter)
                case "rank":
                    split_result = self.rank_col_splitter.split(node, feature)

            if best_split_result.information_gain < split_result.information_gain:
                best_split_result = NodeSplitResult(
                    split_result.information_gain,
                    split_type,
                    feature,
                    split_result.feature_values,
                    split_result.child_masks,
                    split_result.child_na_index,
                )

        return best_split_result
