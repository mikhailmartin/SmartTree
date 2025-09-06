from typing import Literal, NamedTuple

import pandas as pd

from ._column_splitter import (
    CategoricalColumnSplitter,
    NumericalColumnSplitter,
    RankColumnSplitter,
)
from ._constants import (
    CategoricalNanModeOption,
    ClassificationCriterionOption,
    NumericalNanModeOption,
)
from ._dataset import Dataset
from ._tree_node import TreeNode


SplitTypeOption = Literal["numerical", "categorical", "rank"]


class NodeSplitResult(NamedTuple):
    information_gain: float
    split_type: str
    split_feature_name: str
    feature_values: list[list[str]]
    child_masks: list[pd.Series]


class NodeSplitter:

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        criterion: ClassificationCriterionOption,
        max_depth: int | float,
        min_samples_split: int,
        min_samples_leaf: int,
        max_leaf_nodes: int | float,
        min_impurity_decrease: float,
        max_childs: int | float,
        numerical_feature_names: list[str],
        categorical_feature_names: list[str],
        rank_feature_names: dict[str, list],
        numerical_nan_mode: NumericalNanModeOption,
        categorical_nan_mode: CategoricalNanModeOption,
    ) -> None:

        self.dataset = Dataset(X, y)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.max_childs = max_childs
        self.numerical_feature_names = numerical_feature_names
        self.categorical_feature_names = categorical_feature_names
        self.rank_feature_names = rank_feature_names
        self.numerical_nan_mode = numerical_nan_mode
        self.categorical_nan_mode = categorical_nan_mode

        self.leaf_counter: int = 0

        self.feature_split_type: dict[str, SplitTypeOption] = dict()
        for feature in self.numerical_feature_names:
            self.feature_split_type[feature] = "numerical"
        for feature in self.categorical_feature_names:
            self.feature_split_type[feature] = "categorical"
        for feature in self.rank_feature_names:
            self.feature_split_type[feature] = "rank"

        self.num_col_splitter = NumericalColumnSplitter(
            dataset=self.dataset,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            numerical_nan_mode=self.numerical_nan_mode,
        )
        self.cat_col_splitter = CategoricalColumnSplitter(
            dataset=self.dataset,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            categorical_nan_mode=self.categorical_nan_mode,
            max_childs=self.max_childs,
        )
        self.rank_col_splitter = RankColumnSplitter(
            dataset=self.dataset,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            rank_feature_names=self.rank_feature_names,
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
            node.split_feature_name = split_result.split_feature_name
            node.feature_values = split_result.feature_values
            node.child_masks = split_result.child_masks
            return True
        else:
            return False

    def find_best_split_for(self, node: TreeNode) -> NodeSplitResult:
        """Finds the best tree node split, if it exists."""
        best_split_result = NodeSplitResult(float("-inf"), "", "", [], [])
        for feature_name in node.available_feature_names:
            split_type = self.feature_split_type[feature_name]
            match split_type:
                case "numerical":
                    split_result = self.num_col_splitter.split(node, feature_name)
                case "categorical":
                    split_result = self.cat_col_splitter.split(node, feature_name, self.leaf_counter)
                case "rank":
                    split_result = self.rank_col_splitter.split(node, feature_name)

            if best_split_result.information_gain < split_result.information_gain:
                best_split_result = NodeSplitResult(
                    split_result.information_gain,
                    split_type,
                    feature_name,
                    split_result.feature_values,
                    split_result.child_masks,
                )

        return best_split_result
