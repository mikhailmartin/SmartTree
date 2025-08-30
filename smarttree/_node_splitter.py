from collections import namedtuple

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
from ._tree_node import TreeNode

SplitResult = namedtuple(
    typename="SplitResult",
    field_names=[
        "inf_gain", "split_type", "split_feature_name", "feature_values", "child_masks"
    ],
)


class NodeSplitter:
    """TODO."""

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
        self.X = X
        self.y = y
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

        self.feature_split_type: dict = dict()
        for feature in self.numerical_feature_names:
            self.feature_split_type[feature] = "numerical"
        for feature in self.categorical_feature_names:
            self.feature_split_type[feature] = "categorical"
        for feature in self.rank_feature_names:
            self.feature_split_type[feature] = "rank"

        self.class_names: list[str] = sorted(y.unique())

        # column splitters
        self.numerical_column_splitter = NumericalColumnSplitter(
            X=self.X,
            y=self.y,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            numerical_nan_mode=self.numerical_nan_mode,
        )
        self.categorical_column_splitter = CategoricalColumnSplitter(
            X=self.X,
            y=self.y,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            categorical_nan_mode=self.categorical_nan_mode,
            max_childs=self.max_childs,
        )
        self.rank_column_splitter = RankColumnSplitter(
            X=self.X,
            y=self.y,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            rank_feature_names=self.rank_feature_names,
        )

    # TODO: не только проверяет, но и мутирует узел
    def is_splittable(self, node: TreeNode) -> bool:
        """Checks whether a tree node can be split."""
        if node.depth >= self.max_depth:
            return False

        if node.samples < self.min_samples_split:
            return False

        split_result = self.find_best_split(node.mask, node.available_feature_names)
        if split_result.inf_gain < self.min_impurity_decrease:
            return False
        else:
            node.split_result = split_result
            return True

    def find_best_split(
        self,
        parent_mask: pd.Series,
        available_feature_names: list[str],
    ) -> SplitResult:
        """
        Finds the best tree node split, if it exists.

        Parameters:
            parent_mask: boolean mask of the tree node.
            available_feature_names: the list of features available for splitting.

        Returns:
            Tuple `(inf_gain, split_type, split_feature_name, feature_values,
            child_masks)`.
              inf_gain: information gain of the split.
              split_type: split type.
              split_feature_name: the feature by which it is best to split the input set.
              feature_values: feature values corresponding to child nodes.
              child_masks: list of child masks.
        """
        best_inf_gain = float("-inf")
        best_split_type = None
        best_split_feature_name = None
        best_feature_values = None
        best_child_masks = None
        for split_feature_name in available_feature_names:
            split_type = self.feature_split_type[split_feature_name]
            match split_type:
                case "numerical":
                    inf_gain, feature_values, child_masks = \
                        self.numerical_column_splitter.split(parent_mask, split_feature_name)
                case "categorical":
                    inf_gain, feature_values, child_masks = \
                        self.categorical_column_splitter.split(parent_mask, split_feature_name, self.leaf_counter)
                case "rank":
                    inf_gain, feature_values, child_masks = \
                        self.rank_column_splitter.split(parent_mask, split_feature_name)

            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_split_type = split_type
                best_split_feature_name = split_feature_name
                best_feature_values = feature_values
                best_child_masks = child_masks

        return SplitResult(
            inf_gain=best_inf_gain,
            split_type=best_split_type,
            split_feature_name=best_split_feature_name,
            feature_values=best_feature_values,
            child_masks=best_child_masks,
        )
