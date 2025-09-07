import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Generator, NamedTuple

import numpy as np
import pandas as pd

from ._dataset import Dataset
from ._tree_node import TreeNode
from ._types import (
    CategoricalNanModeType,
    ClassificationCriterionType,
    NumericalNanModeType,
)


NO_INFORMATION_GAIN = float("-inf")


class ColumnSplitResult(NamedTuple):
    information_gain: float
    feature_values: list[list]
    child_masks: list[pd.Series]


class BaseColumnSplitter(ABC):

    def __init__(
        self,
        dataset: Dataset,
        criterion: ClassificationCriterionType,
        min_samples_split: int,
        min_samples_leaf: int,
    ) -> None:

        self.dataset = dataset
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        match self.criterion:
            case "gini":
                self.impurity = self.gini_index
            case "entropy" | "log_loss":
                self.impurity = self.entropy

    @abstractmethod
    def split(self, *args, **kwargs) -> ColumnSplitResult:
        raise NotImplementedError

    def information_gain(
        self,
        parent_mask: pd.Series,
        child_masks: list[pd.Series],
        nan_mode: str | None = None,  # TODO
    ) -> float:
        r"""
        Calculates information gain of the split.

        Parameters:
            parent_mask: pd.Series
              boolean mask of parent node.
            child_masks: pd.Series
              list of boolean masks of child nodes.
            nan_mode: str, default=None
              missing values handling node.
              - If 'include', then turn on normalization of child nodes impurity.

        Returns:
            float: information gain.

        References:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

        Formula in LaTeX:
            \begin{align*}
            \text{Information Gain} = \frac{N_{\text{parent}}}{N} \cdot \Biggl( & \text{impurity}_{\text{parent}} - \\
            & \sum^C_{i=1} \frac{N_{\text{child}_i}}{N_{\text{parent}}} \cdot \text{impurity}_{\text{child}_i} \Biggr)
            \end{align*}
            where:
            \begin{itemize}
                \item $\text{Information Gain}$ — information gain;
                \item $N$ — the number of samples in the entire training set;
                \item $N_{\text{parent}}$ — the number of samples in the parent node;
                \item $\text{impurity}_{\text{parent}}$ — the parent node impurity;
                \item $C$ — the number of child nodes;
                \item $N_{\text{child}_i}$ — the number of samples in the child node;
                \item $\text{impurity}_{\text{child}_i}$ — the child node impurity.
            \end{itemize}
        """
        N = self.dataset.size
        N_parent = parent_mask.sum()

        impurity_parent = self.impurity(parent_mask)

        weighted_impurity_childs = 0
        N_childs = 0
        for child_mask_i in child_masks:
            N_child_i = child_mask_i.sum()
            N_childs += N_child_i
            impurity_child_i = self.impurity(child_mask_i)
            weighted_impurity_childs += (N_child_i / N_parent) * impurity_child_i

        if nan_mode == "include_all":
            norm_coef = N_parent / N_childs
            weighted_impurity_childs *= norm_coef

        local_information_gain = impurity_parent - weighted_impurity_childs

        information_gain = (N_parent / N) * local_information_gain

        return information_gain

    def gini_index(self, mask: pd.Series) -> float:
        r"""
        Calculates Gini index in a tree node.

        Gini index formula in LaTeX:
            \text{Gini Index} = 1 - \sum^C_{i=1} p_i^2
            where
            C - total number of classes;
            p_i - the probability of choosing a sample with class i.
        """
        N = mask.sum()

        gini_index = 1
        for label in self.dataset.class_names:
            N_i = (mask & (self.dataset.y == label)).sum()
            p_i = N_i / N
            gini_index -= pow(p_i, 2)

        return gini_index

    def entropy(self, mask: pd.Series) -> float:
        r"""
        Calculates entropy in a tree node.

        Entropy formula in LaTeX:
        H = \log{\overline{N}} = \sum^N_{i=1} p_i \log{(1/p_i)} = -\sum^N_{i=1} p_i \log{p_i}
        where
        H - entropy;
        \overline{N} - effective number of states;
        p_i - probability of the i-th system state.
        """
        N = mask.sum()

        entropy = 0
        for label in self.dataset.class_names:
            N_i = (mask & (self.dataset.y == label)).sum()
            if N_i != 0:
                p_i = N_i / N
                entropy -= p_i * math.log2(p_i)

        return entropy


class NumericalColumnSplitter(BaseColumnSplitter):

    def __init__(
        self,
        dataset: Dataset,
        criterion: ClassificationCriterionType,
        min_samples_split: int,
        min_samples_leaf: int,
        numerical_nan_mode: NumericalNanModeType,
    ) -> None:

        super().__init__(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        self.numerical_nan_mode = numerical_nan_mode

    def split(self, node: TreeNode, split_feature_name: str) -> ColumnSplitResult:
        """Finds the best tree node split by set numerical feature, if it exists."""
        use_including_na = (
            self.numerical_nan_mode == "include"
            # and there are samples with missing values
            and (node.mask & self.dataset.mask_na[split_feature_name]).sum()
        )

        if use_including_na:
            mask_notna = node.mask & ~self.dataset.mask_na[split_feature_name]
            # if split by feature value is not possible
            if mask_notna.sum() <= 1:
                return ColumnSplitResult(NO_INFORMATION_GAIN, [], [])
            mask_na = node.mask & self.dataset.mask_na[split_feature_name]

            points = self.dataset.X.loc[mask_notna, split_feature_name].to_numpy()
        else:
            points = self.dataset.X.loc[node.mask, split_feature_name].to_numpy()

        thresholds = self.get_thresholds(points)

        best_split_result = ColumnSplitResult(NO_INFORMATION_GAIN, [], [])
        for threshold in thresholds:
            mask_less = node.mask & (self.dataset.X[split_feature_name] <= threshold)
            mask_more = node.mask & (self.dataset.X[split_feature_name] > threshold)

            if use_including_na:
                mask_less = mask_less | mask_na
                mask_more = mask_more | mask_na

            if (
                mask_less.sum() < self.min_samples_leaf
                or mask_more.sum() < self.min_samples_leaf
            ):
                continue

            child_masks = [mask_less, mask_more]

            inf_gain = self.information_gain(
                node.mask, child_masks, nan_mode=self.numerical_nan_mode
            )

            if best_split_result.information_gain < inf_gain:
                less_values = [f"<= {threshold}"]
                more_values = [f"> {threshold}"]
                feature_values = [less_values, more_values]

                best_split_result = ColumnSplitResult(
                    inf_gain, feature_values, child_masks
                )

        return best_split_result

    def get_thresholds(self, array: np.ndarray) -> np.ndarray:
        array.sort()
        array = np.unique(array)
        thresholds = np.array([]) if len(array) == 1 else self.moving_average(array, 2)

        return thresholds

    @staticmethod
    def moving_average(array: np.ndarray, window: int) -> np.ndarray:
        return np.convolve(array, np.ones(window), mode="valid") / window


class CategoricalColumnSplitter(BaseColumnSplitter):

    def __init__(
        self,
        dataset: Dataset,
        criterion: ClassificationCriterionType,
        min_samples_split: int,
        min_samples_leaf: int,
        max_leaf_nodes: int | float,
        max_childs: int | float,
        categorical_nan_mode: CategoricalNanModeType,
    ) -> None:

        super().__init__(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        self.max_leaf_nodes = max_leaf_nodes
        self.max_childs = max_childs
        self.categorical_nan_mode = categorical_nan_mode

    def split(
        self,
        node: TreeNode,
        split_feature_name: str,
        leaf_counter: int,
    ) -> ColumnSplitResult:
        """Split a node according to a categorical feature in the best way."""
        category_column: pd.Series = self.dataset.X.loc[node.mask, split_feature_name]  # type: ignore
        categories = category_column.dropna().unique().tolist()

        if len(categories) <= 1:
            return ColumnSplitResult(NO_INFORMATION_GAIN, [], [])

        best_split_result = ColumnSplitResult(NO_INFORMATION_GAIN, [], [])
        for cat_partitions in self.cat_partitions(categories):  # type: ignore
            # if partitions is not really partitions
            if len(cat_partitions) <= 1:
                continue
            # limitation of branching
            if len(cat_partitions) > self.max_childs:
                continue
            # if the number of leaves exceeds the limit after splitting
            if leaf_counter + len(cat_partitions) > self.max_leaf_nodes:
                continue

            information_gain, child_masks = self.__cat_split(
                node.mask, split_feature_name, cat_partitions
            )
            if best_split_result.information_gain < information_gain:
                best_split_result = ColumnSplitResult(
                    information_gain, cat_partitions, child_masks
                )

        return best_split_result

    def __cat_split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
        feature_values: list[list],
    ) -> tuple[float, list[pd.Series]]:
        """
        Split a node according to a categorical feature according to
        the defined feature values.
        """
        mask_na = parent_mask & self.dataset.mask_na[split_feature_name]

        child_masks = []
        for partition in feature_values:
            partition_mask = self.dataset.X[split_feature_name].isin(partition)
            child_mask = parent_mask & partition_mask
            child_masks.append(child_mask)

        if self.categorical_nan_mode == "as_category":
            information_gain = self.information_gain(parent_mask, child_masks)
            return information_gain, child_masks

        elif self.categorical_nan_mode == "include_all":
            for i, child_mask in enumerate(child_masks):
                child_masks[i] = child_mask | (parent_mask & mask_na)  # update
                if child_masks[i].sum() < self.min_samples_leaf:
                    return NO_INFORMATION_GAIN, []

            information_gain = self.information_gain(
                parent_mask, child_masks, nan_mode=self.categorical_nan_mode
            )

            return information_gain, child_masks

        elif self.categorical_nan_mode == "include_best":
            candidates = []
            origin_child_masks = child_masks
            for i, child_mask in enumerate(origin_child_masks):
                child_masks = deepcopy(origin_child_masks)
                child_masks[i] = child_mask | (parent_mask & mask_na)  # update
                for child_mask in child_masks:
                    if child_mask.sum() < self.min_samples_leaf:
                        break
                else:
                    candidates.append(child_masks)

            best_information_gain = NO_INFORMATION_GAIN
            best_child_masks = []
            for child_masks in candidates:
                information_gain = self.information_gain(parent_mask, child_masks)
                if best_information_gain < information_gain:
                    best_information_gain = information_gain
                    best_child_masks = child_masks

            return best_information_gain, best_child_masks

        else:
            assert False


    def cat_partitions(
        self,
        collection: list,
    ) -> Generator[list[list], None, None]:
        """Reference: https://en.wikipedia.org/wiki/Partition_of_a_set."""
        if len(collection) == 1:
            yield [collection]
            return

        first = collection[0]
        for smaller in self.cat_partitions(collection[1:]):
            # insert `first` in each of the subpartition's subsets
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
            # put `first` in its own subset
            yield [[first]] + smaller


class RankColumnSplitter(BaseColumnSplitter):
    def __init__(
        self,
        dataset: Dataset,
        criterion: ClassificationCriterionType,
        min_samples_split: int,
        min_samples_leaf: int,
        rank_feature_names: dict[str, list],
    ) -> None:

        super().__init__(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        self.rank_feature_names = rank_feature_names

    def split(self, node: TreeNode, split_feature_name: str) -> ColumnSplitResult:
        """Split a node according to a rank feature in the best way."""
        available_feature_values = self.rank_feature_names[split_feature_name]

        best_split_result = ColumnSplitResult(NO_INFORMATION_GAIN, [], [])
        for feature_values in self.rank_partitions(available_feature_values):
            inf_gain, child_masks = \
                self.__rank_split(node.mask, split_feature_name, feature_values)
            if best_split_result.information_gain < inf_gain:
                best_split_result = ColumnSplitResult(
                    inf_gain, list(feature_values), child_masks
                )

        return best_split_result

    def __rank_split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
        feature_values: tuple[list, list],
    ) -> tuple[float, list[pd.Series]]:
        """
        Splits a node according to a rank feature according to the defined feature
        values.
        """
        feature_values_left, feature_values_right = feature_values

        mask_left = parent_mask & self.dataset.X[split_feature_name].isin(feature_values_left)
        mask_right = parent_mask & self.dataset.X[split_feature_name].isin(feature_values_right)

        if (
            mask_left.sum() < self.min_samples_leaf
            or mask_right.sum() < self.min_samples_leaf
        ):
            return NO_INFORMATION_GAIN, []

        child_masks = [mask_left, mask_right]

        information_gain = self.information_gain(parent_mask, child_masks)

        return information_gain, child_masks

    @staticmethod
    def rank_partitions(collection: list) -> Generator[tuple[list, list], None, None]:
        for i in range(1, len(collection)):
            yield collection[:i], collection[i:]
