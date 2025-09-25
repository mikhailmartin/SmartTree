from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from copy import deepcopy
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._cy_column_splitter import CyBaseColumnSplitter
from ._dataset import Dataset
from ._tree import TreeNode
from ._types import ClassificationCriterionType, Criterion, NaModeType


NO_INFORMATION_GAIN = float("-inf")


class ColumnSplitResult(NamedTuple):

    information_gain: float
    feature_values: list[list]
    child_masks: list[pd.Series]
    child_na_index: int = -1

    @classmethod
    def no_split(cls) -> ColumnSplitResult:
        return cls(
            information_gain=NO_INFORMATION_GAIN,
            feature_values=[],
            child_masks=[],
        )


class BaseColumnSplitter(ABC):

    mapping: dict[ClassificationCriterionType, Criterion] = {
        "gini": Criterion.GINI,
        "entropy": Criterion.ENTROPY,
        "log_loss": Criterion.LOG_LOSS,
    }

    def __init__(
        self,
        dataset: Dataset,
        criterion: ClassificationCriterionType,
        min_samples_split: int,
        min_samples_leaf: int,
        feature_na_mode: dict[str, NaModeType],
    ) -> None:

        self.dataset = dataset
        self.criterion = self.mapping[criterion]
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_na_mode = feature_na_mode

    @abstractmethod
    def split(self, *args, **kwargs) -> ColumnSplitResult:
        raise NotImplementedError

    def foo(
        self,
        parent_mask: pd.Series,
        split_feature: str,
        child_masks: list[pd.Series],
    ) -> tuple[float, list[pd.Series], int]:

        if self.dataset.has_na[split_feature]:
            mask_na = parent_mask & self.dataset.mask_na[split_feature]
            na_mode = self.feature_na_mode[split_feature]
            if na_mode == "include_all":
                return self.include_all_split(parent_mask, mask_na, child_masks)
            elif na_mode == "include_best":
                return self.include_best_split(parent_mask, mask_na, child_masks)
            else:
                assert False
        else:
            information_gain = self.information_gain(parent_mask, child_masks)
            return information_gain, child_masks, -1

    def include_all_split(
        self,
        parent_mask: pd.Series,
        mask_na: pd.Series,
        child_masks: list[pd.Series],
    ) -> tuple[float, list[pd.Series], int]:

        for i, child_mask in enumerate(child_masks):
            child_masks[i] = child_mask | (parent_mask & mask_na)
            if child_masks[i].sum() < self.min_samples_leaf:
                return NO_INFORMATION_GAIN, [], -1

        information_gain = self.information_gain(parent_mask, child_masks, normalize=True)

        return information_gain, child_masks, -1

    def include_best_split(
        self,
        parent_mask: pd.Series,
        mask_na: pd.Series,
        child_masks: list[pd.Series],
    ) -> tuple[float, list[pd.Series], int]:

        candidates = []
        origin_child_masks = child_masks
        for i, child_mask in enumerate(origin_child_masks):
            child_masks = deepcopy(origin_child_masks)
            child_masks[i] = child_mask | (parent_mask & mask_na)
            for child_mask in child_masks:
                if child_mask.sum() < self.min_samples_leaf:
                    break
            else:
                candidates.append(child_masks)

        best_information_gain = NO_INFORMATION_GAIN
        best_child_masks = []
        best_child_na_index = -1
        for child_na_index, child_masks in enumerate(candidates):
            information_gain = self.information_gain(parent_mask, child_masks)
            if best_information_gain < information_gain:
                best_information_gain = information_gain
                best_child_masks = child_masks
                best_child_na_index = child_na_index

        return best_information_gain, best_child_masks, best_child_na_index

    def information_gain(
        self,
        parent_mask: pd.Series,
        child_masks: list[pd.Series],
        normalize: bool = False,
    ) -> float:
        r"""
        Calculates information gain of the split.

        Parameters:
            parent_mask: pd.Series
              boolean mask of parent node.
            child_masks: pd.Series
              list of boolean masks of child nodes.
            normalize: bool, default=False
              if True, normalizes information gain by split factor to handle
              unbalanced splits. Uses child node counts for normalization.

        Returns:
            float: information gain.

        Formula in LaTeX:
            \begin{align*}
            \text{Information Gain} =
            \frac{N_{\text{parent}}}{N} \cdot
            \Biggl( & \text{impurity}_{\text{parent}} - \\
            & \sum^C_{i=1} \frac{N_{\text{child}_i}}{N_{\text{parent}}}
            \cdot \text{impurity}_{\text{child}_i} \Biggr)
            \end{align*}
            where:
            \begin{itemize}
                \item $\text{Information Gain}$ — information gain;
                \item $N$ — number of samples in entire training set;
                \item $N_{\text{parent}}$ — number of samples in parent node;
                \item $\text{impurity}_{\text{parent}}$ — parent node impurity;
                \item $C$ — number of child nodes;
                \item $N_{\text{child}_i}$ — number of samples in child node;
                \item $\text{impurity}_{\text{child}_i}$ — child node impurity.
            \end{itemize}
        """
        cs = CyBaseColumnSplitter(self.dataset, self.criterion)
        return cs.information_gain(parent_mask, child_masks, normalize)


class NumColumnSplitter(BaseColumnSplitter):

    def __init__(
        self,
        dataset: Dataset,
        criterion: ClassificationCriterionType,
        min_samples_split: int,
        min_samples_leaf: int,
        feature_na_mode: dict[str, NaModeType],
    ) -> None:

        super().__init__(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            feature_na_mode=feature_na_mode,
        )

    def split(self, node: TreeNode, split_feature: str) -> ColumnSplitResult:

        numerical_column = self.dataset.X.loc[node.mask, split_feature]
        points = numerical_column.dropna().to_numpy()
        thresholds = self.__get_thresholds(points)

        best_split_result = ColumnSplitResult.no_split()
        for threshold in thresholds:
            information_gain, child_masks, child_na_index = self.__num_split(
                node.mask, split_feature, threshold
            )
            if best_split_result.information_gain < information_gain:
                feature_values = [[f"<= {threshold}"], [f"> {threshold}"]]
                best_split_result = ColumnSplitResult(
                    information_gain, feature_values, child_masks, child_na_index
                )

        return best_split_result

    def __get_thresholds(self, array: NDArray) -> NDArray:

        array = np.sort(np.unique(array))
        thresholds = np.array([]) if len(array) <= 1 else self.__moving_average(array)

        return thresholds

    @staticmethod
    def __moving_average(array: NDArray, window: int = 2) -> NDArray:
        return np.convolve(array, np.ones(window), mode="valid") / window

    def __num_split(
        self,
        parent_mask: pd.Series,
        split_feature: str,
        threshold: float,
    ) -> tuple[float, list[pd.Series], int]:

        mask_less = parent_mask & (self.dataset.X[split_feature] <= threshold)
        mask_more = parent_mask & (self.dataset.X[split_feature] > threshold)
        child_masks = [mask_less, mask_more]

        return self.foo(parent_mask, split_feature, child_masks)


class CatColumnSplitter(BaseColumnSplitter):

    def __init__(
        self,
        dataset: Dataset,
        criterion: ClassificationCriterionType,
        min_samples_split: int,
        min_samples_leaf: int,
        max_leaf_nodes: int | float,
        max_childs: int | float,
        feature_na_mode: dict[str, NaModeType],
    ) -> None:

        super().__init__(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            feature_na_mode=feature_na_mode,
        )
        self.max_leaf_nodes = max_leaf_nodes
        self.max_childs = max_childs

    def split(
        self,
        node: TreeNode,
        split_feature: str,
        leaf_counter: int,
    ) -> ColumnSplitResult:

        category_column: pd.Series = self.dataset.X.loc[node.mask, split_feature]  # type: ignore
        categories = category_column.dropna().unique().tolist()

        if len(categories) <= 1:
            return ColumnSplitResult.no_split()

        best_split_result = ColumnSplitResult.no_split()
        for cat_partitions in self.__cat_partitions(categories):  # type: ignore
            # if partitions is not really partitions
            if len(cat_partitions) <= 1:
                continue
            # limitation of branching
            if len(cat_partitions) > self.max_childs:
                continue
            # if the number of leaves exceeds the limit after splitting
            if leaf_counter + len(cat_partitions) > self.max_leaf_nodes:
                continue

            information_gain, child_masks, child_na_index = self.__cat_split(
                node.mask, split_feature, cat_partitions
            )
            if best_split_result.information_gain < information_gain:
                best_split_result = ColumnSplitResult(
                    information_gain, cat_partitions, child_masks, child_na_index
                )

        return best_split_result

    def __cat_split(
        self,
        parent_mask: pd.Series,
        split_feature: str,
        feature_values: list[list],
    ) -> tuple[float, list[pd.Series], int]:

        child_masks = []
        for partition in feature_values:
            partition_mask = self.dataset.X[split_feature].isin(partition)
            child_mask = parent_mask & partition_mask
            child_masks.append(child_mask)

        return self.foo(parent_mask, split_feature, child_masks)

    def __cat_partitions(
        self,
        collection: list,
    ) -> Generator[list[list], None, None]:
        """Reference: https://en.wikipedia.org/wiki/Partition_of_a_set."""
        if len(collection) == 1:
            yield [collection]
            return

        first = collection[0]
        for smaller in self.__cat_partitions(collection[1:]):
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
        rank_features: dict[str, list],
        feature_na_mode: dict[str, NaModeType],
    ) -> None:

        super().__init__(
            dataset=dataset,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            feature_na_mode=feature_na_mode,
        )
        self.rank_features = rank_features

    def split(self, node: TreeNode, split_feature: str) -> ColumnSplitResult:

        available_feature_values = self.rank_features[split_feature]

        best_split_result = ColumnSplitResult.no_split()
        for feature_values in self.__rank_partitions(available_feature_values):
            information_gain, child_masks, child_na_index = self.__rank_split(
                node.mask, split_feature, feature_values
            )
            if best_split_result.information_gain < information_gain:
                best_split_result = ColumnSplitResult(
                    information_gain, list(feature_values), child_masks, child_na_index
                )

        return best_split_result

    def __rank_split(
        self,
        parent_mask: pd.Series,
        split_feature: str,
        feature_values: tuple[list, list],
    ) -> tuple[float, list[pd.Series], int]:

        feature_values_left, feature_values_right = feature_values

        mask_left = parent_mask & self.dataset.X[split_feature].isin(feature_values_left)
        mask_right = parent_mask & self.dataset.X[split_feature].isin(feature_values_right)
        child_masks = [mask_left, mask_right]

        return self.foo(parent_mask, split_feature, child_masks)

    @staticmethod
    def __rank_partitions(collection: list) -> Generator[tuple[list, list], None, None]:
        for i in range(1, len(collection)):
            yield collection[:i], collection[i:]
