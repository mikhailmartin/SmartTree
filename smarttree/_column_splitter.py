import math
from abc import ABC, abstractmethod
from typing import Generator

import numpy as np
import pandas as pd

from ._constants import (
    CategoricalNanModeOption,
    ClassificationCriterionOption,
    NumericalNanModeOption,
)


class BaseColumnSplitter(ABC):

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        criterion: ClassificationCriterionOption,
        min_samples_split: int,
        min_samples_leaf: int,
    ) -> None:
        self.X = X
        self.y = y
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        match self.criterion:
            case "gini":
                self.impurity = self.gini_index
            case "entropy" | "log_loss":
                self.impurity = self.entropy

        if self.criterion in ("gini", "entropy", "log_loss"):
            self.class_names = sorted(self.y.unique())

    @abstractmethod
    def split(
        self, *args, **kwargs
    ) -> tuple[float, list[list[str]] | None, list[pd.Series] | None]:
        pass

    def information_gain(
        self,
        parent_mask: pd.Series,
        child_masks: list[pd.Series],
        nan_mode: str | None = None,
    ) -> float:
        r"""
        Calculates information gain of the split.

        Parameters:
            parent_mask: boolean mask of parent node.
            child_masks: list of boolean masks of child nodes.
            nan_mode: missing values handling node.
              If 'include', then turn on normalization of child nodes impurity.

        Returns:
            information gain.

        References:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

        Formula in LaTeX:
            \text{Information Gain} = \frac{N_{\text{parent}}}{N} * \Big(\text{impurity}_{\text{parent}} - \sum^C_{i=1}{\frac{N_{\text{child}_i}}{N_{\text{parent}}}} * \text{impurity}_{\text{child}_i} \Big)
            where
            \text{Information Gain} - information fain;
            N - the number of samples in the entire training set;
            N_{\text{parent}} - the number of samples in the parent node;
            \text{impurity}_{\text{parent}} - the parent node impurity;
            С - the number of child nodes;
            N_{\text{child}_i} - the number of samples in the child node;
            \text{impurity}_{\text{child}_i} - the child node impurity.
        """
        N = self.y.shape[0]
        N_parent = parent_mask.sum()

        impurity_parent = self.impurity(parent_mask)

        weighted_impurity_childs = 0
        N_childs = 0
        for child_mask_i in child_masks:
            N_child_i = child_mask_i.sum()
            N_childs += N_child_i
            impurity_child_i = self.impurity(child_mask_i)
            weighted_impurity_childs += (N_child_i / N_parent) * impurity_child_i

        if nan_mode == "include":
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
        for label in self.class_names:
            N_i = (mask & (self.y == label)).sum()
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
        for label in self.class_names:
            N_i = (mask & (self.y == label)).sum()
            if N_i != 0:
                p_i = N_i / N
                entropy -= p_i * math.log2(p_i)

        return entropy


class NumericalColumnSplitter(BaseColumnSplitter):

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        criterion,
        min_samples_split: int,
        min_samples_leaf: int,
        numerical_nan_mode: NumericalNanModeOption,
    ) -> None:
        super().__init__(
            X=X,
            y=y,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        self.numerical_nan_mode = numerical_nan_mode

    def split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
    ) -> tuple[float, list[list[str]] | None, list[pd.Series] | None]:
        """
        Finds the best tree node split by set numerical feature, if it exists.

        Parameters:
            parent_mask: boolean mask of the tree node.
            split_feature_name: The name of the set numerical feature by which to find
              the best split.

        Returns:
            Tuple `(inf_gain, feature_values, child_masks)`.
              inf_gain: information gain of the split.
              feature_values: feature values corresponding to child nodes.
              child_masks: boolean masks of child nodes.
        """
        use_including_na = (
            self.numerical_nan_mode == "include"
            # and there are samples with missing values
            and (parent_mask & self.X[split_feature_name].isna()).sum()
        )

        if use_including_na:
            mask_notna = parent_mask & self.X[split_feature_name].notna()
            # if split by feature value is not possible
            if mask_notna.sum() <= 1:
                return float("-inf"), None, None
            mask_na = parent_mask & self.X[split_feature_name].isna()

            points = self.X.loc[mask_notna, split_feature_name].to_numpy()
        else:
            points = self.X.loc[parent_mask, split_feature_name].to_numpy()

        thresholds = self.get_thresholds(points)

        best_inf_gain = float("-inf")
        best_feature_values = None
        best_child_masks = None
        for threshold in thresholds:
            mask_less = parent_mask & (self.X[split_feature_name] <= threshold)
            mask_more = parent_mask & (self.X[split_feature_name] > threshold)

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
                parent_mask, child_masks, nan_mode=self.numerical_nan_mode
            )

            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                less_values = [f"<= {threshold}"]
                more_values = [f"> {threshold}"]
                best_feature_values = [less_values, more_values]
                best_child_masks = child_masks

        return best_inf_gain, best_feature_values, best_child_masks

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
        X: pd.DataFrame,
        y: pd.Series,
        criterion: ClassificationCriterionOption,
        min_samples_split: int,
        min_samples_leaf: int,
        max_leaf_nodes: int | float,
        max_childs: int | float,
        categorical_nan_mode: CategoricalNanModeOption,
    ) -> None:
        super().__init__(
            X=X,
            y=y,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        self.max_leaf_nodes = max_leaf_nodes
        self.max_childs = max_childs
        self.categorical_nan_mode = categorical_nan_mode

    def split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
        leaf_counter: int,
    ) -> tuple[float, list[list[str]] | None, list[pd.Series] | None]:
        """
        Split a node according to a categorical feature in the best way.

        Parameters:
            parent_mask: boolean mask of split node.
            split_feature_name: feature according to which node should be split.

        Returns:
            Tuple `(inf_gain, feature_values, child_masks)`.
              inf_gain: information gain of the split.
              feature_values: feature values corresponding to child nodes.
              child_masks: boolean masks of child nodes.
        """
        available_feature_values = self.X.loc[parent_mask, split_feature_name].unique()
        if (
            self.categorical_nan_mode == "include"
            and pd.isna(available_feature_values).any()  # if contains missing values
        ):
            available_feature_values = available_feature_values[
                ~pd.isna(available_feature_values)]
        if len(available_feature_values) <= 1:
            return float("-inf"), None, None
        available_feature_values = sorted(available_feature_values)

        # get list of all possible partitions
        partitions = []
        for partition in self.cat_partitions(available_feature_values):
            # if partitions is not really partitions
            if len(partition) < 2:
                continue
            # limitation of branching
            if len(partition) > self.max_childs:
                continue
            # if the number of leaves exceeds the limit after splitting
            if leaf_counter + len(partition) > self.max_leaf_nodes:
                continue

            partitions.append(partition)

        best_inf_gain = float("-inf")
        best_feature_values = None
        best_child_masks = None
        for feature_values in partitions:
            inf_gain, child_masks = \
                self.__cat_split(parent_mask, split_feature_name, feature_values)
            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_child_masks = child_masks
                best_feature_values = feature_values

        return best_inf_gain, best_feature_values, best_child_masks

    def __cat_split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
        feature_values: list[list],
    ) -> tuple[float, list[pd.Series] | None]:
        """
        Split a node according to a categorical feature according to the
        defined feature values.

        Parameters:
            parent_mask: boolean mask of split node.
            split_feature_name: feature according to which node should be split.
            feature_values: feature values corresponding to child nodes.

        Returns:
            Tuple `(inf_gain, child_masks)`.
              inf_gain: information gain of the split.
              child_masks: boolean masks of child nodes.
        """
        mask_na = parent_mask & self.X[split_feature_name].isna()

        child_masks = []
        for list_ in feature_values:
            child_mask = parent_mask & (self.X[split_feature_name].isin(list_) | mask_na)
            if child_mask.sum() < self.min_samples_leaf:
                return float("-inf"), None
            child_masks.append(child_mask)

        inf_gain = self.information_gain(
            parent_mask, child_masks, nan_mode=self.categorical_nan_mode
        )

        return inf_gain, child_masks

    def cat_partitions(
        self, collection: list
    ) -> Generator[list[list[list]], None, None]:
        """
        References:
            https://en.wikipedia.org/wiki/Partition_of_a_set
        """
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
        X: pd.DataFrame,
        y: pd.Series,
        criterion,
        min_samples_split: int,
        min_samples_leaf: int,
        rank_feature_names: dict[str, list],
    ) -> None:
        super().__init__(
            X=X,
            y=y,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        self.rank_feature_names = rank_feature_names

    def split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
    ) -> tuple[float, list[list[str]] | None, list[pd.Series] | None]:
        """Split a node according to a rank feature in the best way."""
        available_feature_values = self.rank_feature_names[split_feature_name]

        best_inf_gain = float("-inf")
        best_child_masks = None
        best_feature_values = None
        for feature_values in self.rank_partitions(available_feature_values):
            inf_gain, child_masks = \
                self.__rank_split(parent_mask, split_feature_name, feature_values)
            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_child_masks = child_masks
                best_feature_values = feature_values

        return best_inf_gain, best_feature_values, best_child_masks

    def __rank_split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
        feature_values: list[list[str]],
    ) -> tuple[float, list[pd.Series] | None]:
        """
        Splits a node according to a rank feature according to the defined feature
        values.
        """
        left_list_, right_list_ = feature_values

        mask_left = parent_mask & self.X[split_feature_name].isin(left_list_)
        mask_right = parent_mask & self.X[split_feature_name].isin(right_list_)

        if (
            mask_left.sum() < self.min_samples_leaf
            or mask_right.sum() < self.min_samples_leaf
        ):
            return float("-inf"), None

        child_masks = [mask_left, mask_right]

        inf_gain = self.information_gain(parent_mask, child_masks)

        return inf_gain, child_masks

    @staticmethod
    def rank_partitions(collection: list) -> Generator[tuple[list, list], None, None]:
        for i in range(1, len(collection)):
            yield collection[:i], collection[i:]
