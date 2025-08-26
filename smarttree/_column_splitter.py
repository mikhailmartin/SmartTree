from abc import abstractmethod
import math

import numpy as np
import pandas as pd

# from smarttree._utils import cat_partitions, rank_partitions


class BaseColumnSplitter:

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        criterion,
        max_depth,
        min_samples_split,
        min_samples_leaf,
    ) -> None:
        self.X = X
        self.y = y
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        match self.criterion:
            case "gini":
                self.impurity = self.gini_index
            case "entropy" | "log_loss":
                self.impurity = self.entropy

        if self.criterion in ("gini", "entropy", "log_loss"):
            self.class_names = sorted(y.unique())

    @abstractmethod
    def split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
    ) -> tuple[float, list[list[str]] | None, list[pd.Series] | None]:
        raise NotImplementedError

    def information_gain(
        self,
        parent_mask: pd.Series,
        child_masks: list[pd.Series],
        nan_mode: str | None = None,
    ) -> float:
        """
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
        """
        Calculates Gini index in a tree node.

        Gini index formula in LaTeX:
            \text{Gini Index} = \sum^C_{i=1} p_i \times (1 - p_i)
            where
            \text{Gini Index} - Gini index;
            C - total number of classes;
            p_i - the probability of choosing a sample with class i.
        """
        N = mask.sum()

        gini_index = 0
        for label in self.class_names:
            N_i = (mask & (self.y == label)).sum()
            p_i = N_i / N
            gini_index += p_i * (1 - p_i)

        return gini_index

    def entropy(self, mask: pd.Series) -> float:
        # TODO: посмотреть разницу между Джини индексом и проч
        """
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
        max_depth,
        min_samples_split,
        min_samples_leaf,
        numerical_nan_mode,  # TODO: annotation
    ) -> None:
        super().__init__(
            X=X,
            y=y,
            criterion=criterion,
            max_depth=max_depth,
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
