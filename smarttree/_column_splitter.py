from abc import abstractmethod
import math

import pandas as pd

from smarttree._utils import cat_partitions, get_thresholds, rank_partitions


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
