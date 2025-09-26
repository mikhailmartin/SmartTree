import bisect
import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._node_splitter import NodeSplitter
from ._tree import Tree, TreeNode
from ._types import ClassificationCriterionType


class Builder:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        criterion: ClassificationCriterionType,
        splitter: NodeSplitter,
        max_leaf_nodes: int | float,
        hierarchy: dict[str, str | list[str]],
    ) -> None:

        self.available_features = X.columns.to_list()
        self.y = y
        self.criterion = criterion
        self.splitter = splitter
        self.max_leaf_nodes = max_leaf_nodes
        self.hierarchy = hierarchy

        match self.criterion:
            case "gini":
                self.impurity = self.gini_index
            case "entropy" | "log_loss":
                self.impurity = self.entropy

        if self.criterion in ("gini", "entropy", "log_loss"):
            self.class_names = np.sort(self.y.unique())

    def build(self, tree: Tree) -> None:

        for value in self.hierarchy.values():
            if isinstance(value, list):
                for feature in value:
                    self.available_features.remove(feature)
            else:  # str
                self.available_features.remove(value)

        mask = self.y.apply(lambda x: True)
        root = tree.create_node(
            mask=mask,
            hierarchy=self.hierarchy,
            distribution=self.distribution(mask),
            impurity=self.impurity(mask),
            label=self.y[mask].mode()[0],
            available_features=self.available_features,
            depth=0,
            is_root=True,
        )

        splittable_leaf_nodes: list[TreeNode] = []

        if self.splitter.is_splittable(root, tree.leaf_counter):
            splittable_leaf_nodes.append(root)

        while len(splittable_leaf_nodes) > 0 and tree.leaf_counter < self.max_leaf_nodes:

            node = splittable_leaf_nodes.pop()
            tree.feature_importances[node.split_feature] += node.information_gain

            for child_mask, feature_value in zip(node.child_masks, node.feature_values):
                # add opened features
                if node.split_feature in node.hierarchy:
                    value = node.hierarchy.pop(node.split_feature)
                    if isinstance(value, list):  # list[str]
                        node.available_features.extend(value)
                    else:  # str
                        node.available_features.append(value)

                child_node = tree.create_node(
                    mask=child_mask,
                    hierarchy=node.hierarchy,
                    distribution=self.distribution(child_mask),
                    impurity=self.impurity(child_mask),
                    label=self.y[child_mask].mode()[0],
                    available_features=node.available_features,
                    depth=node.depth+1,
                )
                child_node.feature_value = feature_value

                node.childs.append(child_node)
                if self.splitter.is_splittable(child_node, tree.leaf_counter):
                    bisect.insort(
                        splittable_leaf_nodes,
                        child_node,
                        key=lambda n: n.information_gain,
                    )

            node.is_leaf = False
            tree.leaf_counter -= 1

    def distribution(self, mask: pd.Series) -> NDArray[np.integer]:

        mask_arr = mask.to_numpy()
        y_arr = self.y.to_numpy()

        result = np.zeros(len(self.class_names), dtype=np.int32)
        for i, class_name in enumerate(self.class_names):
            result[i] = np.sum(mask_arr & (y_arr == class_name))

        return result

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
