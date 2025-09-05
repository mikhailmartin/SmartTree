import bisect
import math
from collections import defaultdict

import pandas as pd

from ._constants import ClassificationCriterionOption
from ._node_splitter import NodeSplitter
from ._tree_node import TreeNode


class Builder:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        criterion: ClassificationCriterionOption,
        splitter: NodeSplitter,
        max_leaf_nodes: int | float,
        hierarchy: dict[str, str | list[str]],
    ) -> None:
        self.X = X
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
            self.class_names = sorted(self.y.unique())

        self.node_counter: int = 0

    def build(self) -> tuple[TreeNode, defaultdict[str, float]]:
        hierarchy = self.hierarchy.copy()
        available_feature_names = self.X.columns.tolist()
        # remove those features that cannot be considered yet
        for value in hierarchy.values():
            if isinstance(value, str):
                available_feature_names.remove(value)
            elif isinstance(value, list):
                for feature_name in value:
                    available_feature_names.remove(feature_name)
            else:
                assert False

        root = self.create_node(
            mask=self.y.apply(lambda x: True),
            hierarchy=hierarchy,
            available_feature_names=available_feature_names,
            depth=0,
        )

        splittable_leaf_nodes: list[TreeNode] = []
        feature_importances = defaultdict(float)

        if self.splitter.is_splittable(root):
            splittable_leaf_nodes.append(root)

        while (
            len(splittable_leaf_nodes) > 0
            and self.splitter.leaf_counter < self.max_leaf_nodes
        ):
            node = splittable_leaf_nodes.pop()
            feature_importances[node.split_feature_name] += node.information_gain

            for child_mask, feature_value in zip(node.child_masks, node.feature_values):
                # add opened features
                if node.split_feature_name in node.hierarchy:
                    value = node.hierarchy.pop(node.split_feature_name)
                    if isinstance(value, list):  # list[str]
                        node.available_feature_names.extend(value)
                    else:  # str
                        node.available_feature_names.append(value)

                child_node = self.create_node(
                    mask=child_mask,
                    hierarchy=node.hierarchy,
                    available_feature_names=node.available_feature_names,
                    depth=node.depth + 1,
                )
                child_node.feature_value = feature_value
                self.splitter.leaf_counter += 1

                node.childs.append(child_node)
                if self.splitter.is_splittable(child_node):
                    bisect.insort(
                        splittable_leaf_nodes,
                        child_node,
                        key=lambda x: x.information_gain,
                    )

            node.is_leaf = False
            self.splitter.leaf_counter -= 1

        return root, feature_importances

    def create_node(
        self,
        mask: pd.Series,
        hierarchy: dict[str, str | list[str]],
        available_feature_names: list[str],
        depth: int,
    ) -> TreeNode:
        """Creates a node of the tree."""
        tree_node = TreeNode(
            number=self.node_counter,
            num_samples=mask.sum(),
            distribution=self.distribution(mask),
            impurity=self.impurity(mask),
            label=self.y[mask].value_counts().index[0],
            depth=depth,
            mask=mask,
            hierarchy=hierarchy.copy(),
            available_feature_names=available_feature_names.copy(),
        )
        self.node_counter += 1
        return tree_node

    def distribution(self, mask: pd.Series) -> list[int]:
        """Calculates the class distribution."""
        distribution = [
            (mask & (self.y == class_name)).sum()
            for class_name in self.class_names
        ]

        return distribution

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
