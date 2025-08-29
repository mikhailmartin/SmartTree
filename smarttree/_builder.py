import bisect
import math
from collections import defaultdict

import pandas as pd

from ._tree_node import TreeNode
from ._constants import ClassificationCriterionOption
from ._node_splitter import NodeSplitter


class Builder:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        criterion: ClassificationCriterionOption,
        splitter: NodeSplitter,
        max_leaf_nodes: int,
    ) -> None:
        self.X = X
        self.y = y
        self.criterion = criterion
        self.splitter = splitter
        self.max_leaf_nodes = max_leaf_nodes

        match self.criterion:
            case "gini":
                self.impurity = self.gini_index
            case "entropy" | "log_loss":
                self.impurity = self.entropy

        if self.criterion in ("gini", "entropy", "log_loss"):
            self.class_names = sorted(self.y.unique())

        self.node_counter: int = 0

    def build(self, hierarchy, available_feature_names) -> TreeNode:
        """TODO."""
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
            best_node = splittable_leaf_nodes.pop()
            feature_importances[best_node.split_result.split_feature_name] += best_node.split_result.inf_gain

            for child_mask, feature_value in zip(
                best_node.split_result.child_masks,
                best_node.split_result.feature_values
            ):
                # add opened features
                if best_node.split_result.split_feature_name in best_node.hierarchy:
                    value = best_node.hierarchy.pop(best_node.split_result.split_feature_name)
                    if isinstance(value, str):
                        best_node.available_feature_names.append(value)
                    elif isinstance(value, list):
                        best_node.available_feature_names.extend(value)
                    else:
                        assert False

                child_node = self.create_node(
                    mask=child_mask,
                    hierarchy=best_node.hierarchy,
                    available_feature_names=best_node.available_feature_names,
                    depth=best_node.depth + 1,
                )
                child_node.feature_value = feature_value
                self.splitter.leaf_counter += 1

                best_node.childs.append(child_node)
                if self.splitter.is_splittable(child_node):
                    bisect.insort(
                        splittable_leaf_nodes,
                        child_node,
                        key=lambda x: x.split_result.inf_gain,
                    )

            best_node.is_leaf = False
            best_node.split_type = best_node.split_result.split_type
            best_node.split_feature_name = best_node.split_result.split_feature_name
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
            samples=mask.sum(),
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
        # TODO: посмотреть разницу между Джини индексом и проч
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
