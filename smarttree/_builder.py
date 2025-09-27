import bisect

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._criterion import ClassificationCriterion, Entropy, Gini
from ._dataset import Dataset
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

        self.X = X
        self.y = y
        self.dataset = Dataset(X, y)
        self.available_features = X.columns.to_list()
        self.splitter = splitter
        self.max_leaf_nodes = max_leaf_nodes
        self.hierarchy = hierarchy

        self.criterion: ClassificationCriterion
        if criterion == "gini":
            self.criterion = Gini(self.dataset)
        else:  # "entropy" | "log_loss"
            self.criterion = Entropy(self.dataset)

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
            impurity=self.criterion.impurity(mask.to_numpy(dtype=np.int8)),
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
                    impurity=self.criterion.impurity(mask.to_numpy(dtype=np.int8)),
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

        result = np.zeros(len(self.dataset.classes), dtype=np.int32)
        for i, class_name in enumerate(self.dataset.classes):
            result[i] = np.sum(mask_arr & (y_arr == class_name))

        return result
