import bisect

import numpy as np

from ._criterion import Criterion, Entropy, Gini
from ._dataset import Dataset
from ._node_splitter import NodeSplitter
from ._tree import Tree, TreeNode
from ._types import CriterionType


class Builder:

    def __init__(
        self,
        dataset: Dataset,
        criterion: CriterionType,
        splitter: NodeSplitter,
        max_leaf_nodes: int | float,
        hierarchy: dict[str, str | list[str]],
    ) -> None:

        self.dataset = dataset
        self.available_features = list(dataset.columns)
        self.splitter = splitter
        self.max_leaf_nodes = max_leaf_nodes
        self.hierarchy = hierarchy

        self.criterion: Criterion
        if criterion == "gini":
            self.criterion = Gini(self.dataset)
        elif criterion in ("entropy", "log_loss"):
            self.criterion = Entropy(self.dataset)
        else:
            assert False

    def build(self, tree: Tree) -> None:

        for value_to_close in self.hierarchy.values():
            if isinstance(value_to_close, list):
                for feature in value_to_close:
                    self.available_features.remove(feature)
            else:  # str
                self.available_features.remove(value_to_close)

        mask = np.ones(self.dataset.n_samples, dtype=bool)
        value = np.frombuffer(self.criterion.value(mask), dtype=np.int64)
        label = self.dataset.classes[value.argmax()]
        root = tree.create_node(
            mask=mask,
            hierarchy=self.hierarchy,
            distribution=value,
            impurity=self.criterion.impurity(mask),
            label=label,
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
                    value_to_open = node.hierarchy.pop(node.split_feature)
                    if isinstance(value_to_open, list):  # list[str]
                        node.available_features.extend(value_to_open)
                    else:  # str
                        node.available_features.append(value_to_open)

                distribution = np.frombuffer(
                    self.criterion.value(child_mask), dtype=np.int64
                )
                label = self.dataset.classes[distribution.argmax()]
                child_node = tree.create_node(
                    mask=child_mask,
                    hierarchy=node.hierarchy,
                    distribution=distribution,
                    impurity=self.criterion.impurity(child_mask),
                    label=label,
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
