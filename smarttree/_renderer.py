from graphviz import Digraph

from ._tree_node import TreeNode
from ._types import ClassificationCriterionType


class Renderer:
    def __init__(self, rounded: bool, criterion: ClassificationCriterionType) -> None:
        node_attr = {"shape": "box"}
        if rounded:
            node_attr["style"] = "rounded"
        self.graph = Digraph(name="decision tree", node_attr=node_attr)

        self.criterion = criterion

    def render(
        self,
        tree: TreeNode,
        *,
        show_impurity: bool = False,
        show_num_samples: bool = False,
        show_distribution: bool = False,
        show_label: bool = False,
        **kwargs,
    ) -> Digraph:
        self.__add_node(
            node=tree,
            parent_name=None,
            show_impurity=show_impurity,
            show_num_samples=show_num_samples,
            show_distribution=show_distribution,
            show_label=show_label,
        )
        if kwargs:
            self.graph.render(**kwargs)

        return self.graph

    def __add_node(
        self,
        node: TreeNode,
        parent_name: str | None,
        show_impurity: bool,
        show_num_samples: bool,
        show_distribution: bool,
        show_label: bool,
    ) -> None:
        """
        Recursively adds a description of the node and its relationship to the parent
        node (if available).
        """
        node_name = f"node {node.number}"

        node_content_buffer = [f"Node {node.number}"]
        if node.split_feature_name:
            node_content_buffer.append(f"{node.split_feature_name}")
        if show_impurity:
            node_content_buffer.append(f"{self.criterion} = {node.impurity:.2f}")
        if show_num_samples:
            node_content_buffer.append(f"samples = {node.num_samples}")
        if show_distribution:
            node_content_buffer.append(f"distribution = {node.distribution}")
        if show_label:
            node_content_buffer.append(f"label = {node.label}")
        node_content = "\n".join(node_content_buffer)

        self.graph.node(name=node_name, label=node_content)

        if parent_name:
            edge_label = "\n".join(map(str, node.feature_value))
            self.graph.edge(
                tail_name=parent_name,
                head_name=node_name,
                label=edge_label,
            )

        for child in node.childs:
            self.__add_node(
                node=child,
                parent_name=node_name,
                show_impurity=show_impurity,
                show_num_samples=show_num_samples,
                show_distribution=show_distribution,
                show_label=show_label,
            )
