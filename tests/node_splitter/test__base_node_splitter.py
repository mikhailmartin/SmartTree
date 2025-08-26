import pytest

from smarttree._node_splitter import BaseNodeSplitter
from smarttree._tree_node import TreeNode


@pytest.fixture(scope="module")
def concrete_node_splitter(
    X, y, numerical_feature_names, categorical_feature_names, rank_feature_names
) -> BaseNodeSplitter:
    class ConcreteNodeSplitter(BaseNodeSplitter):
        pass

    return ConcreteNodeSplitter(
        X=X,
        y=y,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=float("+inf"),
        max_childs=float("+inf"),
        numerical_feature_names=numerical_feature_names,
        categorical_feature_names=categorical_feature_names,
        rank_feature_names=rank_feature_names,
        numerical_nan_mode="min",
        categorical_nan_mode="include",
    )


def test__is_splittable(concrete_node_splitter, X):
    node = TreeNode(
        number=0,
        samples=X.shape[0],
        depth=0,
    )
    concrete_node_splitter.is_splittable(node)


def test__find_best_split(concrete_node_splitter, X, y):
    parent_mask = y.apply(lambda x: True)
    available_feature_names = X.columns.tolist()
    leaf_counter = 0
    concrete_node_splitter.find_best_split(
        parent_mask, available_feature_names, leaf_counter
    )
