import pytest

from smarttree._node_splitter import NodeSplitter


@pytest.fixture(scope="module")
def node_splitter(
    X, y, num_features, cat_features, rank_features, feature_na_mode
) -> NodeSplitter:
    return NodeSplitter(
        X=X,
        y=y,
        criterion="gini",
        max_depth=float("+inf"),
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=.0,
        max_leaf_nodes=float("+inf"),
        max_childs=float("+inf"),
        num_features=num_features,
        cat_features=cat_features,
        rank_features=rank_features,
        feature_na_mode=feature_na_mode,
    )


def test__find_best_split(node_splitter, root_node):
    node_splitter.find_best_split_for(root_node, leaf_counter=0)


def test__is_splittable(node_splitter, root_node):
    node_splitter.find_best_split_for(root_node, leaf_counter=0)
    node_splitter.is_splittable(root_node, leaf_counter=0)
