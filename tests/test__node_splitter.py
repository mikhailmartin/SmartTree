import pytest

from smarttree._node_splitter import NodeSplitter


@pytest.fixture(scope="module")
def concrete_node_splitter(
    X, y, numerical_features, categorical_features, rank_features, feature_na_mode
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
        num_features=numerical_features,
        cat_features=categorical_features,
        rank_features=rank_features,
        feature_na_mode=feature_na_mode,
    )


def test__find_best_split(concrete_node_splitter, root_node):
    concrete_node_splitter.find_best_split_for(root_node)


def test__is_splittable(concrete_node_splitter, root_node):
    concrete_node_splitter.find_best_split_for(root_node)
    concrete_node_splitter.is_splittable(root_node)
