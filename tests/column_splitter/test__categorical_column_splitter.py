import pytest

from smarttree._column_splitter import CategoricalColumnSplitter


@pytest.fixture(scope="module")
def categorical_column_splitter(X, y) -> CategoricalColumnSplitter:
    return CategoricalColumnSplitter(
        X=X,
        y=y,
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        categorical_nan_mode="include",
        max_childs=float("+inf"),
        max_leaf_nodes=float("+inf"),
    )


def test__split(categorical_column_splitter, root_node):

    split_feature_name = "3. Семейное положение"
    leaf_counter = 0
    split_result = categorical_column_splitter.split(
        root_node, split_feature_name, leaf_counter
    )
