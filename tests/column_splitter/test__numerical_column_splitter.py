import pytest

from smarttree._column_splitter import NumericalColumnSplitter


@pytest.fixture(scope="module")
def numerical_column_splitter(X, y) -> NumericalColumnSplitter:
    return NumericalColumnSplitter(
        X=X,
        y=y,
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        numerical_nan_mode="min",
    )


def test__split(numerical_column_splitter, root_node):

    split_feature_name = "2. Возраст"
    split_result = numerical_column_splitter.split(root_node, split_feature_name)
