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


def test__split(numerical_column_splitter, y):

    parent_mask = y.apply(lambda x: True)
    split_feature_name = "2. Возраст"
    inf_gain, feature_values, child_masks = numerical_column_splitter.split(
        parent_mask, split_feature_name
    )
