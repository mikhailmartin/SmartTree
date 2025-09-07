import pytest

from smarttree._column_splitter import NumericalColumnSplitter
from smarttree._dataset import Dataset
from smarttree._types import NumericalNanModeType


@pytest.fixture(scope="module")
def numerical_column_splitter(X, y) -> NumericalColumnSplitter:
    return NumericalColumnSplitter(
        dataset=Dataset(X, y),
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        numerical_nan_mode="min",
    )

@pytest.mark.parametrize(
    "numerical_nan_mode",
    ["min", "max", "include"],
    ids=lambda param: str(param),
)
def test__split(X, y, numerical_nan_mode, root_node):

    numerical_nan_mode: NumericalNanModeType
    numerical_column_splitter = NumericalColumnSplitter(
        dataset=Dataset(X, y),
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        numerical_nan_mode=numerical_nan_mode,
    )

    split_feature_name_with_nan = "2. Возраст"
    split_result = numerical_column_splitter.split(root_node, split_feature_name_with_nan)
