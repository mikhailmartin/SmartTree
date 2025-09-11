import pytest

from smarttree._column_splitter import NumericalColumnSplitter
from smarttree._dataset import Dataset
from smarttree._types import NumericalNaModeType


@pytest.mark.parametrize(
    "numerical_na_mode",
    ["min", "max", "include_all", "include_best"],
    ids=lambda param: str(param),
)
def test__split(X, y, numerical_na_mode, root_node, feature_na_mode):

    numerical_na_mode: NumericalNaModeType
    numerical_column_splitter = NumericalColumnSplitter(
        dataset=Dataset(X, y),
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        feature_na_mode=feature_na_mode,
    )

    split_feature_name_with_na = "2. Возраст"
    _ = numerical_column_splitter.split(root_node, split_feature_name_with_na)
