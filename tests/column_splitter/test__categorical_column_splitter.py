import pytest

from smarttree._column_splitter import CategoricalColumnSplitter
from smarttree._dataset import Dataset
from smarttree._types import CategoricalNaModeType


@pytest.mark.parametrize(
    "categorical_na_mode",
    ["as_category", "include_all", "include_best"],
    ids=lambda param: str(param),
)
def test__split(X, y, categorical_na_mode, root_node):

    categorical_na_mode: CategoricalNaModeType
    categorical_column_splitter = CategoricalColumnSplitter(
        dataset=Dataset(X, y),
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        categorical_na_mode=categorical_na_mode,
        max_childs=float("+inf"),
        max_leaf_nodes=float("+inf"),
    )

    split_feature_name_with_na = "25. Каким транспортом Вы обычно пользуетесь?"
    leaf_counter = 0
    split_result = categorical_column_splitter.split(
        root_node, split_feature_name_with_na, leaf_counter
    )
