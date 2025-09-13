import pytest

from smarttree._column_splitter import CatColumnSplitter
from smarttree._dataset import Dataset
from smarttree._types import CatNaModeType


@pytest.mark.parametrize(
    "categorical_na_mode",
    ["as_category", "include_all", "include_best"],
    ids=lambda param: str(param),
)
def test__split(X, y, categorical_na_mode, root_node, feature_na_mode):

    categorical_na_mode: CatNaModeType
    categorical_column_splitter = CatColumnSplitter(
        dataset=Dataset(X, y),
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        max_childs=float("+inf"),
        max_leaf_nodes=float("+inf"),
        feature_na_mode=feature_na_mode,
    )

    split_feature_name_with_na = "25. Каким транспортом Вы обычно пользуетесь?"
    leaf_counter = 0
    _ = categorical_column_splitter.split(root_node, split_feature_name_with_na, leaf_counter)
