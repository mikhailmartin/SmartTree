import pandas as pd
import pytest

from smarttree._column_splitter import BaseColumnSplitter
from smarttree._dataset import Dataset


@pytest.fixture(scope="module")
def concrete_column_splitter(X, y, feature_na_mode) -> BaseColumnSplitter:
    class ConcreteColumnSplitter(BaseColumnSplitter):
        def split(
            self,
            parent_mask: pd.Series,
            split_feature_name: str,
        ) -> tuple[float, list[list[str]] | None, list[pd.Series] | None]:
            ...

    return ConcreteColumnSplitter(
        dataset=Dataset(X, y),
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        feature_na_mode=feature_na_mode,
    )


def test__gini_index(concrete_column_splitter, y):
    parent_mask = y.apply(lambda x: True)
    gini_index = concrete_column_splitter.gini_index(parent_mask)
    assert gini_index == 0.6666591342419322


def test__entropy(concrete_column_splitter, y):
    parent_mask = y.apply(lambda x: True)
    entropy = concrete_column_splitter.entropy(parent_mask)
    assert entropy == 1.584946181877191


def test__information_gain(concrete_column_splitter, y):

    parent_mask = y.apply(lambda x: True)

    n = int(parent_mask.shape[0] / 2)
    split_index = parent_mask.sample(n, random_state=42).index

    left_child_mask = parent_mask.copy()
    left_child_mask.loc[split_index] = False
    right_child_mask = ~left_child_mask

    child_masks = [left_child_mask, right_child_mask]
    inf_gain = concrete_column_splitter.information_gain(parent_mask, child_masks)

    assert inf_gain == 0.0016794443115909496
