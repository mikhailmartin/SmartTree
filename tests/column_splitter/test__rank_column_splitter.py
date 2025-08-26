import pytest

from smarttree._column_splitter import RankColumnSplitter


@pytest.fixture(scope="module")
def rank_column_splitter(X, y):
    return RankColumnSplitter(
        X=X,
        y=y,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    )


def test__split(rank_column_splitter, y):

    parent_mask = y.apply(lambda x: True)
    split_feature_name = "3. Семейное положение"
    inf_gain, feature_values, child_masks = rank_column_splitter.split(
        parent_mask, split_feature_name
    )
