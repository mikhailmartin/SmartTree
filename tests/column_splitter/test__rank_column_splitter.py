import pytest

from smarttree._column_splitter import RankColumnSplitter


@pytest.fixture(scope="module")
def rank_column_splitter(X, y) -> RankColumnSplitter:
    return RankColumnSplitter(
        X=X,
        y=y,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        rank_feature_names={
            "5. В какой семье Вы выросли?": [
                "полная семья, кровные родители",
                "мачеха/отчим",
                "мать/отец одиночка",
                "с бабушкой и дедушкой",
                "в детском доме",
            ],
        },
    )


def test__split(rank_column_splitter, y):

    parent_mask = y.apply(lambda x: True)
    split_feature_name = "5. В какой семье Вы выросли?"
    inf_gain, feature_values, child_masks = rank_column_splitter.split(
        parent_mask, split_feature_name
    )
