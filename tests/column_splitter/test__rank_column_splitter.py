import pytest

from smarttree._column_splitter import RankColumnSplitter
from smarttree._dataset import Dataset


@pytest.fixture(scope="module")
def rank_column_splitter(X, y) -> RankColumnSplitter:
    return RankColumnSplitter(
        dataset=Dataset(X, y),
        criterion="gini",
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


def test__split(rank_column_splitter, root_node):

    split_feature_name = "5. В какой семье Вы выросли?"
    split_result = rank_column_splitter.split(root_node, split_feature_name)
