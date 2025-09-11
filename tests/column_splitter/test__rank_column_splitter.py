from smarttree._column_splitter import RankColumnSplitter
from smarttree._dataset import Dataset


def test__split(X, y, root_node, feature_na_mode):

    rank_column_splitter = RankColumnSplitter(
        dataset=Dataset(X, y),
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        rank_features={
            "5. В какой семье Вы выросли?": [
                "полная семья, кровные родители",
                "мачеха/отчим",
                "мать/отец одиночка",
                "с бабушкой и дедушкой",
                "в детском доме",
            ],
        },
        feature_na_mode=feature_na_mode,
    )

    split_feature_name = "5. В какой семье Вы выросли?"
    _ = rank_column_splitter.split(root_node, split_feature_name)
