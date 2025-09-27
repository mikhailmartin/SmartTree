from smarttree._column_splitter import RankColumnSplitter


def test__split(dataset, root_node, feature_na_mode):

    rank_column_splitter = RankColumnSplitter(
        dataset=dataset,
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
