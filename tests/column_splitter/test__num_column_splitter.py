from smarttree._column_splitter import NumColumnSplitter


def test__split(dataset, root_node, feature_na_mode):

    numerical_column_splitter = NumColumnSplitter(
        dataset=dataset,
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        feature_na_mode=feature_na_mode,
    )

    split_feature_name_with_na = "2. Возраст"
    _ = numerical_column_splitter.split(root_node, split_feature_name_with_na)
