import numpy as np
import pytest

from smarttree import SmartDecisionTreeClassifier


NUM_FEATURE_WITHOUT_NA = "6. Жив ли хотя бы один из Ваших родителей (да/нет)?"
NUM_FEATURE_WITH_NA = "4. Если имеете супруга или партнера, как долго вы живете вместе (в годах)?"
CAT_FEATURE_WITHOUT_NA = "3. Семейное положение"
CAT_FEATURE_WITH_NA = "23. Каков тип Вашего дома?"
RANK_FEATURE_WITHOUT_NA = ...
RANK_FEATURE_WITH_NA = "5. В какой семье Вы выросли?"


@pytest.mark.parametrize(
    ("X_scenario", "na_mode"),
    [
        ("single_num_feature_without_na", dict()),
        ("single_num_feature_with_na", dict(numerical_na_mode="min")),
        ("single_num_feature_with_na", dict(numerical_na_mode="max")),
        ("single_cat_feature_without_na", dict()),
        ("single_cat_feature_with_na", dict(categorical_na_mode="as_category")),
        ("single_cat_feature_with_na", dict(categorical_na_mode="include_all")),
        ("single_rank_feature_with_na", dict()),
    ],
)
def test__fit_predict(X, y, X_scenario, na_mode):

    X_map = {
        "single_num_feature_without_na": X[[NUM_FEATURE_WITHOUT_NA]],
        "single_num_feature_with_na": X[[NUM_FEATURE_WITH_NA]],
        "single_cat_feature_without_na": X[[CAT_FEATURE_WITHOUT_NA]],
        "single_cat_feature_with_na": X[[CAT_FEATURE_WITH_NA]],
        "single_rank_feature_with_na": X[[RANK_FEATURE_WITH_NA]],
    }

    X_input = X_map[X_scenario]

    tree = SmartDecisionTreeClassifier(**na_mode)
    tree.fit(X_input, y)

    assert tree.all_features == X_input.columns.to_list()
    np.testing.assert_array_equal(tree.classes_, np.sort(y.unique()))

    _ = tree.predict(X_input)
    _ = tree.predict_log_proba(X_input)
