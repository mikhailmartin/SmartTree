import re
from contextlib import nullcontext as does_not_raise
from copy import deepcopy

import pytest


DEFAULT_PARAMS_FROM_GET = {
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_leaf_nodes": None,
    "min_impurity_decrease": .0,
    "max_childs": None,
    "num_features": [],
    "cat_features": [],
    "rank_features": {},
    "hierarchy": {},
    "num_na_mode": "min",
    "cat_na_mode": "as_category",
    "cat_na_filler": "missing_value",
    "feature_na_mode": {},
}


def test__get_params(concrete_smart_tree):
    params = concrete_smart_tree.get_params()
    expected_params = DEFAULT_PARAMS_FROM_GET
    assert params == expected_params


@pytest.mark.parametrize(
    ("params_to_set", "expected_context"),
    [
        ({}, does_not_raise()),
        ({"criterion": "entropy"}, does_not_raise()),
        ({"max_depth": 4}, does_not_raise()),
        ({"min_samples_split": 4}, does_not_raise()),
        ({"min_samples_leaf": 2}, does_not_raise()),
        ({"max_leaf_nodes": 2}, does_not_raise()),
        ({"min_impurity_decrease": .01}, does_not_raise()),
        ({"max_childs": 2}, does_not_raise()),
        ({"num_features": ["num_feature"]}, does_not_raise()),
        ({"cat_features": ["cat_feature"]}, does_not_raise()),
        ({"rank_features": {"rank_feature": ["1", "2", "3"]}}, does_not_raise()),
        ({"hierarchy": {"num_feature": "rank_feature"}}, does_not_raise()),
        ({"num_na_mode": "max"}, does_not_raise()),
        ({"cat_na_mode": "as_category"}, does_not_raise()),
        ({"cat_na_filler": "NA"}, does_not_raise()),
        ({"feature_na_mode": {"feature": "max"}}, does_not_raise()),
        (
            {"aboba": "aboba"},
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Invalid parameter `aboba` for estimator ConcreteSmartTree."
                    " Valid parameters are: criterion, max_depth, min_samples_split,"
                    " min_samples_leaf, max_leaf_nodes, min_impurity_decrease,"
                    " max_childs, num_features, cat_features, rank_features, hierarchy,"
                    " num_na_mode, cat_na_mode, cat_na_filler, feature_na_mode."
                ),
            ),
        ),
    ],
    ids=[
        "void",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_leaf_nodes",
        "min_impurity_decrease",
        "max_childs",
        "num_features",
        "cat_features",
        "rank_features",
        "hierarchy",
        "num_na_mode",
        "cat_na_mode",
        "cat_na_filler",
        "feature_na_mode",
        "invalid",
    ],
)
def test__set_params(concrete_smart_tree, params_to_set, expected_context):
    with expected_context:
        concrete_smart_tree.set_params(**params_to_set)
        expected_params = deepcopy(DEFAULT_PARAMS_FROM_GET)
        expected_params.update(params_to_set)
        params = concrete_smart_tree.get_params()
        assert params == expected_params
