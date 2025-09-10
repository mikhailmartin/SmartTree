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
    "numerical_features": [],
    "categorical_features": [],
    "rank_feature_names": {},
    "hierarchy": {},
    "numerical_na_mode": "min",
    "categorical_na_mode": "as_category",
    "categorical_na_filler": "missing_value",
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
        ({"numerical_features": ["num_feature"]}, does_not_raise()),
        ({"categorical_features": ["cat_feature"]}, does_not_raise()),
        ({"rank_feature_names": {"rank_feature": ["1", "2", "3"]}}, does_not_raise()),
        ({"hierarchy": {"num_feature": "rank_feature"}}, does_not_raise()),
        ({"numerical_na_mode": "max"}, does_not_raise()),
        ({"categorical_na_mode": "as_category"}, does_not_raise()),
        ({"categorical_na_filler": "NA"}, does_not_raise()),
        (
            {"aboba": "aboba"},
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Invalid parameter `aboba` for estimator ConcreteSmartTree."
                    " Valid parameters are: criterion, max_depth, min_samples_split,"
                    " min_samples_leaf, max_leaf_nodes, min_impurity_decrease,"
                    " max_childs, numerical_features, categorical_features,"
                    " rank_feature_names, hierarchy, numerical_na_mode,"
                    " categorical_na_mode, categorical_na_filler."
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
        "numerical_features",
        "categorical_features",
        "rank_feature_names",
        "hierarchy",
        "numerical_na_mode",
        "categorical_na_mode",
        "categorical_na_filler",
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
