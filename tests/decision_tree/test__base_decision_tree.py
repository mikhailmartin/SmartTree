import re
from contextlib import nullcontext as does_not_raise
from copy import deepcopy

import pandas as pd
import pytest

from smarttree import BaseSmartDecisionTree

IMPLEMENTATION_PARAMETRIZATION = {
    "argnames": "method_call",
    "argvalues": [
        lambda tree, X, y: tree.fit(X, y),
        lambda tree, X, y: tree.predict(X),
        lambda tree, X, y: tree.predict_proba(X),
        lambda tree, X, y: tree.score(X, y),
        lambda tree, X, y: tree.render(),
    ],
}

DEFAULT_PARAMS_FROM_GET = {
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_leaf_nodes": None,
    "min_impurity_decrease": .0,
    "max_childs": None,
    "numerical_feature_names": [],
    "categorical_feature_names": [],
    "rank_feature_names": {},
    "hierarchy": {},
    "numerical_nan_mode": "min",
    "categorical_nan_mode": "as_category",
    "categorical_nan_filler": "missing_value",
}


@pytest.fixture(scope="function")
def not_implemented_smart_tree() -> BaseSmartDecisionTree:
    class NotImplementedSmartTree(BaseSmartDecisionTree):
        ...
    return NotImplementedSmartTree()


@pytest.mark.parametrize(**IMPLEMENTATION_PARAMETRIZATION)
def test__not_implemented(not_implemented_smart_tree, X, y, method_call):
    with pytest.raises(NotImplementedError):
        method_call(not_implemented_smart_tree, X, y)


@pytest.fixture(scope="function")
def implemented_smart_tree() -> BaseSmartDecisionTree:
    class ImplementedSmartTree(BaseSmartDecisionTree):
        def fit(self, X, y):
            return "Implemented"

        def predict(self, X):
            return "Implemented"

        def predict_proba(self, X):
            return "Implemented"

        def score(self, X, y, sample_weight: pd.Series | None = None):
            return "Implemented"

        def render(
            self,
            *,
            rounded: bool = False,
            show_impurity: bool = False,
            show_num_samples: bool = False,
            show_distribution: bool = False,
            show_label: bool = False,
            **kwargs,
        ):
            return "Implemented"

    return ImplementedSmartTree()


@pytest.mark.parametrize(**IMPLEMENTATION_PARAMETRIZATION)
def test__implemented(implemented_smart_tree, X, y, method_call):
    with does_not_raise():
        method_call(implemented_smart_tree, X, y)


def test__get_params(implemented_smart_tree):
    params = implemented_smart_tree.get_params()
    expected_params = DEFAULT_PARAMS_FROM_GET
    assert params == expected_params


@pytest.mark.parametrize(
    "params_to_set, expected_context",
    [
        ({}, does_not_raise()),
        ({"criterion": "entropy"}, does_not_raise()),
        ({"max_depth": 4}, does_not_raise()),
        ({"min_samples_split": 4}, does_not_raise()),
        ({"min_samples_leaf": 2}, does_not_raise()),
        ({"max_leaf_nodes": 2}, does_not_raise()),
        ({"min_impurity_decrease": .01}, does_not_raise()),
        ({"max_childs": 2}, does_not_raise()),
        ({"numerical_feature_names": ["num_feature"]}, does_not_raise()),
        ({"categorical_feature_names": ["cat_feature"]}, does_not_raise()),
        ({"rank_feature_names": {"rank_feature": ["1", "2", "3"]}}, does_not_raise()),
        ({"hierarchy": {"num_feature": "rank_feature"}}, does_not_raise()),
        ({"numerical_nan_mode": "max"}, does_not_raise()),
        ({"categorical_nan_mode": "include"}, does_not_raise()),
        ({"categorical_nan_filler": "NaN"}, does_not_raise()),
        (
            {"aboba": "aboba"},
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Invalid parameter `aboba` for estimator ImplementedSmartTree."
                    " Valid parameters are: criterion, max_depth, min_samples_split,"
                    " min_samples_leaf, max_leaf_nodes, min_impurity_decrease,"
                    " max_childs, numerical_feature_names, categorical_feature_names,"
                    " rank_feature_names, hierarchy, numerical_nan_mode,"
                    " categorical_nan_mode, categorical_nan_filler."
                ),
            ),
        ),
    ],
)
def test__set_params(implemented_smart_tree, params_to_set, expected_context):
    with expected_context:
        implemented_smart_tree.set_params(**params_to_set)
        expected_params = deepcopy(DEFAULT_PARAMS_FROM_GET)
        expected_params.update(params_to_set)
        params = implemented_smart_tree.get_params()
        assert params == expected_params
