from contextlib import nullcontext as does_not_raise

import pandas as pd
import pytest

from smarttree import BaseSmartDecisionTree


@pytest.fixture(scope="module")
def not_implemented_smart_tree() -> BaseSmartDecisionTree:
    class NotImplementedSmartTree(BaseSmartDecisionTree):
        ...
    return NotImplementedSmartTree()


def test__fit(not_implemented_smart_tree, X, y):
    with pytest.raises(NotImplementedError):
        not_implemented_smart_tree.fit(X, y)


def test__predict(not_implemented_smart_tree, X):
    with pytest.raises(NotImplementedError):
        not_implemented_smart_tree.predict(X)


def test__predict_proba(not_implemented_smart_tree, X):
    with pytest.raises(NotImplementedError):
        not_implemented_smart_tree.predict_proba(X)


def test__score(not_implemented_smart_tree, X, y):
    with pytest.raises(NotImplementedError):
        not_implemented_smart_tree.score(X, y)


def test__get_params(not_implemented_smart_tree):
    with pytest.raises(NotImplementedError):
        not_implemented_smart_tree.get_params()


def test__set_params(not_implemented_smart_tree):
    with pytest.raises(NotImplementedError):
        not_implemented_smart_tree.set_params()


def test__render(not_implemented_smart_tree):
    with pytest.raises(NotImplementedError):
        not_implemented_smart_tree.render()


@pytest.fixture(scope="module")
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

        def get_params(self, deep: bool = True):
            return "Implemented"

        def set_params(self):
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


def test__implemented(implemented_smart_tree, X, y):
    with does_not_raise():
        implemented_smart_tree.fit(X, y)
        implemented_smart_tree.predict(X)
        implemented_smart_tree.predict_proba(X)
        implemented_smart_tree.score(X, y)
        implemented_smart_tree.get_params()
        implemented_smart_tree.set_params()
        implemented_smart_tree.render()
