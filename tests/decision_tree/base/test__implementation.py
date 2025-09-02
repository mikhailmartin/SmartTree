from contextlib import nullcontext as does_not_raise

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
    "ids": ["fit", "predict", "predict_proba", "score", "render"],
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
