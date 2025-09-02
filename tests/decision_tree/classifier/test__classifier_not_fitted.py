import pytest

from smarttree import SmartDecisionTreeClassifier
from smarttree._exceptions import NotFittedError


@pytest.fixture(scope="module")
def not_fitted_tree() -> SmartDecisionTreeClassifier:
    return SmartDecisionTreeClassifier()


@pytest.mark.parametrize(
    "method_call",
    [
        lambda tree, X, y: tree.predict(X),
        lambda tree, X, y: tree.predict_proba(X),
        lambda tree, X, y: tree.score(X, y),
        lambda tree, X, y: tree.render(),
    ],
    ids=["predict", "predict_proba", "score", "render"],
)
def test__not_fitted__method(not_fitted_tree, X, y, method_call):
    with pytest.raises(NotFittedError):
        method_call(not_fitted_tree, X, y)


@pytest.mark.parametrize(
    "property_name",
    ["classes_"],
    ids=lambda param: str(param),
)
def test__not_fitted__property(not_fitted_tree, property_name):
    with pytest.raises(NotFittedError):
        property_ = getattr(not_fitted_tree, property_name)
        _ = property_
