from pytest import raises

from smarttree import SmartDecisionTreeClassifier
from smarttree._exceptions import NotFittedError


NOT_FITTED_MSDT = SmartDecisionTreeClassifier()


def test__tree():
    with raises(NotFittedError):
        _ = NOT_FITTED_MSDT.tree


def test__class_names():
    with raises(NotFittedError):
        _ = NOT_FITTED_MSDT.class_names


def test__feature_names():
    with raises(NotFittedError):
        _ = NOT_FITTED_MSDT.feature_names


def test__feature_importances():
    with raises(NotFittedError):
        _ = NOT_FITTED_MSDT.feature_importances_


def test__predict():
    with raises(NotFittedError):
        _ = NOT_FITTED_MSDT.predict(None)


def test__predict_proba():
    with raises(NotFittedError):
        _ = NOT_FITTED_MSDT.predict_proba(None)


def test__score():
    with raises(NotFittedError):
        _ = NOT_FITTED_MSDT.score(None, None)


def test__render():
    with raises(NotFittedError):
        _ = NOT_FITTED_MSDT.render()
