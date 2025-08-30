import pytest
from pytest import raises

from smarttree import BaseSmartDecisionTree, SmartDecisionTreeClassifier
from smarttree._exceptions import NotFittedError


@pytest.fixture(scope="module")
def not_fitted_tree() -> BaseSmartDecisionTree:
    return SmartDecisionTreeClassifier()


def test__tree(not_fitted_tree):
    with raises(NotFittedError):
        _ = not_fitted_tree.tree


def test__feature_names(not_fitted_tree):
    with raises(NotFittedError):
        _ = not_fitted_tree.feature_names


def test__feature_importances(not_fitted_tree):
    with raises(NotFittedError):
        _ = not_fitted_tree.feature_importances_


def test__predict(not_fitted_tree):
    with raises(NotFittedError):
        _ = not_fitted_tree.predict(None)


def test__predict_proba(not_fitted_tree, X):
    with raises(NotFittedError):
        _ = not_fitted_tree.predict_proba(X)


def test__score(not_fitted_tree, X, y):
    with raises(NotFittedError):
        _ = not_fitted_tree.score(X, y)


def test__render(not_fitted_tree):
    with raises(NotFittedError):
        _ = not_fitted_tree.render()


@pytest.fixture(scope="module")
def not_fitted_tree_classifier() -> SmartDecisionTreeClassifier:
    return SmartDecisionTreeClassifier()


def test__classes_(not_fitted_tree_classifier):
    with raises(NotFittedError):
        _ = not_fitted_tree_classifier.classes_
