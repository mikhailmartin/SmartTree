import pytest

from smarttree import SmartDecisionTreeClassifier


@pytest.fixture(scope="module")
def tree():
    return SmartDecisionTreeClassifier()


def test__fit(tree, X, y):

    tree.fit(X, y)

    assert tree.all_feature_names == X.columns.tolist()
    assert tree.classes_ == sorted(y.unique())


# def test__predict(tree, X):
#     _ = tree.predict(X)
