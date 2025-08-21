import pytest
from pytest import param

from smarttree import SmartTreeClassifier


@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        param("gini", f"{SmartTreeClassifier.__name__}()"),
        param("entropy", f"{SmartTreeClassifier.__name__}(criterion='entropy')"),
    ],
)
def test_repr_tree__criterion(criterion, expected):
    tree_classifier = SmartTreeClassifier(criterion=criterion)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("max_depth", "expected"),
    [
        param(None, f"{SmartTreeClassifier.__name__}()"),
        param(1, f"{SmartTreeClassifier.__name__}(max_depth=1)"),
    ],
)
def test_repr_tree__max_depth(max_depth, expected):
    tree_classifier = SmartTreeClassifier(max_depth=max_depth)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("min_samples_split", "expected"),
    [
        param(2, f"{SmartTreeClassifier.__name__}()"),
        param(3, f"{SmartTreeClassifier.__name__}(min_samples_split=3)"),
        param(.5, f"{SmartTreeClassifier.__name__}(min_samples_split=0.5)"),
    ],
)
def test_repr_tree__min_samples_split(min_samples_split, expected):
    tree_classifier = SmartTreeClassifier(min_samples_split=min_samples_split)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("min_samples_leaf", "expected"),
    [
        param(1, f"{SmartTreeClassifier.__name__}()"),
        param(2, f"{SmartTreeClassifier.__name__}(min_samples_leaf=2)"),
        param(.5, f"{SmartTreeClassifier.__name__}(min_samples_leaf=0.5)"),
    ],
)
def test_repr_tree__min_samples_leaf(min_samples_leaf, expected):
    tree_classifier = SmartTreeClassifier(min_samples_leaf=min_samples_leaf)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("max_leaf_nodes", "expected"),
    [
        param(float("+inf"), f"{SmartTreeClassifier.__name__}()"),
        param(2, f"{SmartTreeClassifier.__name__}(max_leaf_nodes=2)"),
    ],
)
def test_repr_tree__max_leaf_nodes(max_leaf_nodes, expected):
    tree_classifier = SmartTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("min_impurity_decrease", "expected"),
    [
        param(.0, f"{SmartTreeClassifier.__name__}()"),
        param(.5, f"{SmartTreeClassifier.__name__}(min_impurity_decrease=0.5)"),
    ],
)
def test_repr_tree__min_impurity_decrease(min_impurity_decrease, expected):
    tree_classifier = SmartTreeClassifier(min_impurity_decrease=min_impurity_decrease)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("max_childs", "expected"),
    [
        param(float("+inf"), f"{SmartTreeClassifier.__name__}()"),
        param(2, f"{SmartTreeClassifier.__name__}(max_childs=2)"),
    ],
)
def test_repr_tree__max_childs(max_childs, expected):
    tree_classifier = SmartTreeClassifier(max_childs=max_childs)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("numerical_feature_names", "expected"),
    [
        param(None, f"{SmartTreeClassifier.__name__}()"),
        param("feature", f"{SmartTreeClassifier.__name__}(numerical_feature_names=['feature'])"),
        param(["feature"], f"{SmartTreeClassifier.__name__}(numerical_feature_names=['feature'])"),
    ],
)
def test_repr_tree__numerical_feature_names(numerical_feature_names, expected):
    tree_classifier = SmartTreeClassifier(
        numerical_feature_names=numerical_feature_names)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("categorical_feature_names", "expected"),
    [
        param(None, f"{SmartTreeClassifier.__name__}()"),
        param("feature", f"{SmartTreeClassifier.__name__}(categorical_feature_names=['feature'])"),
        param(["feature"], f"{SmartTreeClassifier.__name__}(categorical_feature_names=['feature'])"),
    ],
)
def test_repr_tree__categorical_feature_names(categorical_feature_names, expected):
    tree_classifier = SmartTreeClassifier(
        categorical_feature_names=categorical_feature_names)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("rank_feature_names", "expected"),
    [
        param(None, f"{SmartTreeClassifier.__name__}()"),
        param({"f": ["v1", "v2"]}, f"{SmartTreeClassifier.__name__}(rank_feature_names={{'f': ['v1', 'v2']}})"),
    ],
)
def test_repr_tree__rank_feature_names(rank_feature_names, expected):
    tree_classifier = SmartTreeClassifier(rank_feature_names=rank_feature_names)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("hierarchy", "expected"),
    [
        param(None, f"{SmartTreeClassifier.__name__}()"),
        param({"f": "f1"}, f"{SmartTreeClassifier.__name__}(hierarchy={{'f': 'f1'}})"),
        param({"f": ["f1", "f2"]}, f"{SmartTreeClassifier.__name__}(hierarchy={{'f': ['f1', 'f2']}})"),
    ],
)
def test_repr_tree__hierarchy(hierarchy, expected):
    tree_classifier = SmartTreeClassifier(hierarchy=hierarchy)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("numerical_nan_mode", "expected"),
    [
        param("min", f"{SmartTreeClassifier.__name__}()"),
        param("max", f"{SmartTreeClassifier.__name__}(numerical_nan_mode='max')"),
        param("include", f"{SmartTreeClassifier.__name__}(numerical_nan_mode='include')"),
    ],
)
def test_repr_tree__numerical_nan_mode(numerical_nan_mode, expected):
    tree_classifier = SmartTreeClassifier(numerical_nan_mode=numerical_nan_mode)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("categorical_nan_mode", "expected"),
    [
        param("include", f"{SmartTreeClassifier.__name__}()"),
        param("as_category", f"{SmartTreeClassifier.__name__}(categorical_nan_mode='as_category')"),
    ],
)
def test_repr_tree__categorical_nan_mode(categorical_nan_mode, expected):
    tree_classifier = SmartTreeClassifier(categorical_nan_mode=categorical_nan_mode)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("categorical_nan_filler", "expected"),
    [
        param("missing_value", f"{SmartTreeClassifier.__name__}()"),
        param("NULL", f"{SmartTreeClassifier.__name__}(categorical_nan_filler='NULL')"),
    ],
)
def test_repr_tree__categorical_nan_filler(categorical_nan_filler, expected):
    tree_classifier = SmartTreeClassifier(
        categorical_nan_filler=categorical_nan_filler)
    assert repr(tree_classifier) == expected
