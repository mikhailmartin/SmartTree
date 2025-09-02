import pytest

from smarttree import SmartDecisionTreeClassifier
from smarttree._constants import (
    CategoricalNanModeOption,
    ClassificationCriterionOption,
    NumericalNanModeOption,
)


@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        ("gini", f"{SmartDecisionTreeClassifier.__name__}()"),
        ("entropy", f"{SmartDecisionTreeClassifier.__name__}(criterion='entropy')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__criterion(criterion, expected):
    criterion: ClassificationCriterionOption
    tree_classifier = SmartDecisionTreeClassifier(criterion=criterion)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("max_depth", "expected"),
    [
        (None, f"{SmartDecisionTreeClassifier.__name__}()"),
        (1, f"{SmartDecisionTreeClassifier.__name__}(max_depth=1)"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__max_depth(max_depth, expected):
    tree_classifier = SmartDecisionTreeClassifier(max_depth=max_depth)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("min_samples_split", "expected"),
    [
        (2, f"{SmartDecisionTreeClassifier.__name__}()"),
        (3, f"{SmartDecisionTreeClassifier.__name__}(min_samples_split=3)"),
        (.5, f"{SmartDecisionTreeClassifier.__name__}(min_samples_split=0.5)"),
    ],
    ids=["default value", "not default value(int)", "not default value(float)"],
)
def test_repr_tree__min_samples_split(min_samples_split, expected):
    tree_classifier = SmartDecisionTreeClassifier(min_samples_split=min_samples_split)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("min_samples_leaf", "expected"),
    [
        (1, f"{SmartDecisionTreeClassifier.__name__}()"),
        (2, f"{SmartDecisionTreeClassifier.__name__}(min_samples_leaf=2)"),
        (.5, f"{SmartDecisionTreeClassifier.__name__}(min_samples_leaf=0.5)"),
    ],
    ids=["default value", "not default value(int)", "not default value(float)"],
)
def test_repr_tree__min_samples_leaf(min_samples_leaf, expected):
    tree_classifier = SmartDecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("max_leaf_nodes", "expected"),
    [
        (None, f"{SmartDecisionTreeClassifier.__name__}()"),
        (2, f"{SmartDecisionTreeClassifier.__name__}(max_leaf_nodes=2)"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__max_leaf_nodes(max_leaf_nodes, expected):
    tree_classifier = SmartDecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("min_impurity_decrease", "expected"),
    [
        (.0, f"{SmartDecisionTreeClassifier.__name__}()"),
        (.5, f"{SmartDecisionTreeClassifier.__name__}(min_impurity_decrease=0.5)"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__min_impurity_decrease(min_impurity_decrease, expected):
    tree_classifier = SmartDecisionTreeClassifier(min_impurity_decrease=min_impurity_decrease)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("max_childs", "expected"),
    [
        (None, f"{SmartDecisionTreeClassifier.__name__}()"),
        (2, f"{SmartDecisionTreeClassifier.__name__}(max_childs=2)"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__max_childs(max_childs, expected):
    tree_classifier = SmartDecisionTreeClassifier(max_childs=max_childs)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("numerical_feature_names", "expected"),
    [
        (None, f"{SmartDecisionTreeClassifier.__name__}()"),
        ("feature", f"{SmartDecisionTreeClassifier.__name__}(numerical_feature_names=['feature'])"),
        (["feature"], f"{SmartDecisionTreeClassifier.__name__}(numerical_feature_names=['feature'])"),
    ],
    ids=["default value", "not default value(str)", "not default value(list[str])"],
)
def test_repr_tree__numerical_feature_names(numerical_feature_names, expected):
    tree_classifier = SmartDecisionTreeClassifier(
        numerical_feature_names=numerical_feature_names)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("categorical_feature_names", "expected"),
    [
        (None, f"{SmartDecisionTreeClassifier.__name__}()"),
        ("feature", f"{SmartDecisionTreeClassifier.__name__}(categorical_feature_names=['feature'])"),
        (["feature"], f"{SmartDecisionTreeClassifier.__name__}(categorical_feature_names=['feature'])"),
    ],
    ids=["default value", "not default value(str)", "not default value(list[str])"],
)
def test_repr_tree__categorical_feature_names(categorical_feature_names, expected):
    tree_classifier = SmartDecisionTreeClassifier(
        categorical_feature_names=categorical_feature_names)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("rank_feature_names", "expected"),
    [
        (None, f"{SmartDecisionTreeClassifier.__name__}()"),
        ({"f": ["v1", "v2"]}, f"{SmartDecisionTreeClassifier.__name__}(rank_feature_names={{'f': ['v1', 'v2']}})"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__rank_feature_names(rank_feature_names, expected):
    tree_classifier = SmartDecisionTreeClassifier(rank_feature_names=rank_feature_names)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("hierarchy", "expected"),
    [
        (None, f"{SmartDecisionTreeClassifier.__name__}()"),
        ({"f": "f1"}, f"{SmartDecisionTreeClassifier.__name__}(hierarchy={{'f': 'f1'}})"),
        ({"f": ["f1", "f2"]}, f"{SmartDecisionTreeClassifier.__name__}(hierarchy={{'f': ['f1', 'f2']}})"),
    ],
    ids=[
        "default value",
        "not default value(dict[str, str])",
        "not default value(dict[str, list[str]])",
    ],
)
def test_repr_tree__hierarchy(hierarchy, expected):
    tree_classifier = SmartDecisionTreeClassifier(hierarchy=hierarchy)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("numerical_nan_mode", "expected"),
    [
        ("min", f"{SmartDecisionTreeClassifier.__name__}()"),
        ("max", f"{SmartDecisionTreeClassifier.__name__}(numerical_nan_mode='max')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__numerical_nan_mode(numerical_nan_mode, expected):
    numerical_nan_mode: NumericalNanModeOption
    tree_classifier = SmartDecisionTreeClassifier(numerical_nan_mode=numerical_nan_mode)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("categorical_nan_mode", "expected"),
    [
        ("include", f"{SmartDecisionTreeClassifier.__name__}()"),
        ("as_category", f"{SmartDecisionTreeClassifier.__name__}(categorical_nan_mode='as_category')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__categorical_nan_mode(categorical_nan_mode, expected):
    categorical_nan_mode: CategoricalNanModeOption
    tree_classifier = SmartDecisionTreeClassifier(categorical_nan_mode=categorical_nan_mode)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("categorical_nan_filler", "expected"),
    [
        ("missing_value", f"{SmartDecisionTreeClassifier.__name__}()"),
        ("NULL", f"{SmartDecisionTreeClassifier.__name__}(categorical_nan_filler='NULL')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__categorical_nan_filler(categorical_nan_filler, expected):
    tree_classifier = SmartDecisionTreeClassifier(
        categorical_nan_filler=categorical_nan_filler)
    assert repr(tree_classifier) == expected
