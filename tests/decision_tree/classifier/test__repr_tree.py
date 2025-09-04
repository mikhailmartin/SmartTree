import pytest

from smarttree import SmartDecisionTreeClassifier
from smarttree._constants import (
    CategoricalNanModeOption,
    ClassificationCriterionOption,
    NumericalNanModeOption,
)


CLASS_NAME = SmartDecisionTreeClassifier.__name__


@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        ("gini", f"{CLASS_NAME}()"),
        ("entropy", f"{CLASS_NAME}(criterion='entropy')"),
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
        (None, f"{CLASS_NAME}()"),
        (1, f"{CLASS_NAME}(max_depth=1)"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__max_depth(max_depth, expected):
    tree_classifier = SmartDecisionTreeClassifier(max_depth=max_depth)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("min_samples_split", "expected"),
    [
        (2, f"{CLASS_NAME}()"),
        (3, f"{CLASS_NAME}(min_samples_split=3)"),
        (.5, f"{CLASS_NAME}(min_samples_split=0.5)"),
    ],
    ids=["default value", "not default value(int)", "not default value(float)"],
)
def test_repr_tree__min_samples_split(min_samples_split, expected):
    tree_classifier = SmartDecisionTreeClassifier(min_samples_split=min_samples_split)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("min_samples_split", "min_samples_leaf", "expected"),
    [
        (2, 1, f"{CLASS_NAME}()"),
        (4, 2, f"{CLASS_NAME}(min_samples_split=4, min_samples_leaf=2)"),
        (2, .5, f"{CLASS_NAME}(min_samples_leaf=0.5)"),
    ],
    ids=["default value", "not default value(int)", "not default value(float)"],
)
def test_repr_tree__min_samples_leaf(min_samples_split, min_samples_leaf, expected):
    tree_classifier = SmartDecisionTreeClassifier(
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf
    )
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("max_leaf_nodes", "expected"),
    [
        (None, f"{CLASS_NAME}()"),
        (2, f"{CLASS_NAME}(max_leaf_nodes=2)"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__max_leaf_nodes(max_leaf_nodes, expected):
    tree_classifier = SmartDecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("min_impurity_decrease", "expected"),
    [
        (.0, f"{CLASS_NAME}()"),
        (.5, f"{CLASS_NAME}(min_impurity_decrease=0.5)"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__min_impurity_decrease(min_impurity_decrease, expected):
    tree_classifier = SmartDecisionTreeClassifier(
        min_impurity_decrease=min_impurity_decrease
    )
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("max_childs", "expected"),
    [
        (None, f"{CLASS_NAME}()"),
        (2, f"{CLASS_NAME}(max_childs=2)"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__max_childs(max_childs, expected):
    tree_classifier = SmartDecisionTreeClassifier(max_childs=max_childs)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("numerical_feature_names", "expected"),
    [
        (None, f"{CLASS_NAME}()"),
        ("feature", f"{CLASS_NAME}(numerical_feature_names=['feature'])"),
        (["feature"], f"{CLASS_NAME}(numerical_feature_names=['feature'])"),
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
        (None, f"{CLASS_NAME}()"),
        ("feature", f"{CLASS_NAME}(categorical_feature_names=['feature'])"),
        (["feature"], f"{CLASS_NAME}(categorical_feature_names=['feature'])"),
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
        (None, f"{CLASS_NAME}()"),
        (
            {"f": ["v1", "v2"]},
            f"{CLASS_NAME}(rank_feature_names={{'f': ['v1', 'v2']}})",
        ),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__rank_feature_names(rank_feature_names, expected):
    tree_classifier = SmartDecisionTreeClassifier(rank_feature_names=rank_feature_names)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("hierarchy", "expected"),
    [
        (None, f"{CLASS_NAME}()"),
        ({"f": "f1"}, f"{CLASS_NAME}(hierarchy={{'f': 'f1'}})"),
        ({"f": ["f1", "f2"]}, f"{CLASS_NAME}(hierarchy={{'f': ['f1', 'f2']}})"),
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
        ("min", f"{CLASS_NAME}()"),
        ("max", f"{CLASS_NAME}(numerical_nan_mode='max')"),
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
        ("include", f"{CLASS_NAME}()"),
        ("as_category", f"{CLASS_NAME}(categorical_nan_mode='as_category')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__categorical_nan_mode(categorical_nan_mode, expected):
    categorical_nan_mode: CategoricalNanModeOption
    tree_classifier = SmartDecisionTreeClassifier(
        categorical_nan_mode=categorical_nan_mode
    )
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("categorical_nan_filler", "expected"),
    [
        ("missing_value", f"{CLASS_NAME}()"),
        ("NULL", f"{CLASS_NAME}(categorical_nan_filler='NULL')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__categorical_nan_filler(categorical_nan_filler, expected):
    tree_classifier = SmartDecisionTreeClassifier(
        categorical_nan_filler=categorical_nan_filler)
    assert repr(tree_classifier) == expected
