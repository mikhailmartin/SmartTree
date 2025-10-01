import pytest

from ...conftest import ConcreteSmartTree
from smarttree._types import (
    CatNaModeType,
    ClassificationCriterionType,
    CommonNaModeType,
    NumNaModeType,
)


CLASS_NAME = ConcreteSmartTree.__name__


@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        ("gini", f"{CLASS_NAME}()"),
        ("entropy", f"{CLASS_NAME}(criterion='entropy')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__criterion(criterion, expected):
    criterion: ClassificationCriterionType
    tree_classifier = ConcreteSmartTree(criterion=criterion)
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
    tree_classifier = ConcreteSmartTree(max_depth=max_depth)
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
    tree_classifier = ConcreteSmartTree(min_samples_split=min_samples_split)
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
    tree_classifier = ConcreteSmartTree(
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
    tree_classifier = ConcreteSmartTree(max_leaf_nodes=max_leaf_nodes)
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
    tree_classifier = ConcreteSmartTree(min_impurity_decrease=min_impurity_decrease)
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
    tree_classifier = ConcreteSmartTree(max_childs=max_childs)
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
    tree_classifier = ConcreteSmartTree(hierarchy=hierarchy)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("na_mode", "expected"),
    [
        ("include_best", f"{CLASS_NAME}()"),
        ("include_all", f"{CLASS_NAME}(na_mode='include_all')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__na_mode(na_mode, expected):
    na_mode: CommonNaModeType
    tree_classifier = ConcreteSmartTree(na_mode=na_mode)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("num_na_mode", "expected"),
    [
        (None, f"{CLASS_NAME}()"),
        ("max", f"{CLASS_NAME}(num_na_mode='max')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__num_na_mode(num_na_mode, expected):
    num_na_mode: NumNaModeType
    tree_classifier = ConcreteSmartTree(num_na_mode=num_na_mode)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("cat_na_mode", "expected"),
    [
        (None, f"{CLASS_NAME}()"),
        ("include_all", f"{CLASS_NAME}(cat_na_mode='include_all')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__cat_na_mode(cat_na_mode, expected):
    cat_na_mode: CatNaModeType
    tree_classifier = ConcreteSmartTree(cat_na_mode=cat_na_mode)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("cat_na_filler", "expected"),
    [
        ("missing_value", f"{CLASS_NAME}()"),
        ("NULL", f"{CLASS_NAME}(cat_na_filler='NULL')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree__cat_na_filler(cat_na_filler, expected):
    tree_classifier = ConcreteSmartTree(cat_na_filler=cat_na_filler)
    assert repr(tree_classifier) == expected


@pytest.mark.parametrize(
    ("rank_na_mode", "expected"),
    [
        (None, f"{CLASS_NAME}()"),
        ("include_all", f"{CLASS_NAME}(rank_na_mode='include_all')"),
    ],
    ids=["default value", "not default value"],
)
def test_repr_tree_rank_na_mode(rank_na_mode, expected):
    tree_classifier = ConcreteSmartTree(rank_na_mode=rank_na_mode)
    assert repr(tree_classifier) == expected
