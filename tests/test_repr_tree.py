import pytest
from pytest import param

from smarttree import MultiSplitDecisionTreeClassifier


@pytest.mark.parametrize(
    ('criterion', 'expected'),
    [
        param('gini', 'MultiSplitDecisionTreeClassifier()'),
        param('entropy', "MultiSplitDecisionTreeClassifier(criterion='entropy')"),
    ],
)
def test_repr_tree__criterion(criterion, expected):
    msdt = MultiSplitDecisionTreeClassifier(criterion=criterion)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('max_depth', 'expected'),
    [
        param(None, 'MultiSplitDecisionTreeClassifier()'),
        param(1, 'MultiSplitDecisionTreeClassifier(max_depth=1)'),
    ],
)
def test_repr_tree__max_depth(max_depth, expected):
    msdt = MultiSplitDecisionTreeClassifier(max_depth=max_depth)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('min_samples_split', 'expected'),
    [
        param(2, 'MultiSplitDecisionTreeClassifier()'),
        param(3, 'MultiSplitDecisionTreeClassifier(min_samples_split=3)'),
        param(.5, 'MultiSplitDecisionTreeClassifier(min_samples_split=0.5)'),
    ],
)
def test_repr_tree__min_samples_split(min_samples_split, expected):
    msdt = MultiSplitDecisionTreeClassifier(min_samples_split=min_samples_split)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('min_samples_leaf', 'expected'),
    [
        param(1, 'MultiSplitDecisionTreeClassifier()'),
        param(2, 'MultiSplitDecisionTreeClassifier(min_samples_leaf=2)'),
        param(.5, 'MultiSplitDecisionTreeClassifier(min_samples_leaf=0.5)'),
    ],
)
def test_repr_tree__min_samples_leaf(min_samples_leaf, expected):
    msdt = MultiSplitDecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('max_leaf_nodes', 'expected'),
    [
        param(float('+inf'), 'MultiSplitDecisionTreeClassifier()'),
        param(2, 'MultiSplitDecisionTreeClassifier(max_leaf_nodes=2)'),
    ],
)
def test_repr_tree__max_leaf_nodes(max_leaf_nodes, expected):
    msdt = MultiSplitDecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('min_impurity_decrease', 'expected'),
    [
        param(.0, 'MultiSplitDecisionTreeClassifier()'),
        param(.5, 'MultiSplitDecisionTreeClassifier(min_impurity_decrease=0.5)'),
    ],
)
def test_repr_tree__min_impurity_decrease(min_impurity_decrease, expected):
    msdt = MultiSplitDecisionTreeClassifier(min_impurity_decrease=min_impurity_decrease)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('max_childs', 'expected'),
    [
        param(float('+inf'), 'MultiSplitDecisionTreeClassifier()'),
        param(2, 'MultiSplitDecisionTreeClassifier(max_childs=2)'),
    ],
)
def test_repr_tree__max_childs(max_childs, expected):
    msdt = MultiSplitDecisionTreeClassifier(max_childs=max_childs)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('numerical_feature_names', 'expected'),
    [
        param(None, 'MultiSplitDecisionTreeClassifier()'),
        param('feature', "MultiSplitDecisionTreeClassifier(numerical_feature_names=['feature'])"),
        param(['feature'], "MultiSplitDecisionTreeClassifier(numerical_feature_names=['feature'])"),
    ],
)
def test_repr_tree__numerical_feature_names(numerical_feature_names, expected):
    msdt = MultiSplitDecisionTreeClassifier(
        numerical_feature_names=numerical_feature_names)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('categorical_feature_names', 'expected'),
    [
        param(None, 'MultiSplitDecisionTreeClassifier()'),
        param('feature', "MultiSplitDecisionTreeClassifier(categorical_feature_names=['feature'])"),
        param(['feature'], "MultiSplitDecisionTreeClassifier(categorical_feature_names=['feature'])"),
    ],
)
def test_repr_tree__categorical_feature_names(categorical_feature_names, expected):
    msdt = MultiSplitDecisionTreeClassifier(
        categorical_feature_names=categorical_feature_names)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('rank_feature_names', 'expected'),
    [
        param(None, 'MultiSplitDecisionTreeClassifier()'),
        param({'f': ['v1', 'v2']}, "MultiSplitDecisionTreeClassifier(rank_feature_names={'f': ['v1', 'v2']})"),
    ],
)
def test_repr_tree__rank_feature_names(rank_feature_names, expected):
    msdt = MultiSplitDecisionTreeClassifier(rank_feature_names=rank_feature_names)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('hierarchy', 'expected'),
    [
        param(None, 'MultiSplitDecisionTreeClassifier()'),
        param({'f': 'f1'}, "MultiSplitDecisionTreeClassifier(hierarchy={'f': 'f1'})"),
        param({'f': ['f1', 'f2']}, "MultiSplitDecisionTreeClassifier(hierarchy={'f': ['f1', 'f2']})"),
    ],
)
def test_repr_tree__hierarchy(hierarchy, expected):
    msdt = MultiSplitDecisionTreeClassifier(hierarchy=hierarchy)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('numerical_nan_mode', 'expected'),
    [
        param('min', 'MultiSplitDecisionTreeClassifier()'),
        param('max', "MultiSplitDecisionTreeClassifier(numerical_nan_mode='max')"),
        param('include', "MultiSplitDecisionTreeClassifier(numerical_nan_mode='include')"),
    ],
)
def test_repr_tree__numerical_nan_mode(numerical_nan_mode, expected):
    msdt = MultiSplitDecisionTreeClassifier(numerical_nan_mode=numerical_nan_mode)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('categorical_nan_mode', 'expected'),
    [
        param('include', 'MultiSplitDecisionTreeClassifier()'),
        param('as_category', "MultiSplitDecisionTreeClassifier(categorical_nan_mode='as_category')"),
    ],
)
def test_repr_tree__categorical_nan_mode(categorical_nan_mode, expected):
    msdt = MultiSplitDecisionTreeClassifier(categorical_nan_mode=categorical_nan_mode)
    assert repr(msdt) == expected


@pytest.mark.parametrize(
    ('categorical_nan_filler', 'expected'),
    [
        param('missing_value', 'MultiSplitDecisionTreeClassifier()'),
        param('NULL', "MultiSplitDecisionTreeClassifier(categorical_nan_filler='NULL')"),
    ],
)
def test_repr_tree__categorical_nan_filler(categorical_nan_filler, expected):
    msdt = MultiSplitDecisionTreeClassifier(
        categorical_nan_filler=categorical_nan_filler)
    assert repr(msdt) == expected
