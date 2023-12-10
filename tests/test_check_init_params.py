import sys
sys.path.append(sys.path[0] + '/../')
from contextlib import nullcontext as does_not_raise
import re

from multi_split_decision_tree._checkers import _check_init_params

import pytest
from pytest import param, raises


@pytest.mark.parametrize(
    ('criterion', 'expected'),
    [
        param('gini', does_not_raise()),
        param('entropy', does_not_raise()),
        param('log_loss', does_not_raise()),
        param(
            'gjni',
            raises(
                ValueError,
                match=(
                    'Для `criterion` доступны значения "entropy", "gini" и "log_loss".'
                    ' Текущее значение `criterion` = gjni.'
                ),
            ),
            id='invalid-criterion',
        ),
    ],
)
def test_init_param__criterion(criterion, expected):
    with expected:
        _check_init_params(criterion=criterion)


@pytest.mark.parametrize(
    ('max_depth', 'expected'),
    [
        param(None, does_not_raise()),
        param(2, does_not_raise()),
        param(
            -1,
            raises(
                ValueError,
                match=(
                    '`max_depth` должен представлять собой int и быть строго больше 0'
                    ' Текущее значение `max_depth` = -1.'
                ),
            ),
        ),
        param(
            1.5,
            raises(
                ValueError,
                match=(
                    '`max_depth` должен представлять собой int и быть строго больше 0'
                    ' Текущее значение `max_depth` = 1.5.'
                ),
            ),
        ),
        param(
            'string',
            raises(
                ValueError,
                match=(
                    '`max_depth` должен представлять собой int и быть строго больше 0'
                    ' Текущее значение `max_depth` = string.'
                ),
            ),
        ),
    ],
)
def test_init_param__max_depth(max_depth, expected):
    with expected:
        _check_init_params(max_depth=max_depth)


@pytest.mark.parametrize(
    ('min_samples_split', 'expected'),
    [
        param(2, does_not_raise()),
        param(
            1,
            raises(
                ValueError,
                match=re.escape(
                    '`min_samples_split` должен представлять собой либо int и лежать в'
                    ' диапазоне [2, +inf), либо float и лежать в диапазоне (0, 1).'
                    ' Текущее значение `min_samples_split` = 1.'
                ),
            ),
        ),
        param(.5, does_not_raise()),
        param(
            0.0,
            raises(
                ValueError,
                match=re.escape(
                    '`min_samples_split` должен представлять собой либо int и лежать в'
                    ' диапазоне [2, +inf), либо float и лежать в диапазоне (0, 1).'
                    ' Текущее значение `min_samples_split` = 0.0.'
                ),
            ),
        ),
        param(
            1.0,
            raises(
                ValueError,
                match=re.escape(
                    '`min_samples_split` должен представлять собой либо int и лежать в'
                    ' диапазоне [2, +inf), либо float и лежать в диапазоне (0, 1).'
                    ' Текущее значение `min_samples_split` = 1.0.'
                ),
            ),
        ),
        param(
            'string',
            raises(
                ValueError,
                match=re.escape(
                    '`min_samples_split` должен представлять собой либо int и лежать в'
                    ' диапазоне [2, +inf), либо float и лежать в диапазоне (0, 1).'
                    ' Текущее значение `min_samples_split` = string.'
                ),
            ),
        ),
    ],
)
def test_init_param__min_samples_split(min_samples_split, expected):
    with expected:
        _check_init_params(min_samples_split=min_samples_split)


@pytest.mark.parametrize(
    ('min_samples_leaf', 'expected'),
    [
        param(1, does_not_raise()),
        param(
            0,
            raises(
                ValueError,
                match=re.escape(
                    '`min_samples_leaf` должен представлять собой либо int и лежать в'
                    ' диапазоне [1, +inf), либо float и лежать в диапазоне (0, 1).'
                    ' Текущее значение `min_samples_leaf` = 0.'
                ),
            ),
        ),
        param(.5, does_not_raise()),
        param(
            0.0,
            raises(
                ValueError,
                match=re.escape(
                    '`min_samples_leaf` должен представлять собой либо int и лежать в'
                    ' диапазоне [1, +inf), либо float и лежать в диапазоне (0, 1).'
                    ' Текущее значение `min_samples_leaf` = 0.0.'
                ),
            ),
        ),
        param(
            1.0,
            raises(
                ValueError,
                match=re.escape(
                    '`min_samples_leaf` должен представлять собой либо int и лежать в'
                    ' диапазоне [1, +inf), либо float и лежать в диапазоне (0, 1).'
                    ' Текущее значение `min_samples_leaf` = 1.0.'
                ),
            ),
        ),
        param(
            'string',
            raises(
                ValueError,
                match=re.escape(
                    '`min_samples_leaf` должен представлять собой либо int и лежать в'
                    ' диапазоне [1, +inf), либо float и лежать в диапазоне (0, 1).'
                    ' Текущее значение `min_samples_leaf` = string.'
                ),
            ),
        ),
    ],
)
def test_init_params__min_samples_leaf(min_samples_leaf, expected):
    with expected:
        _check_init_params(min_samples_leaf=min_samples_leaf)


@pytest.mark.parametrize(
    ('max_leaf_nodes', 'expected'),
    [
        param(float('+inf'), does_not_raise()),
        param(2, does_not_raise()),
        param(
            1,
            raises(
                ValueError,
                match=re.escape(
                    '`max_leaf_nodes` должен представлять собой int и быть строго больше 2.'
                    ' Текущее значение `max_leaf_nodes` = 1.'
                ),
            ),
        ),
        param(
            'string',
            raises(
                ValueError,
                match=re.escape(
                    '`max_leaf_nodes` должен представлять собой int и быть строго больше 2.'
                    ' Текущее значение `max_leaf_nodes` = string.'
                ),
            ),
        ),
    ],
)
def test_init_params__max_leaf_nodes(max_leaf_nodes, expected):
    with expected:
        _check_init_params(max_leaf_nodes=max_leaf_nodes)


@pytest.mark.parametrize(
    ('min_impurity_decrease', 'expected'),
    [
        param(.0, does_not_raise()),
        param(
            -1.,
            raises(
                ValueError,
                match=re.escape(
                    '`min_impurity_decrease` должен представлять собой float'
                    ' и быть неотрицательным.'
                    ' Текущее значение `min_impurity_decrease` = -1.0.'
                ),
            ),
        ),
        param(
            'string',
            raises(
                ValueError,
                match=re.escape(
                    '`min_impurity_decrease` должен представлять собой float'
                    ' и быть неотрицательным.'
                    ' Текущее значение `min_impurity_decrease` = string.'
                ),
            ),
        ),
    ],
)
def test_init_params__min_impurity_decrease(min_impurity_decrease, expected):
    with expected:
        _check_init_params(min_impurity_decrease=min_impurity_decrease)


@pytest.mark.parametrize(
    ('max_childs', 'expected'),
    [
        param(float('+inf'), does_not_raise()),
        param(
            1,
            raises(
                ValueError,
                match=re.escape(
                    '`max_childs` должен представлять собой int и быть строго больше 2.'
                    ' Текущее значение `max_childs` = 1.'
                ),
            ),
        ),
        param(
            'string',
            raises(
                ValueError,
                match=re.escape(
                    '`max_childs` должен представлять собой int и быть строго больше 2.'
                    ' Текущее значение `max_childs` = string.'
                ),
            ),
        ),
    ],
)
def test_init_params__max_childs(max_childs, expected):
    with expected:
        _check_init_params(max_childs=max_childs)


@pytest.mark.parametrize(
    ('numerical_feature_names', 'expected'),
    [
        param(None, does_not_raise()),
        param('feature', does_not_raise()),
        param(['feature'], does_not_raise()),
        param(
            1.,
            raises(
                ValueError,
                match=(
                    '`numerical_feature_names` должен представлять собой список строк'
                    ' либо строку.'
                    ' Текущее значение `numerical_feature_names` = 1.0.'
                ),
            ),
        ),
        param(
            [1.],
            raises(
                ValueError,
                match=(
                    'Если `numerical_feature_names` представляет собой список,'
                    ' то должен содержать строки.'
                    f' Элемент списка 1.0 - не строка.'
                ),
            ),
        ),
    ],
)
def test_init_params__numerical_feature_names(numerical_feature_names, expected):
    with expected:
        _check_init_params(numerical_feature_names=numerical_feature_names)


@pytest.mark.parametrize(
    ('categorical_feature_names', 'expected'),
    [
        param(None, does_not_raise()),
        param('feature', does_not_raise()),
        param(['feature'], does_not_raise()),
        param(
            1.,
            raises(
                ValueError,
                match=(
                    '`categorical_feature_names` должен представлять собой список строк'
                    ' либо строку.'
                    ' Текущее значение `categorical_feature_names` = 1.0.'
                ),
            ),
        ),
        param(
            [1.],
            raises(
                ValueError,
                match=(
                    'Если `categorical_feature_names` представляет собой список,'
                    ' то должен содержать строки.'
                    f' Элемент списка 1.0 - не строка.'
                ),
            ),
        ),
    ],
)
def test_init_params__categorical_feature_names(categorical_feature_names, expected):
    with expected:
        _check_init_params(categorical_feature_names=categorical_feature_names)


@pytest.mark.parametrize(
    ('rank_feature_names', 'expected'),
    [
        param(None, does_not_raise()),
        param({'feature': ['a', 'b', 'c']}, does_not_raise()),
        param(
            1,
            raises(
                ValueError,
                match=(
                    '`rank_feature_names` должен представлять собой словарь'
                    ' {название рангового признака: упорядоченный список его значений}.'
                ),
            ),
        ),
        param(
            {1: ['a', 'b', 'c']},
            raises(
                ValueError,
                match=(
                    'Ключи в `rank_feature_names` должны представлять собой строки.'
                    ' 1 - не строка.'
                ),
            ),
        ),
        param(
            {'feature': 'value'},
            raises(
                ValueError,
                match=(
                    'Значения в `rank_feature_names` должны представлять собой списки.'
                    ' Значение feature = value - не список.'
                ),
            ),
        ),
    ],
)
def test_init_params__rank_feature_names(rank_feature_names, expected):
    with expected:
        _check_init_params(rank_feature_names=rank_feature_names)


@pytest.mark.parametrize(
    ('hierarchy', 'expected'),
    [
        param(None, does_not_raise()),
        param({'feature_key': 'feature'}, does_not_raise()),
        param({'feature_key': ['feature1', 'feature2']}, does_not_raise()),
        param(
            'feature',
            raises(
                ValueError,
                match=(
                    '`hierarchy` должен представлять собой словарь {открывающий признак:'
                    ' открывающийся признак / список открывающихся признаков}.'
                    f' Текущее значение `hierarchy` = feature.'
                ),
            ),
        ),
        param(
            {1: 'feature'},
            raises(
                ValueError,
                match=(
                    '`hierarchy` должен представлять собой словарь {открывающий признак:'
                    ' открывающийся признак / список открывающихся признаков}.'
                    ' Значение открывающего признака 1 - не строка.'
                ),
            ),
        ),
        param(
            {'feature_key': 1},
            raises(
                ValueError,
                match=re.escape(
                    '`hierarchy` должен представлять собой словарь {открывающий признак:'
                    ' открывающийся признак / список открывающихся признаков}.'
                    ' Значение открывающегося признака(ов) 1 - не строка (список строк).'
                ),
            ),
        ),
        param(
            {'feature_key': ['feature1', 1]},
            raises(
                ValueError,
                match=re.escape(
                    '`hierarchy` должен представлять собой словарь {открывающий признак:'
                    ' открывающийся признак / список открывающихся признаков}.'
                    " Значение открывающегося признака(ов) ['feature1', 1] - не строка (список строк)."
                ),
            ),
        ),
    ],
)
def test_init_params__hierarchy(hierarchy, expected):
    with expected:
        _check_init_params(hierarchy=hierarchy)


@pytest.mark.parametrize(
    ('numerical_nan_mode', 'expected'),
    [
        param('include', does_not_raise()),
        param('min', does_not_raise()),
        param('max', does_not_raise()),
        param(
            'smth',
            raises(
                ValueError,
                match=(
                    'Для `numerical_nan_mode` доступны значения "include", "min" и "max".'
                    ' Текущее значение `numerical_nan_mode` = smth.'
                ),
            ),
        ),
    ],
)
def test_init_params__numerical_nan_mode(numerical_nan_mode, expected):
    with expected:
        _check_init_params(numerical_nan_mode=numerical_nan_mode)


@pytest.mark.parametrize(
    ('categorical_nan_mode', 'expected'),
    [
        param('include', does_not_raise()),
        param(
            'smth',
            raises(
                ValueError,
                match=(
                    'Для `categorical_nan_mode` доступно значение "include".'
                    ' Текущее значение `categorical_nan_mode` = smth.'
                ),
            ),
        ),
    ],
)
def test_init_params__categorical_nan_mode(categorical_nan_mode, expected):
    with expected:
        _check_init_params(categorical_nan_mode=categorical_nan_mode)
