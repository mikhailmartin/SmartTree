import re
from contextlib import nullcontext as does_not_raise

import pytest
from pytest import raises

from smarttree import SmartDecisionTreeClassifier
from smarttree._constants import (
    CategoricalNanModeOption,
    ClassificationCriterionOption,
    NumericalNanModeOption,
)


@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        ("gini", does_not_raise()),
        ("entropy", does_not_raise()),
        ("log_loss", does_not_raise()),
        (
            "gjni",
            raises(
                ValueError,
                match=re.escape(
                    "`criterion` mist be Literal['entropy', 'log_loss', 'gini']."
                    " The current value of `criterion` is 'gjni'."
                ),
            ),
        ),
    ],
    ids=["gini", "entropy", "log_loss", "invalid"]
)
def test_init_param__criterion(criterion, expected):
    with expected:
        criterion: ClassificationCriterionOption
        SmartDecisionTreeClassifier(criterion=criterion)


@pytest.mark.parametrize(
    ("max_depth", "expected"),
    [
        (None, does_not_raise()),
        (2, does_not_raise()),
        (
            -1,
            raises(
                ValueError,
                match=(
                    "`max_depth` must be an integer and strictly greater than 0."
                    " The current value of `max_depth` is -1."
                ),
            ),
        ),
        (
            1.5,
            raises(
                ValueError,
                match=(
                    "`max_depth` must be an integer and strictly greater than 0."
                    " The current value of `max_depth` is 1.5."
                ),
            ),
        ),
        (
            "string",
            raises(
                ValueError,
                match=(
                    "`max_depth` must be an integer and strictly greater than 0."
                    " The current value of `max_depth` is 'string'."
                ),
            ),
        ),
    ],
    ids=["None", "2", "negative", "float", "str"],
)
def test_init_param__max_depth(max_depth, expected):
    with expected:
        SmartDecisionTreeClassifier(max_depth=max_depth)


@pytest.mark.parametrize(
    ("min_samples_split", "expected"),
    [
        (2, does_not_raise()),
        (
            1,
            raises(
                ValueError,
                match=re.escape(
                    "`min_samples_split` must be an integer and lie in the range"
                    " [2, +inf), or float and lie in the range (0, 1)."
                    " The current value of `min_samples_split` is 1."
                ),
            ),
        ),
        (.5, does_not_raise()),
        (
            0.0,
            raises(
                ValueError,
                match=re.escape(
                    "`min_samples_split` must be an integer and lie in the range"
                    " [2, +inf), or float and lie in the range (0, 1)."
                    " The current value of `min_samples_split` is 0.0."
                ),
            ),
        ),
        (
            1.0,
            raises(
                ValueError,
                match=re.escape(
                    "`min_samples_split` must be an integer and lie in the range"
                    " [2, +inf), or float and lie in the range (0, 1)."
                    " The current value of `min_samples_split` is 1.0."
                ),
            ),
        ),
        (
            "string",
            raises(
                ValueError,
                match=re.escape(
                    "`min_samples_split` must be an integer and lie in the range"
                    " [2, +inf), or float and lie in the range (0, 1)."
                    " The current value of `min_samples_split` is 'string'."
                ),
            ),
        ),
    ],
    ids=["int(2)", "int(1)", "float(0.5)", "float(0.0)", "float(1.0)", "str"],
)
def test_init_param__min_samples_split(min_samples_split, expected):
    with expected:
        SmartDecisionTreeClassifier(min_samples_split=min_samples_split)


@pytest.mark.parametrize(
    ("min_samples_leaf", "expected"),
    [
        (1, does_not_raise()),
        (
            0,
            raises(
                ValueError,
                match=re.escape(
                    "`min_samples_leaf` must be an integer and lie in the range"
                    " [1, +inf), or float and lie in the range (0, 1)."
                    " The current value of `min_samples_leaf` is 0."
                ),
            ),
        ),
        (.5, does_not_raise()),
        (
            0.0,
            raises(
                ValueError,
                match=re.escape(
                    "`min_samples_leaf` must be an integer and lie in the range"
                    " [1, +inf), or float and lie in the range (0, 1)."
                    " The current value of `min_samples_leaf` is 0.0."
                ),
            ),
        ),
        (
            1.0,
            raises(
                ValueError,
                match=re.escape(
                    "`min_samples_leaf` must be an integer and lie in the range"
                    " [1, +inf), or float and lie in the range (0, 1)."
                    " The current value of `min_samples_leaf` is 1.0."
                ),
            ),
        ),
        (
            "string",
            raises(
                ValueError,
                match=re.escape(
                    "`min_samples_leaf` must be an integer and lie in the range"
                    " [1, +inf), or float and lie in the range (0, 1)."
                    " The current value of `min_samples_leaf` is 'string'."
                ),
            ),
        ),
    ],
    ids=["int(1)", "int(0)", "float(0.5)", "float(0.0)", "float(1.0)", "str"],
)
def test_init_params__min_samples_leaf(min_samples_leaf, expected):
    with expected:
        SmartDecisionTreeClassifier(min_samples_leaf=min_samples_leaf)


@pytest.mark.parametrize(
    ("max_leaf_nodes", "expected"),
    [
        (
            .0,
            raises(
                ValueError,
                match=re.escape(
                    "`max_leaf_nodes` must be an integer and strictly greater than 2."
                    " The current value of `max_leaf_nodes` is 0.0."
                ),
            ),
        ),
        (2, does_not_raise()),
        (
            1,
            raises(
                ValueError,
                match=re.escape(
                    "`max_leaf_nodes` must be an integer and strictly greater than 2."
                    " The current value of `max_leaf_nodes` is 1."
                ),
            ),
        ),
        (
            "string",
            raises(
                ValueError,
                match=re.escape(
                    "`max_leaf_nodes` must be an integer and strictly greater than 2."
                    " The current value of `max_leaf_nodes` is 'string'."
                ),
            ),
        ),
    ],
    ids=["float", "2", "1", "str"],
)
def test_init_params__max_leaf_nodes(max_leaf_nodes, expected):
    with expected:
        SmartDecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)


@pytest.mark.parametrize(
    ("min_impurity_decrease", "expected"),
    [
        (.0, does_not_raise()),
        (
            -1.,
            raises(
                ValueError,
                match=re.escape(
                    "`min_impurity_decrease` must be float and non-negative."
                    " The current value of `min_impurity_decrease` is -1.0."
                ),
            ),
        ),
        (
            "string",
            raises(
                ValueError,
                match=re.escape(
                    "`min_impurity_decrease` must be float and non-negative."
                    " The current value of `min_impurity_decrease` is 'string'."
                ),
            ),
        ),
    ],
    ids=["float(0.0)", "float(negative)", "str"]
)
def test_init_params__min_impurity_decrease(min_impurity_decrease, expected):
    with expected:
        SmartDecisionTreeClassifier(min_impurity_decrease=min_impurity_decrease)


@pytest.mark.parametrize(
    ("max_childs", "expected"),
    [
        (None, does_not_raise()),
        (2, does_not_raise()),
        (
            float("+inf"),
            raises(
                ValueError,
                match=re.escape(
                    "`max_childs` must be integer and strictly greater than 2."
                    " The current value of `max_childs` is inf."
                ),
            ),
        ),
        (
            1,
            raises(
                ValueError,
                match=re.escape(
                    "`max_childs` must be integer and strictly greater than 2."
                    " The current value of `max_childs` is 1."
                ),
            ),
        ),
        (
            "string",
            raises(
                ValueError,
                match=re.escape(
                    "`max_childs` must be integer and strictly greater than 2."
                    " The current value of `max_childs` is 'string'."
                ),
            ),
        ),
    ],
    ids=["None", "2", "float", "1", "str"],
)
def test_init_params__max_childs(max_childs, expected):
    with expected:
        SmartDecisionTreeClassifier(max_childs=max_childs)


@pytest.mark.parametrize(
    ("numerical_feature_names", "expected"),
    [
        (None, does_not_raise()),
        ("feature", does_not_raise()),
        (["feature"], does_not_raise()),
        (
            1.,
            raises(
                ValueError,
                match=(
                    "`numerical_feature_names` must be a string or list of strings."
                    " The current value of `numerical_feature_names` is 1.0."
                ),
            ),
        ),
        (
            [1.],
            raises(
                ValueError,
                match=(
                    "If `numerical_feature_names` is a list, it must consists of"
                    " strings. The element 1.0 of the list isnt a string."
                ),
            ),
        ),
    ],
    ids=["None", "str", "list[str]", "float", "list[float]"],
)
def test_init_params__numerical_feature_names(numerical_feature_names, expected):
    with expected:
        SmartDecisionTreeClassifier(numerical_feature_names=numerical_feature_names)


@pytest.mark.parametrize(
    ("categorical_feature_names", "expected"),
    [
        (None, does_not_raise()),
        ("feature", does_not_raise()),
        (["feature"], does_not_raise()),
        (
            1.,
            raises(
                ValueError,
                match=(
                    "`categorical_feature_names` must be a string or list of strings."
                    " The current value of `categorical_feature_names` is 1.0."
                ),
            ),
        ),
        (
            [1.],
            raises(
                ValueError,
                match=(
                    "If `categorical_feature_names` is a list, it must consists of"
                    " strings. The element 1.0 of the list isnt a string."
                ),
            ),
        ),
    ],
    ids=["None", "str", "list[str]", "float", "list[float]"],
)
def test_init_params__categorical_feature_names(categorical_feature_names, expected):
    with expected:
        SmartDecisionTreeClassifier(categorical_feature_names=categorical_feature_names)


@pytest.mark.parametrize(
    ("rank_feature_names", "expected"),
    [
        (None, does_not_raise()),
        ({"feature": ["a", "b", "c"]}, does_not_raise()),
        (
            1,
            raises(
                ValueError,
                match=(
                    "`rank_feature_names` must be a dictionary"
                    " {rang feature name: list of its ordered values}."
                ),
            ),
        ),
        (
            {1: ["a", "b", "c"]},
            raises(
                ValueError,
                match=(
                    "Keys in `rank_feature_names` must be a strings."
                    " The key 1 isnt a string."
                ),
            ),
        ),
        (
            {"feature": "value"},
            raises(
                ValueError,
                match=(
                    "Values in `rank_feature_names` must be lists."
                    " The value value of the key feature isnt a list."
                ),
            ),
        ),
    ],
    ids=["None", "dict[str, list[str]", "int", "dict[int, list[str]]", "dict[str, str]"],
)
def test_init_params__rank_feature_names(rank_feature_names, expected):
    with expected:
        SmartDecisionTreeClassifier(rank_feature_names=rank_feature_names)


@pytest.mark.parametrize(
    ("hierarchy", "expected"),
    [
        (None, does_not_raise()),
        ({"feature_key": "feature"}, does_not_raise()),
        ({"feature_key": ["feature1", "feature2"]}, does_not_raise()),
        (
            "feature",
            raises(
                ValueError,
                match=re.escape(
                    "`hierarchy` must be a dictionary"
                    " {opening feature: opened feature / list of opened features}."
                    " The current value of `hierarchy` is 'feature'."
                ),
            ),
        ),
        (
            {1: "feature"},
            raises(
                ValueError,
                match=(
                    "`hierarchy` must be a dictionary"
                    " {opening feature: opened feature / list of opened features}."
                    f" Value 1 of opening feature isnt a string."
                ),
            ),
        ),
        (
            {"feature_key": 1},
            raises(
                ValueError,
                match=re.escape(
                    "`hierarchy` must be a dictionary"
                    " {opening feature: opened feature / list of opened features}."
                    " Value 1 of opened feature(s) isnt a string (list of strings)."
                ),
            ),
        ),
        (
            {"feature_key": ["feature1", 1]},
            raises(
                ValueError,
                match=(
                    "`hierarchy` must be a dictionary"
                    " {opening feature: opened feature / list of opened features}."
                    " Value 1 of opened feature isnt a string."
                ),
            ),
        ),
    ],
    ids=[
        "None",
        "dict[str, str]",
        "dict[str, list[str]",
        "str",
        "dict[int: str]",
        "dict[str, int]",
        "dict[str, list[str | int]]",
    ],
)
def test_init_params__hierarchy(hierarchy, expected):
    with expected:
        SmartDecisionTreeClassifier(hierarchy=hierarchy)


@pytest.mark.parametrize(
    ("numerical_nan_mode", "expected"),
    [
        ("include", does_not_raise()),
        ("min", does_not_raise()),
        ("max", does_not_raise()),
        (
            "smth",
            raises(
                ValueError,
                match=re.escape(
                    "`numerical_nan_mode` must be Literal['include', 'min', 'max']."
                    " The current value of `numerical_nan_mode` is 'smth'."
                ),
            ),
        ),
    ],
    ids=["include", "min", "max", "invalid"],
)
def test_init_params__numerical_nan_mode(numerical_nan_mode, expected):
    with expected:
        numerical_nan_mode: NumericalNanModeOption
        SmartDecisionTreeClassifier(numerical_nan_mode=numerical_nan_mode)


@pytest.mark.parametrize(
    ("categorical_nan_mode", "expected"),
    [
        ("as_category", does_not_raise()),
        ("include_all", does_not_raise()),
        ("include_best", does_not_raise()),
        (
            "smth",
            raises(
                ValueError,
                match=re.escape(
                    "`categorical_nan_mode` must be Literal['as_category', 'include_all', 'include_best']."
                    " The current value of `categorical_nan_mode` is 'smth'."
                ),
            ),
        ),
    ],
    ids=["as_category", "include_all", "include_best", "invalid"],
)
def test_init_params__categorical_nan_mode(categorical_nan_mode, expected):
    with expected:
        categorical_nan_mode: CategoricalNanModeOption
        SmartDecisionTreeClassifier(categorical_nan_mode=categorical_nan_mode)


@pytest.mark.parametrize(
    ("categorical_nan_filler", "expected"),
    [
        ("nan", does_not_raise()),
        (
            1,
            raises(
                ValueError,
                match=(
                    "`categorical_nan_filler` must be a string."
                    " The current value of `categorical_nan_filler` is 1."
                ),
            ),
        ),
    ],
    ids=["str", "int"],
)
def test_init_param__categorical_nan_filler(categorical_nan_filler, expected):
    with expected:
        SmartDecisionTreeClassifier(categorical_nan_filler=categorical_nan_filler)


@pytest.mark.parametrize(
    ("min_samples_split", "min_samples_leaf", "expected"),
    [
        (2, 1, does_not_raise()),
        (
            2,
            2,
            raises(
                ValueError,
                match=(
                    "`min_samples_split` must be strictly 2 times greater than"
                    " `min_samples_leaf`. Current values of `min_samples_split` is"
                    " 2, of `min_samples_leaf` is 2."
                ),
            ),
        ),
    ],
    ids=["valid", "invalid"],
)
def test_init_params__min_samples_split__min_samples_leaf(
    min_samples_split, min_samples_leaf, expected
):
    with expected:
        SmartDecisionTreeClassifier(
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf
        )
