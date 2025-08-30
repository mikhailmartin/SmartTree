import re
from contextlib import nullcontext as does_not_raise

import pytest
from pytest import param, raises

from smarttree import SmartDecisionTreeClassifier


@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        param("gini", does_not_raise()),
        param("entropy", does_not_raise()),
        param("log_loss", does_not_raise()),
        param(
            "gjni",
            raises(
                ValueError,
                match=re.escape(
                    "`criterion` mist be Literal['entropy', 'log_loss', 'gini']."
                    " The current value of `criterion` is 'gjni'."
                ),
            ),
            id="invalid-criterion",
        ),
    ],
)
def test_init_param__criterion(criterion, expected):
    with expected:
        SmartDecisionTreeClassifier(criterion=criterion)


@pytest.mark.parametrize(
    ("max_depth", "expected"),
    [
        param(None, does_not_raise()),
        param(2, does_not_raise()),
        param(
            -1,
            raises(
                ValueError,
                match=(
                    "`max_depth` must be an integer and strictly greater than 0."
                    " The current value of `max_depth` is -1."
                ),
            ),
        ),
        param(
            1.5,
            raises(
                ValueError,
                match=(
                    "`max_depth` must be an integer and strictly greater than 0."
                    " The current value of `max_depth` is 1.5."
                ),
            ),
        ),
        param(
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
)
def test_init_param__max_depth(max_depth, expected):
    with expected:
        SmartDecisionTreeClassifier(max_depth=max_depth)


@pytest.mark.parametrize(
    ("min_samples_split", "expected"),
    [
        param(2, does_not_raise()),
        param(
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
        param(.5, does_not_raise()),
        param(
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
        param(
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
        param(
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
)
def test_init_param__min_samples_split(min_samples_split, expected):
    with expected:
        SmartDecisionTreeClassifier(min_samples_split=min_samples_split)


@pytest.mark.parametrize(
    ("min_samples_leaf", "expected"),
    [
        param(1, does_not_raise()),
        param(
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
        param(.5, does_not_raise()),
        param(
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
        param(
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
        param(
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
)
def test_init_params__min_samples_leaf(min_samples_leaf, expected):
    with expected:
        SmartDecisionTreeClassifier(min_samples_leaf=min_samples_leaf)


@pytest.mark.parametrize(
    ("max_leaf_nodes", "expected"),
    [
        param(
            .0,
            raises(
                ValueError,
                match=re.escape(
                    "`max_leaf_nodes` must be an integer and strictly greater than 2."
                    " The current value of `max_leaf_nodes` is 0.0."
                ),
            ),
        ),
        param(2, does_not_raise()),
        param(
            1,
            raises(
                ValueError,
                match=re.escape(
                    "`max_leaf_nodes` must be an integer and strictly greater than 2."
                    " The current value of `max_leaf_nodes` is 1."
                ),
            ),
        ),
        param(
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
)
def test_init_params__max_leaf_nodes(max_leaf_nodes, expected):
    with expected:
        SmartDecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)


@pytest.mark.parametrize(
    ("min_impurity_decrease", "expected"),
    [
        param(.0, does_not_raise()),
        param(
            -1.,
            raises(
                ValueError,
                match=re.escape(
                    "`min_impurity_decrease` must be float and non-negative."
                    " The current value of `min_impurity_decrease` is -1.0."
                ),
            ),
        ),
        param(
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
)
def test_init_params__min_impurity_decrease(min_impurity_decrease, expected):
    with expected:
        SmartDecisionTreeClassifier(min_impurity_decrease=min_impurity_decrease)


@pytest.mark.parametrize(
    ("max_childs", "expected"),
    [
        param(None, does_not_raise()),
        param(2, does_not_raise()),
        param(
            float("+inf"),
            raises(
                ValueError,
                match=re.escape(
                    "`max_childs` must be integer and strictly greater than 2."
                    " The current value of `max_childs` is inf."
                ),
            ),
        ),
        param(
            1,
            raises(
                ValueError,
                match=re.escape(
                    "`max_childs` must be integer and strictly greater than 2."
                    " The current value of `max_childs` is 1."
                ),
            ),
        ),
        param(
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
)
def test_init_params__max_childs(max_childs, expected):
    with expected:
        SmartDecisionTreeClassifier(max_childs=max_childs)


@pytest.mark.parametrize(
    ("numerical_feature_names", "expected"),
    [
        param(None, does_not_raise()),
        param("feature", does_not_raise()),
        param(["feature"], does_not_raise()),
        param(
            1.,
            raises(
                ValueError,
                match=(
                    "`numerical_feature_names` must be a string or list of strings."
                    " The current value of `numerical_feature_names` is 1.0."
                ),
            ),
        ),
        param(
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
)
def test_init_params__numerical_feature_names(numerical_feature_names, expected):
    with expected:
        SmartDecisionTreeClassifier(numerical_feature_names=numerical_feature_names)


@pytest.mark.parametrize(
    ("categorical_feature_names", "expected"),
    [
        param(None, does_not_raise()),
        param("feature", does_not_raise()),
        param(["feature"], does_not_raise()),
        param(
            1.,
            raises(
                ValueError,
                match=(
                    "`categorical_feature_names` must be a string or list of strings."
                    " The current value of `categorical_feature_names` is 1.0."
                ),
            ),
        ),
        param(
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
)
def test_init_params__categorical_feature_names(categorical_feature_names, expected):
    with expected:
        SmartDecisionTreeClassifier(categorical_feature_names=categorical_feature_names)


@pytest.mark.parametrize(
    ("rank_feature_names", "expected"),
    [
        param(None, does_not_raise()),
        param({"feature": ["a", "b", "c"]}, does_not_raise()),
        param(
            1,
            raises(
                ValueError,
                match=(
                    "`rank_feature_names` must be a dictionary"
                    " {rang feature name: list of its ordered values}."
                ),
            ),
        ),
        param(
            {1: ["a", "b", "c"]},
            raises(
                ValueError,
                match=(
                    "Keys in `rank_feature_names` must be a strings."
                    " The key 1 isnt a string."
                ),
            ),
        ),
        param(
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
)
def test_init_params__rank_feature_names(rank_feature_names, expected):
    with expected:
        SmartDecisionTreeClassifier(rank_feature_names=rank_feature_names)


@pytest.mark.parametrize(
    ("hierarchy", "expected"),
    [
        param(None, does_not_raise()),
        param({"feature_key": "feature"}, does_not_raise()),
        param({"feature_key": ["feature1", "feature2"]}, does_not_raise()),
        param(
            "feature",
            raises(
                ValueError,
                match=re.escape(
                    "`hierarchy` must be a dictionary"
                    " {opening feature: opened feature / list of opened strings}."
                    " The current value of `hierarchy` is 'feature'."
                ),
            ),
        ),
        param(
            {1: "feature"},
            raises(
                ValueError,
                match=(
                    "`hierarchy` must be a dictionary"
                    " {opening feature: opened feature / list of opened strings}."
                    f" Value 1 of opening feature isnt a string."
                ),
            ),
        ),
        param(
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
        param(
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
)
def test_init_params__hierarchy(hierarchy, expected):
    with expected:
        SmartDecisionTreeClassifier(hierarchy=hierarchy)


@pytest.mark.parametrize(
    ("numerical_nan_mode", "expected"),
    [
        param("include", does_not_raise()),
        param("min", does_not_raise()),
        param("max", does_not_raise()),
        param(
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
)
def test_init_params__numerical_nan_mode(numerical_nan_mode, expected):
    with expected:
        SmartDecisionTreeClassifier(numerical_nan_mode=numerical_nan_mode)


@pytest.mark.parametrize(
    ("categorical_nan_mode", "expected"),
    [
        param("include", does_not_raise()),
        param("as_category", does_not_raise()),
        param(
            "smth",
            raises(
                ValueError,
                match=re.escape(
                    "`categorical_nan_mode` must be Literal['include', 'as_category']."
                    " The current value of `categorical_nan_mode` is 'smth'."
                ),
            ),
        ),
    ],
)
def test_init_params__categorical_nan_mode(categorical_nan_mode, expected):
    with expected:
        SmartDecisionTreeClassifier(categorical_nan_mode=categorical_nan_mode)


@pytest.mark.parametrize(
    ("categorical_nan_filler", "expected"),
    [
        param("nan", does_not_raise()),
        param(
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
)
def test_init_param__categorical_nan_filler(categorical_nan_filler, expected):
    with expected:
        SmartDecisionTreeClassifier(categorical_nan_filler=categorical_nan_filler)
