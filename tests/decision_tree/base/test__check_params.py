import re
from contextlib import nullcontext as does_not_raise

import pytest

from smarttree import SmartDecisionTreeClassifier
from smarttree._types import (
    CatNaModeType,
    ClassificationCriterionType,
    CommonNaModeType,
    NaModeType,
    NumNaModeType,
)


@pytest.mark.parametrize(
    ("criterion", "expected_context"),
    [
        ("gini", does_not_raise()),
        ("entropy", does_not_raise()),
        ("log_loss", does_not_raise()),
        (
            "gjni",
            pytest.raises(
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
def test__check_param__criterion(criterion, expected_context):
    with expected_context:
        criterion: ClassificationCriterionType
        SmartDecisionTreeClassifier(criterion=criterion)


@pytest.mark.parametrize(
    ("max_depth", "expected_context"),
    [
        (None, does_not_raise()),
        (2, does_not_raise()),
        (
            -1,
            pytest.raises(
                ValueError,
                match=(
                    "`max_depth` must be an integer and strictly greater than 0."
                    " The current value of `max_depth` is -1."
                ),
            ),
        ),
        (
            1.5,
            pytest.raises(
                ValueError,
                match=(
                    "`max_depth` must be an integer and strictly greater than 0."
                    " The current value of `max_depth` is 1.5."
                ),
            ),
        ),
        (
            "string",
            pytest.raises(
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
def test__check_param__max_depth(max_depth, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(max_depth=max_depth)


@pytest.mark.parametrize(
    ("min_samples_split", "expected_context"),
    [
        (2, does_not_raise()),
        (
            1,
            pytest.raises(
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
            pytest.raises(
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
            pytest.raises(
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
            pytest.raises(
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
def test__check_param__min_samples_split(min_samples_split, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(min_samples_split=min_samples_split)


@pytest.mark.parametrize(
    ("min_samples_leaf", "expected_context"),
    [
        (1, does_not_raise()),
        (
            0,
            pytest.raises(
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
            pytest.raises(
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
            pytest.raises(
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
            pytest.raises(
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
def test__check_params__min_samples_leaf(min_samples_leaf, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(min_samples_leaf=min_samples_leaf)


@pytest.mark.parametrize(
    ("max_leaf_nodes", "expected_context"),
    [
        (
            .0,
            pytest.raises(
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
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`max_leaf_nodes` must be an integer and strictly greater than 2."
                    " The current value of `max_leaf_nodes` is 1."
                ),
            ),
        ),
        (
            "string",
            pytest.raises(
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
def test__check_params__max_leaf_nodes(max_leaf_nodes, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)


@pytest.mark.parametrize(
    ("min_impurity_decrease", "expected_context"),
    [
        (.0, does_not_raise()),
        (
            -1.,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`min_impurity_decrease` must be float and non-negative."
                    " The current value of `min_impurity_decrease` is -1.0."
                ),
            ),
        ),
        (
            "string",
            pytest.raises(
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
def test__check_params__min_impurity_decrease(min_impurity_decrease, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(min_impurity_decrease=min_impurity_decrease)


@pytest.mark.parametrize(
    ("max_childs", "expected_context"),
    [
        (None, does_not_raise()),
        (2, does_not_raise()),
        (
            float("+inf"),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`max_childs` must be integer and strictly greater than 2."
                    " The current value of `max_childs` is inf."
                ),
            ),
        ),
        (
            1,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`max_childs` must be integer and strictly greater than 2."
                    " The current value of `max_childs` is 1."
                ),
            ),
        ),
        (
            "string",
            pytest.raises(
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
def test__check_params__max_childs(max_childs, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(max_childs=max_childs)


@pytest.mark.parametrize(
    ("num_features", "expected_context"),
    [
        (None, does_not_raise()),
        ("feature", does_not_raise()),
        (["feature"], does_not_raise()),
        (
            1.,
            pytest.raises(
                ValueError,
                match=(
                    "`num_features` must be a string or list of strings."
                    " The current value of `num_features` is 1.0."
                ),
            ),
        ),
        (
            [1.],
            pytest.raises(
                ValueError,
                match=(
                    "If `num_features` is a list, it must consists of strings."
                    " The element 1.0 of the list isnt a string."
                ),
            ),
        ),
    ],
    ids=["None", "str", "list[str]", "float", "list[float]"],
)
def test__check_params__num_features(num_features, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(num_features=num_features)


@pytest.mark.parametrize(
    ("cat_features", "expected_context"),
    [
        (None, does_not_raise()),
        ("feature", does_not_raise()),
        (["feature"], does_not_raise()),
        (
            1.,
            pytest.raises(
                ValueError,
                match=(
                    "`cat_features` must be a string or list of strings."
                    " The current value of `cat_features` is 1.0."
                ),
            ),
        ),
        (
            [1.],
            pytest.raises(
                ValueError,
                match=(
                    "If `cat_features` is a list, it must consists of strings."
                    " The element 1.0 of the list isnt a string."
                ),
            ),
        ),
    ],
    ids=["None", "str", "list[str]", "float", "list[float]"],
)
def test__check_params__cat_features(cat_features, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(cat_features=cat_features)


@pytest.mark.parametrize(
    ("rank_features", "expected_context"),
    [
        (None, does_not_raise()),
        ({"feature": ["a", "b", "c"]}, does_not_raise()),
        (
            1,
            pytest.raises(
                ValueError,
                match=(
                    "`rank_features` must be a dictionary"
                    " {rang feature name: list of its ordered values}."
                ),
            ),
        ),
        (
            {1: ["a", "b", "c"]},
            pytest.raises(
                ValueError,
                match=(
                    "Keys in `rank_features` must be a strings."
                    " The key 1 isnt a string."
                ),
            ),
        ),
        (
            {"feature": "value"},
            pytest.raises(
                ValueError,
                match=(
                    "Values in `rank_features` must be lists."
                    " The value value of the key feature isnt a list."
                ),
            ),
        ),
    ],
    ids=["None", "dict[str, list[str]", "int", "dict[int, list[str]]", "dict[str, str]"],
)
def test__check_params__rank_features(rank_features, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(rank_features=rank_features)


@pytest.mark.parametrize(
    ("params_to_set", "expected_context"),
    [
        ({"num_features": ["f1", "f2"]}, does_not_raise()),
        (
            {"num_features": ["f1", "f1"]},
            pytest.raises(ValueError, match="`num_features` contains duplicates.")
        ),
        ({"cat_features": ["f1", "f2"]}, does_not_raise()),
        (
            {"cat_features": ["f1", "f1"]},
            pytest.raises(ValueError, match="`cat_features` contains duplicates.")
        ),
    ],
)
def test__check_params__features_contain_duplicates(params_to_set, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(**params_to_set)


@pytest.mark.parametrize(
    ("n_features", "c_features", "r_features", "expected_context"),
    [
        ("f1", "f2", None, does_not_raise()),
        (
            "f1",
            "f1",
            None,
            pytest.raises(
                ValueError,
                match=(
                    "Following feature names are ambiguous, they are defined in"
                    " both 'num_features' and 'cat_features': {'f1'}."
                ),
            ),
        ),
        ("f1", None, {"f2": ["v2"]}, does_not_raise()),
        (
            "f1",
            None,
            {"f1": ["v1"]},
            pytest.raises(
                ValueError,
                match=(
                    "Following feature names are ambiguous, they are defined in"
                    " both 'num_features' and 'rank_features': {'f1'}."
                ),
            ),
        ),
        (None, "f1", {"f2": ["v2"]}, does_not_raise()),
        (
            None,
            "f1",
            {"f1": ["v1"]},
            pytest.raises(
                ValueError,
                match=(
                    "Following feature names are ambiguous, they are defined in"
                    " both 'cat_features' and 'rank_features': {'f1'}."
                ),
            ),
        ),
    ],
)
def test__check_params__ambiguous(n_features, c_features, r_features, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(
            num_features=n_features, cat_features=c_features, rank_features=r_features
        )


@pytest.mark.parametrize(
    ("hierarchy", "expected_context"),
    [
        (None, does_not_raise()),
        ({"feature_key": "feature"}, does_not_raise()),
        ({"feature_key": ["feature1", "feature2"]}, does_not_raise()),
        (
            "feature",
            pytest.raises(
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
            pytest.raises(
                ValueError,
                match=(
                    "`hierarchy` must be a dictionary"
                    " {opening feature: opened feature / list of opened features}."
                    " Value 1 of opening feature isnt a string."
                ),
            ),
        ),
        (
            {"feature_key": 1},
            pytest.raises(
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
            pytest.raises(
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
def test__check_params__hierarchy(hierarchy, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(hierarchy=hierarchy)


@pytest.mark.parametrize(
    ("na_mode", "expected_context"),
    [
        ("include_all", does_not_raise()),
        ("include_best", does_not_raise()),
        (
            "smth",
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`na_mode` must be Literal['include_all', 'include_best']."
                    " The current value of `na_mode` is 'smth'."
                ),
            ),
        ),
    ],
    ids=["include_all", "include_best", "invalid"],
)
def test__check_params__na_mode(na_mode, expected_context):
    with expected_context:
        na_mode: CommonNaModeType
        SmartDecisionTreeClassifier(na_mode=na_mode)



@pytest.mark.parametrize(
    ("num_na_mode", "expected_context"),
    [
        ("min", does_not_raise()),
        ("max", does_not_raise()),
        ("include_all", does_not_raise()),
        ("include_best", does_not_raise()),
        (
            "smth",
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`num_na_mode` must be Literal['min', 'max', 'include_all', 'include_best']."
                    " The current value of `num_na_mode` is 'smth'."
                ),
            ),
        ),
    ],
    ids=["min", "max", "include_all", "include_best", "invalid"],
)
def test__check_params__num_na_mode(num_na_mode, expected_context):
    with expected_context:
        num_na_mode: NumNaModeType
        SmartDecisionTreeClassifier(num_na_mode=num_na_mode)


@pytest.mark.parametrize(
    ("cat_na_mode", "expected_context"),
    [
        ("as_category", does_not_raise()),
        ("include_all", does_not_raise()),
        ("include_best", does_not_raise()),
        (
            "smth",
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`cat_na_mode` must be Literal['as_category', 'include_all', 'include_best']."
                    " The current value of `cat_na_mode` is 'smth'."
                ),
            ),
        ),
    ],
    ids=["as_category", "include_all", "include_best", "invalid"],
)
def test__check_params__cat_na_mode(cat_na_mode, expected_context):
    with expected_context:
        cat_na_mode: CatNaModeType
        SmartDecisionTreeClassifier(cat_na_mode=cat_na_mode)


@pytest.mark.parametrize(
    ("cat_na_filler", "expected_context"),
    [
        ("na", does_not_raise()),
        (
            1,
            pytest.raises(
                ValueError,
                match=(
                    "`cat_na_filler` must be a string."
                    " The current value of `cat_na_filler` is 1."
                ),
            ),
        ),
    ],
    ids=["str", "int"],
)
def test__check_param__cat_na_filler(cat_na_filler, expected_context):
    with expected_context:
        SmartDecisionTreeClassifier(cat_na_filler=cat_na_filler)


@pytest.mark.parametrize(
    ("rank_na_mode", "expected_context"),
    [
        ("include_all", does_not_raise()),
        ("include_best", does_not_raise()),
        (
            "smth",
            pytest.raises(
                ValueError,
                match=re.escape(
                    "`rank_na_mode` must be Literal['include_all', 'include_best']."
                    " The current value of `rank_na_mode` is 'smth'."
                ),
            ),
        ),
    ],
    ids=["include_all", "include_best", "invalid"],
)
def test__check_params__rank_na_mode(rank_na_mode, expected_context):
    with expected_context:
        rank_na_mode: CommonNaModeType
        SmartDecisionTreeClassifier(rank_na_mode=rank_na_mode)


@pytest.mark.parametrize(
    ("min_samples_split", "min_samples_leaf", "expected_context"),
    [
        (2, 1, does_not_raise()),
        (
            2,
            2,
            pytest.raises(
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
def test__check_params__min_samples_split__min_samples_leaf(
    min_samples_split, min_samples_leaf, expected_context
):
    with expected_context:
        SmartDecisionTreeClassifier(
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf
        )


@pytest.mark.parametrize(
    ("feature_na_mode", "expected_context"),
    [
        ({"feature": "min"}, does_not_raise()),
        (
            "string",
            pytest.raises(
                ValueError,
                match=(
                    "`feature_na_mode` must be a dictionary {feature name: NA mode}."
                    " The current value of `feature_na_mode` is 'string'."
                ),
            ),
        ),
        (
            {1: "min"},
            pytest.raises(
                ValueError,
                match=(
                    "Keys in `feature_na_mode` must be a strings."
                    " The key 1 isnt a string."
                ),
            ),
        ),
        (
            {"feature": "mex"},
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Values in `feature_na_mode` must be "
                    "Literal['min', 'max', 'as_category' 'include_all', 'include_best']."
                    " The current value of `na_mode` for `feature` 'feature' is 'mex'."
                ),
            ),
        ),
    ],
    ids=["valid", "not_dict", "invalid_key", "invalid_value"],
)
def test__check_params__feature_na_mode(feature_na_mode, expected_context):
    with expected_context:
        feature_na_mode: dict[str, NaModeType]
        SmartDecisionTreeClassifier(feature_na_mode=feature_na_mode)
