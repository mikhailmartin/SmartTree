import re
from contextlib import nullcontext as does_not_raise

import pandas as pd
import pytest

from smarttree import SmartDecisionTreeClassifier


NUM_FEATURE = "2. Возраст"
CAT_FEATURE = "3. Семейное положение"
RANK_FEATURE = "5. В какой семье Вы выросли?"
RANK_VALUES = [
    "полная семья, кровные родители",
    "мачеха/отчим",
    "мать/отец одиночка",
    "с бабушкой и дедушкой",
    "в детском доме",
]
SELECTED = [NUM_FEATURE, CAT_FEATURE, RANK_FEATURE]


@pytest.fixture(scope="function")
def tree():
    return SmartDecisionTreeClassifier(
        max_depth=1,
        num_features=[NUM_FEATURE],
        cat_features=[CAT_FEATURE],
        rank_features={RANK_FEATURE: RANK_VALUES},
    )


@pytest.mark.parametrize(
    ("X_scenario", "y_scenario", "expected_context"),
    [
        ("valid", "valid", does_not_raise()),
        ("valid", "not_series", pytest.raises(ValueError, match="y must be a pandas.Series.")),
        ("valid", "short", pytest.raises(ValueError, match="X and y must be the equal length.")),
        ("valid", "contain_na", pytest.raises(ValueError, match="y must not contain NA.")),
        ("not_df", "valid", pytest.raises(ValueError, match="X must be a pandas.DataFrame.")),
        (
            "missing_num",
            "valid",
            pytest.raises(
                ValueError,
                match=(
                    f"`num_features` contain feature {NUM_FEATURE!r},"
                    " which isnt present in the training data."
                ),
            ),
        ),
        (
            "missing_cat",
            "valid",
            pytest.raises(
                ValueError,
                match=(
                    f"`cat_features` contain feature {CAT_FEATURE!r},"
                    " which isnt present in the training data."
                ),
            ),
        ),
        (
            "missing_rank",
            "valid",
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"`rank_features` contain feature {RANK_FEATURE!r},"
                    " which isnt present in the training data."
                ),
            ),
        ),
    ],
    ids=[
        "valid", "not_series", "short", "not_df", "contain_na",
        "missing_num", "missing_cat", "missing_rank",
    ],
)
def test__check_data__fit(X, y, X_scenario, y_scenario, tree, expected_context):

    X_map = {
        "valid": X[SELECTED],
        "not_df": "X",
        "missing_num": X[SELECTED].drop(columns=NUM_FEATURE),
        "missing_cat": X[SELECTED].drop(columns=CAT_FEATURE),
        "missing_rank": X[SELECTED].drop(columns=RANK_FEATURE),
    }
    y_map = {
        "valid": y,
        "not_series": "y",
        "short": y[:-1],
        "contain_na": y.where(y.index != 0, pd.NA),
    }

    X_fit = X_map[X_scenario]
    y_fit = y_map[y_scenario]

    with expected_context:
        tree.fit(X_fit, y_fit)


@pytest.mark.parametrize(
    ("X_scenario", "expected_context"),
    [
        ("valid", does_not_raise()),
        ("not_df", pytest.raises(ValueError, match="X must be a pandas.DataFrame.")),
        (
            "renamed",
            pytest.raises(
                ValueError,
                match=(
                    "Feature names unseen at fit time:\n"
                    "- '2. Age'\n"
                    "Feature names seen at fit time, yet now missing:\n"
                    "- '2. Возраст'"
                ),
            ),
        ),
    ],
    ids=["valid", "not_df", "renamed"],
)
def test__check_data__predict(X, y, X_scenario, tree, expected_context):

    X_map = {
        "valid": X[SELECTED],
        "not_df": "X",
        "renamed": X[SELECTED].rename(columns={"2. Возраст": "2. Age"}),
    }

    X_predict_proba = X_map[X_scenario]

    with expected_context:
        tree.fit(X[SELECTED], y)
        _ = tree.predict(X_predict_proba)


@pytest.mark.parametrize(
    ("X_scenario", "y_scenario", "expected_context"),
    [
        ("valid", "valid", does_not_raise()),
        ("not_df", "valid", pytest.raises(ValueError, match="X must be a pandas.DataFrame.")),
        ("valid", "not_series", pytest.raises(ValueError, match="y must be a pandas.Series.")),
        ("valid", "short", pytest.raises(ValueError, match="X and y must be the equal length.")),
        (
            "renamed",
            "valid",
            pytest.raises(
                ValueError,
                match=(
                    "Feature names unseen at fit time:\n"
                    "- '2. Age'\n"
                    "Feature names seen at fit time, yet now missing:\n"
                    "- '2. Возраст'"
                ),
            ),
        ),
    ],
    ids=["valid", "not_df", "not_series", "short", "renamed"],
)
def test__check_data__score(X, y, X_scenario, y_scenario, tree, expected_context):

    X_map = {
        "valid": X[SELECTED],
        "not_df": "X",
        "renamed": X[SELECTED].rename(columns={"2. Возраст": "2. Age"}),
    }
    y_map = {
        "valid": y,
        "not_series": "y",
        "short": y[:-1],
    }

    X_score = X_map[X_scenario]
    y_score = y_map[y_scenario]

    with expected_context:
        tree.fit(X[SELECTED], y)
        _ = tree.score(X_score, y_score)
