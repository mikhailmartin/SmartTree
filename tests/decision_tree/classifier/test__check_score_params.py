import os
from contextlib import nullcontext as does_not_raise

import pandas as pd
import pytest
from pytest import raises

from smarttree import SmartDecisionTreeClassifier


data_ = pd.read_parquet(os.path.join("tests", "test_dataset.parquet"))
X_ = data_[["2. Возраст", "3. Семейное положение", "5. В какой семье Вы выросли?"]]
y_ = data_["Метка"]


@pytest.mark.parametrize(
    ("X_", "y_", "expected"),
    [
        (X_, y_, does_not_raise()),
        (
            "X", y_,
            raises(ValueError, match="X must be a pandas.DataFrame."),
        ),
        (
            X_, "y",
            raises(
                ValueError,
                match="y must be a pandas.Series.",
            ),
        ),
        (
            X_, y_[:-1],
            raises(ValueError, match="X and y must be the equal length."),
        ),
        (
            X_.rename(columns={"2. Возраст": "2. Age"}), y_,
            raises(
                ValueError,
                match=(
                    "Feature names unseen at fit time:\n"
                    "- 2. Age\n"
                    "Feature names seen at fit time, yet now missing:\n"
                    "- 2. Возраст"
                ),
            ),
        ),
    ],
)
def test_check_score_params(X_, y_, expected, X, y):
    with expected:
        X_fit = X[["2. Возраст", "3. Семейное положение", "5. В какой семье Вы выросли?"]]
        y_fit = y

        tree = SmartDecisionTreeClassifier(
            max_depth=1,
            numerical_feature_names=["2. Возраст"],
            categorical_feature_names=["3. Семейное положение"],
            rank_feature_names={
                "5. В какой семье Вы выросли?": [
                    "полная семья, кровные родители",
                    "мачеха/отчим",
                    "мать/отец одиночка",
                    "с бабушкой и дедушкой",
                    "в детском доме",
                ],
            },
        )
        tree.fit(X_fit, y_fit)
        tree.score(X_, y_)
