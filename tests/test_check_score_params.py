from contextlib import nullcontext as does_not_raise
import os
import sys
sys.path.append(sys.path[0] + "/../")

import pandas as pd
import pytest
from pytest import param, raises

from smarttree import SmartTreeClassifier


data = pd.read_csv(os.path.join("tests", "test_dataset.csv"), index_col=0)
X = data[["2. Возраст", "3. Семейное положение", "5. В какой семье Вы выросли?"]]
y = data["Метка"]


@pytest.mark.parametrize(
    ("X", "y", "expected"),
    [
        param(X, y, does_not_raise()),
        param(
            "X", y,
            raises(ValueError, match="X must be a pandas.DataFrame."),
        ),
        param(
            X, "y",
            raises(
                ValueError,
                match="y must be a pandas.Series.",
            ),
        ),
        param(
            X, y[:-1],
            raises(ValueError, match="X and y must be the equal length."),
        ),
        param(
            X.rename(columns={"2. Возраст": "2. Age"}), y,
            raises(
                ValueError,
                match=(
                    "Feature names unseen at fit time:\n"
                    "- 2. Age\n"
                    "Feature names seen at fit time, yet now missing:\n"
                    "- 2. Возраст\n"
                ),
            ),
        ),
    ],
)
def test_check_score_params(X, y, expected):
    with expected:
        data = pd.read_csv(os.path.join("tests", "test_dataset.csv"), index_col=0)
        X_fit = data[["2. Возраст", "3. Семейное положение", "5. В какой семье Вы выросли?"]]
        y_fit = data["Метка"]

        msdt = SmartTreeClassifier(
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
        msdt.fit(X_fit, y_fit)
        msdt.score(X, y)
