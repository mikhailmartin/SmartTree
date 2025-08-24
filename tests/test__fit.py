from contextlib import nullcontext as does_not_raise
import os
import re

import pandas as pd
import pytest
from pytest import param, raises

from smarttree import SmartDecisionTreeClassifier

data_ = pd.read_parquet(os.path.join("tests", "test_dataset.parquet"))
X_ = data_[["2. Возраст", "3. Семейное положение", "5. В какой семье Вы выросли?"]]
y_ = data_["Метка"]


@pytest.mark.parametrize(
    ("X_", "y_", "expected"),
    [
        param(X_, y_, does_not_raise()),
        param("X", y_, raises(ValueError, match="X must be a pandas.DataFrame.")),
        param(X_, "y", raises(ValueError, match="y must be a pandas.Series.")),
        param(X_, y_[:-1], raises(ValueError, match="X and y must be the equal length.")),
        param(
            X_.drop(columns="2. Возраст"), y_,
            raises(
                ValueError,
                match=(
                    "`numerical_feature_names` contain feature 2. Возраст,"
                    " which isnt present in the training data."
                ),
            ),
        ),
        param(
            X_.drop(columns="3. Семейное положение"), y_,
            raises(
                ValueError,
                match=(
                    "`categorical_feature_names` contain feature 3. Семейное положение,"
                    " which isnt present in the training data."
                ),
            ),
        ),
        param(
            X_.drop(columns="5. В какой семье Вы выросли?"), y_,
            raises(
                ValueError,
                match=re.escape(
                    "`rank_feature_names` contain feature 5. В какой семье Вы выросли?,"
                    " which isnt present in the training data."
                ),
            ),
        ),
    ],
)
def test__check_fit_data(X_, y_, expected):
    with expected:
        tree = SmartDecisionTreeClassifier(
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
        tree.fit(X_, y_)


def test__fit(X, y):
    tree = SmartDecisionTreeClassifier()
    tree.fit(X, y)

    assert tree.feature_names == [
        "2. Возраст", "3. Семейное положение", "5. В какой семье Вы выросли?"
    ]
    assert tree.class_names == [
        "доброкачественная опухоль", "злокачественная опухоль", "норма"
    ]
