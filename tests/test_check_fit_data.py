from contextlib import nullcontext as does_not_raise
import os
import re
import sys
sys.path.append(sys.path[0] + '/../')

import pandas as pd
import pytest
from pytest import param, raises

from smarttree import MultiSplitDecisionTreeClassifier

data = pd.read_csv(os.path.join('tests', 'test_dataset.csv'), index_col=0)
X = data[['2. Возраст', '3. Семейное положение', '5. В какой семье Вы выросли?']]
y = data['Метка']


@pytest.mark.parametrize(
    ('X', 'y', 'expected'),
    [
        param(X, y, does_not_raise()),
        param('X', y, raises(ValueError, match='X must be a pandas.DataFrame.')),
        param(X, 'y', raises(ValueError, match='y must be a pandas.Series.')),
        param(X, y[:-1], raises(ValueError, match='X and y must be the equal length.')),
        param(
            X.drop(columns='2. Возраст'), y,
            raises(
                ValueError,
                match=(
                    '`numerical_feature_names` contain feature 2. Возраст,'
                    ' which isnt present in the training data.'
                ),
            ),
        ),
        param(
            X.drop(columns='3. Семейное положение'), y,
            raises(
                ValueError,
                match=(
                    '`categorical_feature_names` contain feature 3. Семейное положение,'
                    ' which isnt present in the training data.'
                ),
            ),
        ),
        param(
            X.drop(columns='5. В какой семье Вы выросли?'), y,
            raises(
                ValueError,
                match=re.escape(
                    '`rank_feature_names` contain feature 5. В какой семье Вы выросли?,'
                    ' which isnt present in the training data.'
                ),
            ),
        ),
    ],
)
def test_check_fit_params(X, y, expected):
    with expected:
        msdt = MultiSplitDecisionTreeClassifier(
            numerical_feature_names=['2. Возраст'],
            categorical_feature_names=['3. Семейное положение'],
            rank_feature_names={
                '5. В какой семье Вы выросли?': [
                    'полная семья, кровные родители',
                    'мачеха/отчим',
                    'мать/отец одиночка',
                    'с бабушкой и дедушкой',
                    'в детском доме',
                ],
            },
        )

        msdt.fit(X, y)
