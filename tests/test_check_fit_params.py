from contextlib import nullcontext as does_not_raise
import os
import re
import sys
sys.path.append(sys.path[0] + '/../')

import pandas as pd
import pytest
from pytest import param, raises

from multi_split_decision_tree import MultiSplitDecisionTreeClassifier

data = pd.read_csv(os.path.join('tests', 'test_dataset.csv'), index_col=0)
X = data[['2. Возраст', '3. Семейное положение', '5. В какой семье Вы выросли?']]
y = data['Метка']


@pytest.mark.parametrize(
    ('X', 'y', 'expected'),
    [
        param(
            X, y,
            does_not_raise(),
        ),
        param(
            'X', y,
            raises(ValueError, match='X должен представлять собой pd.DataFrame.'),
        ),
        param(
            X, 'y',
            raises(ValueError, match='y должен представлять собой pd.Series.')
        ),
        param(
            X, y[:-1],
            raises(ValueError, match='X и y должны быть одной длины.'),
        ),
        param(
            X.drop(columns='2. Возраст'), y,
            raises(
                ValueError,
                match=(
                    '`numerical_feature_names` содержит признак 2. Возраст,'
                    ' которого нет в обучающих данных.'
                ),
            ),
        ),
        param(
            X.drop(columns='3. Семейное положение'), y,
            raises(
                ValueError,
                match=(
                    '`categorical_feature_names` содержит признак 3. Семейное положение,'
                    ' которого нет в обучающих данных.'
                ),
            ),
        ),
        param(
            X.drop(columns='5. В какой семье Вы выросли?'), y,
            raises(
                ValueError,
                match=re.escape(
                    '`rank_feature_names` содержит признак 5. В какой семье Вы выросли?,'
                    ' которого нет в обучающих данных.'
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
