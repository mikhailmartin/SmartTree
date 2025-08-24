import os
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def data() -> pd.DataFrame:
    path = os.path.join("tests", "test_dataset.csv")
    return pd.read_csv(path, index_col=0)


@pytest.fixture(scope="session")
def X(data) -> pd.DataFrame:
    num_col = "2. Возраст"
    cat_col = "3. Семейное положение"
    rank_col = "5. В какой семье Вы выросли?"
    return data[[num_col, cat_col, rank_col]]


@pytest.fixture(scope="session")
def y(data) -> pd.Series:
    target_col = "Метка"
    return data[target_col]
