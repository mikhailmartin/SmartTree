import os
import pandas as pd
import pytest

TARGET_COL = "Метка"
# num_col = "2. Возраст"
# cat_col = "3. Семейное положение"
# rank_col = "5. В какой семье Вы выросли?"


@pytest.fixture(scope="session")
def data() -> pd.DataFrame:
    path = os.path.join("tests", "test_dataset.parquet")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def X(data) -> pd.DataFrame:
    return data.drop(columns=TARGET_COL)


@pytest.fixture(scope="session")
def y(data) -> pd.Series:
    return data[TARGET_COL]
