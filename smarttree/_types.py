from enum import Enum
from typing import Literal


ClassificationCriterionType = Literal["gini", "entropy", "log_loss"]

CommonNaModeType = Literal["include_all", "include_best"]
NumNaModeType = Literal["min", "max", "include_all", "include_best"]
CatNaModeType = Literal["as_category", "include_all", "include_best"]
NaModeType = Literal["min", "max", "as_category", "include_all", "include_best"]

VerboseType = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | int

SplitType = Literal["numerical", "categorical", "rank"]


class Criterion(Enum):
    GINI = 1
    ENTROPY = 2
    LOG_LOSS = 2
