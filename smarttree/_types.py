from typing import Literal


ClassificationCriterionType = Literal["gini", "entropy", "log_loss"]
NumNaModeType = Literal["min", "max", "include_all", "include_best"]
CatNaModeType = Literal["as_category", "include_all", "include_best"]
NaModeType = Literal["min", "max", "as_category", "include_all", "include_best"]
VerboseType = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | int
SplitType = Literal["numerical", "categorical", "rank"]
