from typing import Literal


ClassificationCriterionType = Literal["gini", "entropy", "log_loss"]
NumericalNaModeType = Literal["min", "max", "include_all"]
CategoricalNaModeType = Literal["as_category", "include_all", "include_best"]
VerboseType = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | int
SplitType = Literal["numerical", "categorical", "rank"]
