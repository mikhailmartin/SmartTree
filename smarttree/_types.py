from typing import Literal


ClassificationCriterionType = Literal["gini", "entropy", "log_loss"]
NumericalNaModeType = Literal["include", "min", "max"]
CategoricalNaModeType = Literal["as_category", "include_all", "include_best"]
VerboseType = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | int
SplitTypeType = Literal["numerical", "categorical", "rank"]
