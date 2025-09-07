from typing import Literal


ClassificationCriterionType = Literal["gini", "entropy", "log_loss"]
NumericalNanModeType = Literal["include", "min", "max"]
CategoricalNanModeType = Literal["as_category", "include_all", "include_best"]
VerboseType = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | int
SplitTypeType = Literal["numerical", "categorical", "rank"]
