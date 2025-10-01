from typing import Literal


ClassificationCriterionType = Literal["gini", "entropy", "log_loss"]

CommonNaModeType = Literal["include_all", "include_best"]
NumNaModeType = Literal["min", "max", "include_all", "include_best"]
CatNaModeType = Literal["as_category", "include_all", "include_best"]
NaModeType = Literal["min", "max", "as_category", "include_all", "include_best"]

RegressionCriterionType = Literal["squared_error"]

VerboseType = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | int

SplitType = Literal["numerical", "categorical", "rank"]
