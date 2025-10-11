from typing import Literal


CriterionType = Literal["gini", "entropy", "log_loss", "squared_error"]
ClassificationCriterionType = Literal["gini", "entropy", "log_loss"]
RegressionCriterionType = Literal["squared_error"]

CommonNaModeType = Literal["include_all", "include_best"]
NumNaModeType = Literal["min", "max", "include_all", "include_best"]
CatNaModeType = Literal["as_category", "include_all", "include_best"]
NaModeType = Literal["min", "max", "as_category", "include_all", "include_best"]

VerboseType = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | int

SplitType = Literal["numerical", "categorical", "rank"]
