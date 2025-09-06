from typing import Literal


ClassificationCriterionOption = Literal["gini", "entropy", "log_loss"]
NumericalNanModeOption = Literal["include", "min", "max"]
CategoricalNanModeOption = Literal["as_category", "include_all", "include_best"]
VerboseOption = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | int
