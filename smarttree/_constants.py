from typing import Literal

ClassificationCriterionOption = Literal["gini", "entropy", "log_loss"]
NumericalNanModeOption = Literal["include", "min", "max"]
CategoricalNanModeOption = Literal["include", "as_category"]
VerboseOption = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] | int
