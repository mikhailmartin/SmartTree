"""Custom realization of Decision Tree which can handle categorical features."""
import logging
import math
from abc import abstractmethod
from typing import Self

import numpy as np
import pandas as pd
from graphviz import Digraph
from sklearn.metrics import accuracy_score

from smarttree._builder import Builder
from smarttree._constants import (
    CategoricalNanModeOption,
    ClassificationCriterionOption,
    NumericalNanModeOption,
    VerboseOption,
)
from smarttree._exceptions import NotFittedError
from smarttree._node_splitter import NodeSplitter
from smarttree._renderer import Renderer
from smarttree._tree_node import TreeNode


class BaseSmartDecisionTree:
    """Base class for smart decision trees."""

    @abstractmethod
    def __init__(
        self,
        *,
        criterion: ClassificationCriterionOption = "gini",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = .0,
        max_childs: int | None = None,
        numerical_feature_names: list[str] | str | None = None,
        categorical_feature_names: list[str] | str | None = None,
        rank_feature_names: dict[str, list] | None = None,
        hierarchy: dict[str, str | list[str]] | None = None,
        numerical_nan_mode: NumericalNanModeOption = "min",
        categorical_nan_mode: CategoricalNanModeOption = "as_category",
        categorical_nan_filler: str = "missing_value",
        verbose: VerboseOption = "WARNING",
    ) -> None:

        self.logger = logging.getLogger()
        self.logger.setLevel(verbose)

        # criteria for limiting branching
        self.__criterion = criterion
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__max_leaf_nodes = max_leaf_nodes
        self.__min_impurity_decrease = min_impurity_decrease
        self.__max_childs = max_childs
        self.__hierarchy = hierarchy

        self.__numerical_feature_names = numerical_feature_names
        self.__categorical_feature_names = categorical_feature_names
        self.__rank_feature_names = rank_feature_names
        self._feature_names: list[str] = []
        self.__numerical_nan_mode = numerical_nan_mode
        self.__categorical_nan_mode = categorical_nan_mode
        self.__categorical_nan_filler = categorical_nan_filler

        self._is_fitted: bool = False
        self._root: TreeNode | None = None
        self._feature_importances: dict = dict()
        self._fill_numerical_nan_values: dict = dict()

        # check
        self.__check__criterion()
        self.__check__max_depth()
        self.__check__min_samples_split()
        self.__check__min_samples_leaf()
        self.__check__max_leaf_nodes()
        self.__check__min_impurity_decrease()
        self.__check__max_childs()
        self.__check__numerical_feature_names()
        self.__check__categorical_feature_names()
        self.__check__rank_feature_names()
        self.__check__hierarchy()
        self.__check__numerical_nan_mode()
        self.__check__categorical_nan_mode()
        self.__check__categorical_nan_filler()

        # mutate
        if self.__numerical_feature_names is None:
            self.__numerical_feature_names = []
            self.logger.debug(
                f"[{self.__class__.__name__}] [Debug] `numerical_feature_names`"
                f" is set to {self.__numerical_feature_names}."
            )
        elif isinstance(numerical_feature_names, str):
            self.__numerical_feature_names = [self.__numerical_feature_names]
            self.logger.debug(
                f"[{self.__class__.__name__}] [Debug] `numerical_feature_names`"
                f" is set to {self.__numerical_feature_names}."
            )

        if self.__categorical_feature_names is None:
            self.__categorical_feature_names = []
            self.logger.debug(
                f"[{self.__class__.__name__}] [Debug] `categorical_feature_names`"
                f" is set to {self.__categorical_feature_names}."
            )
        elif isinstance(self.__categorical_feature_names, str):
            self.__categorical_feature_names = [self.__categorical_feature_names]
            self.logger.debug(
                f"[{self.__class__.__name__}] [Debug] `categorical_feature_names`"
                f" is set to {self.__categorical_feature_names}."
            )

        if self.__rank_feature_names is None:
            self.__rank_feature_names = dict()
            self.logger.debug(
                f"[{self.__class__.__name__}] [Debug] `rank_feature_names`"
                f" is set to {self.__rank_feature_names}."
            )

        if self.__hierarchy is None:
            self.__hierarchy = dict()
            self.logger.debug(
                f"[{self.__class__.__name__}] [Debug] `hierarchy`"
                f" is set to {self.__hierarchy}."
            )

    def __check__criterion(self) -> None:
        if self.__criterion not in ("entropy", "gini", "log_loss"):
            raise ValueError(
                "`criterion` mist be Literal['entropy', 'log_loss', 'gini']."
                f" The current value of `criterion` is {self.__criterion!r}."
            )

    def __check__max_depth(self) -> None:
        if (
            self.__max_depth is not None
            and (not isinstance(self.__max_depth, int) or self.__max_depth <= 0)
        ):
            raise ValueError(
                "`max_depth` must be an integer and strictly greater than 0."
                f" The current value of `max_depth` is {self.__max_depth!r}."
            )

    def __check__min_samples_split(self) -> None:
        if (
            not isinstance(self.__min_samples_split, (int, float))
            or (
                isinstance(self.__min_samples_split, int)
                and self.__min_samples_split < 2
            )
            or (
                isinstance(self.__min_samples_split, float)
                and (self.__min_samples_split <= 0 or self.__min_samples_split >= 1)
            )
        ):
            raise ValueError(
                "`min_samples_split` must be an integer and lie in the range"
                " [2, +inf), or float and lie in the range (0, 1)."
                f" The current value of `min_samples_split` is"
                f" {self.__min_samples_split!r}."
            )

    def __check__min_samples_leaf(self) -> None:
        if (
            not isinstance(self.__min_samples_leaf, (int, float))
            or (
                isinstance(self.__min_samples_leaf, int)
                and self.__min_samples_leaf < 1
            )
            or (
                isinstance(self.__min_samples_leaf, float)
                and (self.__min_samples_leaf <= 0 or self.__min_samples_leaf >= 1)
            )
        ):
            raise ValueError(
                "`min_samples_leaf` must be an integer and lie in the range"
                " [1, +inf), or float and lie in the range (0, 1)."
                f" The current value of `min_samples_leaf` is"
                f" {self.__min_samples_leaf!r}."
            )

    def __check__max_leaf_nodes(self) -> None:
        if (
            self.__max_leaf_nodes is not None
            and (
                not isinstance(self.__max_leaf_nodes, int) or self.__max_leaf_nodes < 2
            )
        ):
            raise ValueError(
                "`max_leaf_nodes` must be an integer and strictly greater than 2."
                f" The current value of `max_leaf_nodes` is {self.__max_leaf_nodes!r}."
            )

    def __check__min_impurity_decrease(self) -> None:
        # TODO: could impurity_decrease be greater 1?
        if (
            not isinstance(self.__min_impurity_decrease, float)
            or self.__min_impurity_decrease < 0
        ):
            raise ValueError(
                "`min_impurity_decrease` must be float and non-negative."
                f" The current value of `min_impurity_decrease` is {self.__min_impurity_decrease!r}."
            )

    def __check__max_childs(self) -> None:
        if (
            self.__max_childs is not None
            and (
                not isinstance(self.__max_childs, int) or self.__max_childs < 2
            )
        ):
            raise ValueError(
                "`max_childs` must be integer and strictly greater than 2."
                f" The current value of `max_childs` is {self.__max_childs!r}."
            )

    def __check__numerical_feature_names(self) -> None:
        if isinstance(self.__numerical_feature_names, list):
            for numerical_feature_name in self.__numerical_feature_names:
                if not isinstance(numerical_feature_name, str):
                    raise ValueError(
                        "If `numerical_feature_names` is a list, it must consists of"
                        " strings."
                        f" The element {numerical_feature_name} of the list isnt a"
                        " string."
                    )
        elif not (
            isinstance(self.__numerical_feature_names, str)
            or self.__numerical_feature_names is None
        ):
            raise ValueError(
                "`numerical_feature_names` must be a string or list of strings."
                f" The current value of `numerical_feature_names` is"
                f" {self.__numerical_feature_names!r}."
            )

    def __check__categorical_feature_names(self) -> None:
        if isinstance(self.__categorical_feature_names, list):
            for categorical_feature_name in self.__categorical_feature_names:
                if not isinstance(categorical_feature_name, str):
                    raise ValueError(
                        "If `categorical_feature_names` is a list, it must consists of"
                        " strings."
                        f" The element {categorical_feature_name} of the list isnt a"
                        " string."
                    )
        elif not (
            isinstance(self.__categorical_feature_names, str)
            or self.__categorical_feature_names is None
        ):
            raise ValueError(
                "`categorical_feature_names` must be a string or list of strings."
                f" The current value of `categorical_feature_names` is"
                f" {self.__categorical_feature_names!r}."
            )

    def __check__rank_feature_names(self) -> None:
        if isinstance(self.__rank_feature_names, dict):
            for rank_feature_name, value_list in self.__rank_feature_names.items():
                if not isinstance(rank_feature_name, str):
                    raise ValueError(
                        "Keys in `rank_feature_names` must be a strings."
                        f" The key {rank_feature_name} isnt a string."
                    )
                if not isinstance(value_list, list):
                    raise ValueError(
                        "Values in `rank_feature_names` must be lists."
                        f" The value {value_list} of the key {rank_feature_name} isnt a"
                        " list."
                    )
        elif self.__rank_feature_names is not None:
            raise ValueError(
                "`rank_feature_names` must be a dictionary"
                " {rang feature name: list of its ordered values}."
            )

    def __check__hierarchy(self) -> None:
        common_message = (
            "`hierarchy` must be a dictionary"
            " {opening feature: opened feature / list of opened features}."
        )

        if isinstance(self.__hierarchy, dict):
            for key, value in self.__hierarchy.items():
                if not isinstance(key, str):
                    raise ValueError(
                        common_message
                        + f" Value {key!r} of opening feature isnt a string."
                    )
                if not isinstance(value, (str, list)):
                    raise ValueError(
                        common_message
                        + f" Value {value} of opened feature(s) isnt a string (list of"
                        " strings)."
                    )
                if isinstance(value, list):
                    for elem in value:
                        if not isinstance(elem, str):
                            raise ValueError(
                                common_message
                                + f" Value {elem} of opened feature isnt a string."
                            )
        elif self.__hierarchy is not None:
            raise ValueError(
                common_message
                + f" The current value of `hierarchy` is {self.__hierarchy!r}."
            )

    def __check__numerical_nan_mode(self) -> None:
        if self.__numerical_nan_mode not in ["include", "min", "max"]:
            raise ValueError(
                "`numerical_nan_mode` must be Literal['include', 'min', 'max']."
                f" The current value of `numerical_nan_mode` is {self.__numerical_nan_mode!r}."
            )

    def __check__categorical_nan_mode(self) -> None:
        if self.__categorical_nan_mode not in ["include", "as_category"]:
            raise ValueError(
                "`categorical_nan_mode` must be Literal['include', 'as_category']."
                f" The current value of `categorical_nan_mode` is {self.__categorical_nan_mode!r}."
            )

    def __check__categorical_nan_filler(self) -> None:
        if not isinstance(self.__categorical_nan_filler, str):
            raise ValueError(
                "`categorical_nan_filler` must be a string."
                f" The current value of `categorical_nan_filler` is {self.__categorical_nan_filler!r}."
            )

    @property
    def criterion(self) -> ClassificationCriterionOption:
        return self.__criterion

    @property
    def max_depth(self) -> int | None:
        return self.__max_depth

    @property
    def min_samples_split(self) -> int | float:
        return self.__min_samples_split

    @property
    def min_samples_leaf(self) -> int | float:
        return self.__min_samples_leaf

    @property
    def max_leaf_nodes(self) -> int | None:
        return self.__max_leaf_nodes

    @property
    def min_impurity_decrease(self) -> float:
        return self.__min_impurity_decrease

    @property
    def max_childs(self) -> int | None:
        return self.__max_childs

    @property
    def numerical_feature_names(self) -> list[str]:
        return self.__numerical_feature_names

    @property
    def categorical_feature_names(self) -> list[str]:
        return self.__categorical_feature_names

    @property
    def rank_feature_names(self) -> dict[str, list]:
        return self.__rank_feature_names

    @property
    def feature_names(self) -> list[str]:
        self._check_is_fitted()
        return self._feature_names

    @property
    def hierarchy(self) -> dict[str, str | list[str]]:
        return self.__hierarchy

    @property
    def numerical_nan_mode(self) -> NumericalNanModeOption:
        return self.__numerical_nan_mode

    @property
    def categorical_nan_mode(self) -> CategoricalNanModeOption:
        return self.__categorical_nan_mode

    @property
    def categorical_nan_filler(self) -> str:
        return self.__categorical_nan_filler

    @property
    def tree(self) -> TreeNode:
        self._check_is_fitted()
        return self._root

    @property
    def feature_importances_(self) -> dict[str, float]:
        self._check_is_fitted()
        return self._feature_importances

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet."
                " Call `fit` with appropriate arguments before using this estimator."
            )

    @abstractmethod
    def predict(self, X: pd.DataFrame | pd.Series) -> list[str] | str:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def score(self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> float:
        raise NotImplementedError

    def get_params(
        self,
        deep: bool = True,  # implemented for sklearn.model_selection.GridSearchCV
    ) -> dict:
        """Returns the parameters of this estimator instance."""
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_childs": self.max_childs,
            "numerical_feature_names": self.numerical_feature_names,
            "categorical_feature_names": self.categorical_feature_names,
            "rank_feature_names": self.rank_feature_names,
            "hierarchy": self.hierarchy,
            "numerical_nan_mode": self.numerical_nan_mode,
            "categorical_nan_mode": self.categorical_nan_mode,
            "categorical_nan_filler": self.categorical_nan_filler,
        }

    def set_params(self, **params) -> Self:
        """Set the parameters of this estimator instance."""
        # Simple optimization to gain speed (inspect is slow)
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        valid_params = ", ".join(valid_params.keys())

        for param, value in params.items():
            if param not in valid_params:
                raise ValueError(
                    f"Invalid parameter `{param}` for estimator {self.__class__.__name__}."
                    f" Valid parameters are: {valid_params}."
                )
            setattr(self, f"_{self.__class__.__bases__[0].__name__}__{param}", value)

        return self

    @abstractmethod
    def render(
        self,
        *,
        rounded: bool = False,
        show_impurity: bool = False,
        show_num_samples: bool = False,
        show_distribution: bool = False,
        show_label: bool = False,
        **kwargs,
    ):
        raise NotImplementedError


class SmartDecisionTreeClassifier(BaseSmartDecisionTree):
    """
    A decision tree classifier.

    Parameters:
        criterion: {'gini', 'entropy', 'log_loss'}, default='gini'
          The function to measure the quality of a split. Supported criteria are
           'gini' for the Gini impurity and 'log_loss' and 'entropy' both for
           the Shannon information gain.

        max_depth: int, default=None
          The maximum depth of the tree. If None, then nodes are expanded until
          all leaves are pure or until all leaves contain less than
          `min_samples_split` samples.

        min_samples_split: int or float, default=2
          The minimum number of samples required to split an internal node:

          - If int, then consider `min_samples_split` as the minimum number.
          - If float, then `min_samples_split` is a fraction and
            `ceil(min_samples_split * n_samples)` are the minimum number of
            samples for each split.

        min_samples_leaf: int or float, default=1
          The minimum number of samples required to be a leaf node.
          A split point at eny depth will only be considered if it leaves at
          least `min_samples_leaf` training samples in each of the left and
          right branches. This may have the effect of smoothing the model,
          especially in regression.

          - If int, then consider `min_samples_leaf` as the minimum number.
          - If float, then `min_samples_leaf` is a fraction and
            `ceil(min_samples_leaf * n_samples)` are the minimum number of
            samples for each node.

        max_leaf_nodes: int, default=None
          Grow a tree with `max_leaf_nodes` in best-first fashion. Best nodes
          are defined as relative reduction in impurity. If None then unlimited
          number of leaf nodes.

        min_impurity_decrease: float, default=0.0
          A node wil be split if this split induces a decrease of the impurity
          greater than or equal to this value.

          ...

        max_childs: int, default=None
          When choosing a categorical split, `max_childs` limits the maximum
          number of child nodes. If None then unlimited number of child nodes.

        numerical_feature_names: list[str] or str, default=None
          List of numerical feature names. If None `numerical_feature_names`
          will be set from unset feature names in X while .fit().

        categorical_feature_names: list[str] or str, default=None
          List of categorical feature names. If None `categorical_feature_names`
          will be set from unset feature names in X while .fit().

        rank_feature_names: list[str] or str, default=None
          List of rank feature names.

        hierarchy: dict, default=None
          ...

        numerical_nan_mode: {'include', 'min', 'max'}, default='include'
          The mode of handling missing values in a numerical feature.

          - If 'include': While training samples with missing values are
            included into all child nodes. While predicting decision is weighted
            mean of all decisions in child nodes.
          - If 'min', missing values are filled with minimum value of
            a numerical feature in training data.
          - If 'max', missing values are filled with maximum value of
            a numerical feature in training data.

        categorical_nan_mode: {'include', 'as_category'}, default='include'
          The mode of handling missing values in a categorical feature.

          - If 'include': While training samples with missing values are
            included into all child nodes. While predicting decision is
            weighted mean of all decisions in child nodes.
          - If 'as_category': While training and predicting missing values
            will be filled with `categorical_nan_filler`.

        categorical_nan_filler: str, default='missing_value'
          If `categorical_nan_mode` is set to "as_category", then during
          training and predicting missing values will be filled with
          `categorical_nan_filler`.

        verbose: {'critical', 'error', 'warning', 'info', 'debug'} or int, default="warning"
          Controls the level of decision tree verbosity.

          - If 'critical'
          - If 'error'
          - If 'warning'
          - If 'info'
          - If 'debug'
    """

    def __init__(
        self,
        *,
        criterion: ClassificationCriterionOption = "gini",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = .0,
        max_childs: int | None = None,
        numerical_feature_names: list[str] | str | None = None,
        categorical_feature_names: list[str] | str | None = None,
        rank_feature_names: dict[str, list] | None = None,
        hierarchy: dict[str, str | list[str]] | None = None,
        numerical_nan_mode: NumericalNanModeOption = "min",
        categorical_nan_mode: CategoricalNanModeOption = "include",
        categorical_nan_filler: str = "missing_value",
        verbose: VerboseOption = "WARNING",
    ) -> None:

        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            max_childs=max_childs,
            numerical_feature_names=numerical_feature_names,
            categorical_feature_names=categorical_feature_names,
            rank_feature_names=rank_feature_names,
            hierarchy=hierarchy,
            numerical_nan_mode=numerical_nan_mode,
            categorical_nan_mode=categorical_nan_mode,
            categorical_nan_filler=categorical_nan_filler,
            verbose=verbose,
        )

        # attributes that are open for reading
        self.__graph = None
        self.__classes: list[str] = []

    def __check_init_params(self) -> None:
        # TODO: finish this part
        # if (
        #     (isinstance(min_samples_split, int) and isinstance(min_samples_leaf, int))
        #     and min_samples_split < 2 * min_samples_leaf
        # ):
        #     raise ValueError(
        #         '`min_samples_split` должен быть строго в 2 раза больше'
        #         ' `min_samples_leaf`. Текущее значение `min_samples_split` ='
        #         f' {min_samples_split}, `min_samples_leaf` = {min_samples_leaf}.'
        #     )
        ...

    @property
    def classes_(self) -> list[str]:  # TODO: -> np.array
        self._check_is_fitted()
        return self.__classes

    def __repr__(self) -> str:
        repr_ = []

        # if a parameter value differs from default, then it added to the representation
        if self.criterion != "gini":
            repr_.append(f"criterion={self.criterion!r}")
        if self.max_depth:
            repr_.append(f"max_depth={self.max_depth}")
        if self.min_samples_split != 2:
            repr_.append(f"min_samples_split={self.min_samples_split}")
        if self.min_samples_leaf != 1:
            repr_.append(f"min_samples_leaf={self.min_samples_leaf}")
        if self.max_leaf_nodes:
            repr_.append(f"max_leaf_nodes={self.max_leaf_nodes}")
        if self.min_impurity_decrease != .0:
            repr_.append(f"min_impurity_decrease={self.min_impurity_decrease}")
        if self.max_childs:
            repr_.append(f"max_childs={self.max_childs}")
        if self.numerical_feature_names:
            repr_.append(f"numerical_feature_names={self.numerical_feature_names}")
        if self.categorical_feature_names:
            repr_.append(f"categorical_feature_names={self.categorical_feature_names}")
        if self.rank_feature_names:
            repr_.append(f"rank_feature_names={self.rank_feature_names}")
        if self.hierarchy:
            repr_.append(f"hierarchy={self.hierarchy}")
        if self.numerical_nan_mode != "min":
            repr_.append(f"numerical_nan_mode={self.numerical_nan_mode!r}")
        if self.categorical_nan_mode != "include":
            repr_.append(f"categorical_nan_mode={self.categorical_nan_mode!r}")
        if self.categorical_nan_filler != "missing_value":
            repr_.append(f"categorical_nan_filler={self.categorical_nan_filler!r}")

        return (
            f"{self.__class__.__name__}({', '.join(repr_)})"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters:
            X: The training input samples.
            y: The target values.
        """
        self.__check_fit_data(X, y)

        ################################################################################
        if self.max_depth is None:
            max_depth = float("+inf")
        else:
            max_depth = self.max_depth

        if isinstance(self.min_samples_split, float):
            min_samples_split = math.ceil(self.min_samples_split * X.shape[0])
        else:
            min_samples_split = self.min_samples_split

        if isinstance(self.min_samples_leaf, float):
            min_samples_leaf = math.ceil(self.min_samples_leaf * X.shape[0])
        else:
            min_samples_leaf = self.min_samples_leaf

        if self.max_leaf_nodes is None:
            max_leaf_nodes = float("+inf")
        else:
            max_leaf_nodes = self.max_leaf_nodes

        if self.max_childs is None:
            max_childs = float("+inf")
        else:
            max_childs = self.max_childs

        unsetted_features_set = set(X.columns) - (
            set(self.numerical_feature_names)
            | set(self.categorical_feature_names)
            | set(self.rank_feature_names)
        )

        if unsetted_features_set:
            unsetted_num_features = (
                X[list(unsetted_features_set)].select_dtypes("number").columns.tolist()
            )
            if unsetted_num_features:
                numerical_feature_names = self.numerical_feature_names + unsetted_num_features
                self.logger.info(
                    f"[{self.__class__.__name__}] [Info] {unsetted_num_features} are"
                    " added to `numerical_feature_names`."
                )
            else:
                numerical_feature_names = self.numerical_feature_names
            unsetted_cat_features = (
                X[list(unsetted_features_set)]
                .select_dtypes(include=["category", "object"]).columns.tolist()
            )
            if unsetted_cat_features:
                categorical_feature_names = self.categorical_feature_names + unsetted_cat_features
                self.logger.info(
                    f"[{self.__class__.__name__}] [Info] {unsetted_cat_features} are"
                    " added to `categorical_feature_names`."
                )
            else:
                categorical_feature_names = self.categorical_feature_names
        else:
            numerical_feature_names = self.numerical_feature_names
            categorical_feature_names = self.categorical_feature_names
        ################################################################################

        splitter = NodeSplitter(
            X=X,
            y=y,
            criterion=self.criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            max_childs=max_childs,
            numerical_feature_names=numerical_feature_names,
            categorical_feature_names=categorical_feature_names,
            rank_feature_names=self.rank_feature_names,
            numerical_nan_mode=self.numerical_nan_mode,
            categorical_nan_mode=self.categorical_nan_mode,
        )

        self._feature_names = X.columns.tolist()
        self.__classes = sorted(y.unique())

        match self.numerical_nan_mode:
            case "min":
                for num_feature in numerical_feature_names:
                    fill_nan_value = X[num_feature].min()
                    self._fill_numerical_nan_values[num_feature] = fill_nan_value
                    X[num_feature].fillna(fill_nan_value, inplace=True)
            case "max":
                for num_feature_name in numerical_feature_names:
                    fill_nan_value = X[num_feature_name].max()
                    self._fill_numerical_nan_values[fill_nan_value] = fill_nan_value
                    X[num_feature_name].fillna(fill_nan_value, inplace=True)

        if self.categorical_nan_mode == "as_category":
            for cat_feature in categorical_feature_names:
                X[cat_feature].fillna(self.categorical_nan_filler, inplace=True)

        builder = Builder(
            X=X,
            y=y,
            criterion=self.criterion,
            splitter=splitter,
            max_leaf_nodes=max_leaf_nodes,
            hierarchy=self.hierarchy,
        )
        root, feature_importances = builder.build()

        self._root = root
        self._feature_importances = feature_importances

        self._is_fitted = True

    def __check_fit_data(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas.DataFrame.")

        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas.Series.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must be the equal length.")

        for num_feature_name in self.numerical_feature_names:
            if num_feature_name not in X.columns:
                raise ValueError(
                    f"`numerical_feature_names` contain feature {num_feature_name},"
                    " which isnt present in the training data."
                )

        for cat_feature_name in self.categorical_feature_names:
            if cat_feature_name not in X.columns:
                raise ValueError(
                    f"`categorical_feature_names` contain feature {cat_feature_name},"
                    " which isnt present in the training data."
                )

        for rank_feature_name in self.rank_feature_names.keys():
            if rank_feature_name not in X.columns:
                raise ValueError(
                    f"`rank_feature_names` contain feature {rank_feature_name},"
                    " which isnt present in the training data."
                )

    def predict(self, X: pd.DataFrame | pd.Series) -> list[str] | str:
        """
        Predict class for samples in X.

        Parameters:
            X: The input samples.

        Returns:
            The predicted classes.
        """
        self._check_is_fitted()

        y_pred_proba_s = self.predict_proba(X)
        y_pred = [
            self.__classes[y_pred_proba.argmax()] for y_pred_proba in y_pred_proba_s
        ]

        return y_pred

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """
        Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same class in
        a leaf.

        Parameters:
            X: The input samples.

        Returns:
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`class_names`.
        """
        self._check_is_fitted()

        X = self.__preprocess(X)

        y_pred_proba = np.array([
            self.__predict_proba(self._root, point)[0]
            for _, point in X.iterrows()
        ])

        return y_pred_proba

    def __preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses data for prediction.

        Fills in the missing values with the corresponding values according to
        the `categorical_nan_mode` and `numerical_nan_mode`.
        """
        X = X.copy()

        if self.numerical_nan_mode in ["min", "max"]:
            for num_feature in self.numerical_feature_names:
                X.fillna(self._fill_numerical_nan_values[num_feature], inplace=True)

        if self.categorical_nan_mode == "as_category":
            for cat_feature in self.categorical_feature_names:
                X[cat_feature].fillna(self.categorical_nan_filler, inplace=True)

        return X

    def __predict_proba(
        self,
        node: TreeNode,
        point: pd.Series,
    ) -> tuple[np.ndarray, int]:
        """Predicts class for the sample."""
        # if we haven't reached a leaf
        if not node.is_leaf:
            # but we in a node in which the split rule is set according to some feature,
            # and the sample in this feature contains a missing value
            if pd.isna(point[node.split_feature_name]):
                # then we go to the child nodes for predictions, and then we weighted
                # average them.
                distribution_parent = np.array([0., 0., 0.])
                samples_parent = 0
                for child in node.childs:
                    (
                        y_pred_proba_child,
                        samples_child,
                    ) = self.__predict_proba(child, point)
                    distribution_child = y_pred_proba_child * samples_child
                    distribution_parent += distribution_child
                    samples_parent += samples_child
                y_pred_proba = distribution_parent / distribution_parent.sum()
                samples = samples_parent

            elif node.split_feature_name in self.numerical_feature_names:
                # looking for the branch that needs to be followed
                threshold = float(node.childs[0].feature_value[0][3:])
                if point[node.split_feature_name] <= threshold:
                    y_pred_proba, samples = self.__predict_proba(node.childs[0], point)
                elif point[node.split_feature_name] > threshold:
                    y_pred_proba, samples = self.__predict_proba(node.childs[1], point)
                else:
                    assert False

            elif (
                node.split_feature_name in self.categorical_feature_names
                or node.split_feature_name in self.rank_feature_names
            ):
                # looking for the branch that needs to be followed
                for child in node.childs:
                    # if found
                    if child.feature_value == point[node.split_feature_name]:
                        y_pred_proba, samples = self.__predict_proba(child, point)
                        # then we can finish the search
                        break
                else:
                    # if there is no such branch TODO
                    distribution = np.array(node.distribution)
                    y_pred_proba = distribution / distribution.sum()
                    samples = node.samples

            else:
                assert False

        # if we have reached a leaf
        else:
            distribution = np.array(node.distribution)
            y_pred_proba = distribution / distribution.sum()
            samples = node.samples

        return y_pred_proba, samples

    def __check_score_data(self, X, y, sample_weight):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas.DataFrame.")

        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas.Series.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must be the equal length.")

        fitted_features_set = set(self._feature_names)
        X_features_set = set(X.columns)
        if fitted_features_set != X_features_set:
            message = [
                "The feature names should match those that were passed during fit."
            ]

            unexpected_names = sorted(X_features_set - fitted_features_set)
            missing_names = sorted(fitted_features_set - X_features_set)

            def add_names(names: list[str]) -> str:
                output = []
                max_n_names = 5
                for i, name in enumerate(names):
                    if i >= max_n_names:
                        output.append("- ...")
                        break
                    output.append(f"- {name}")
                return "\n".join(output)

            if unexpected_names:
                message.append("Feature names unseen at fit time:")
                message.append(add_names(unexpected_names))

            if missing_names:
                message.append("Feature names seen at fit time, yet now missing:")
                message.append(add_names(missing_names))

            # TODO: same order of features

            raise ValueError("\n".join(message))

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> float:
        """Returns the accuracy metric."""
        self._check_is_fitted()

        self.__check_score_data(X, y, sample_weight)

        score = accuracy_score(y, self.predict(X), sample_weight=sample_weight)

        return score

    def render(
        self,
        *,
        rounded: bool = False,
        show_impurity: bool = False,
        show_num_samples: bool = False,
        show_distribution: bool = False,
        show_label: bool = False,
        **kwargs,
    ) -> Digraph:
        """
        Visualizes the decision tree.

        If named parameters are set, saves the visualization as a file(s).

        Parameters:
            rounded: bool, default=False
              Whether to round the corners of the nodes
              (they are in the shape of a rectangle).
            show_impurity: bool, default=False
              Whether to show the impurity of the node.
            show_num_samples: bool, default=False
              Whether to show the number of samples in the node.
            show_distribution: bool, default=False
              Whether to show the class distribution.
            show_label: bool, default=False
              Whether to show the class to which the node belongs.
            **kwargs: arguments for graphviz.Digraph.render.

        Returns:
            Digraph: class containing a description of the graph structure of
            the tree for visualization.
        """
        self._check_is_fitted()

        renderer = Renderer(criterion=self.criterion, rounded=rounded)
        graph = renderer.render(
            tree=self.tree,
            show_impurity=show_impurity,
            show_num_samples=show_num_samples,
            show_distribution=show_distribution,
            show_label=show_label,
        )

        return graph
