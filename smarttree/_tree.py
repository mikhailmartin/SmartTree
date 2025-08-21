"""Custom realization of Decision Tree which can handle categorical features."""
from abc import abstractmethod
import bisect
import logging
import math
from typing import Literal

from graphviz import Digraph
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from smarttree._tree_node import TreeNode
from smarttree._utils import (
    cat_partitions, get_thresholds, rank_partitions
)
from smarttree._exceptions import NotFittedError


class BaseSmartDecisionTree:
    """Base class for smart decision trees."""

    @abstractmethod
    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_leaf_nodes: int | float = float("+inf"),
        max_childs: int | float = float("+inf"),
        numerical_feature_names: list[str] | str | None = None,
        categorical_feature_names: list[str] | str | None = None,
        rank_feature_names: dict[str, list] | None = None,
        hierarchy: dict[str, str | list[str]] | None = None,
        numerical_nan_mode: Literal["include", "min", "max"] = "min",
        categorical_nan_mode: Literal["include", "as_category"] = "include",
        categorical_nan_filler: str = "missing_value",
    ) -> None:

        # criteria for limiting branching
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__max_leaf_nodes = max_leaf_nodes
        self.__max_childs = max_childs
        self.__hierarchy = hierarchy

        self.__numerical_feature_names = numerical_feature_names
        self.__categorical_feature_names = categorical_feature_names
        self.__rank_feature_names = rank_feature_names
        self.__feature_names = None
        self.__numerical_nan_mode = numerical_nan_mode
        self.__categorical_nan_mode = categorical_nan_mode
        self.__categorical_nan_filler = categorical_nan_filler

        self.__is_fitted = False

        # check
        self.__check__max_depth()
        self.__check__min_samples_split()
        self.__check__min_samples_leaf()
        self.__check__max_leaf_nodes()
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
            logging.debug(
                f"[{self.__class__.__name__}] [Debug] `numerical_feature_names`"
                f" is set to {self.__numerical_feature_names}."
            )
        elif isinstance(numerical_feature_names, str):
            self.__numerical_feature_names = [self.__numerical_feature_names]
            logging.debug(
                f"[{self.__class__.__name__}] [Debug] `numerical_feature_names`"
                f" is set to {self.__numerical_feature_names}."
            )

        if self.__categorical_feature_names is None:
            self.__categorical_feature_names = []
            logging.debug(
                f"[{self.__class__.__name__}] [Debug] `categorical_feature_names`"
                f" is set to {self.__categorical_feature_names}."
            )
        elif isinstance(self.__categorical_feature_names, str):
            self.__categorical_feature_names = [self.__categorical_feature_names]
            logging.debug(
                f"[{self.__class__.__name__}] [Debug] `categorical_feature_names`"
                f" is set to {self.__categorical_feature_names}."
            )

        if self.__rank_feature_names is None:
            self.__rank_feature_names = dict()

        self.__hierarchy = hierarchy if hierarchy else {}

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
            not (
                isinstance(self.__max_leaf_nodes, int)
                or self.__max_leaf_nodes == float("+inf")
            )
            or self.__max_leaf_nodes < 2
        ):
            raise ValueError(
                "`max_leaf_nodes` must be an integer and strictly greater than 2."
                f" The current value of `max_leaf_nodes` is {self.__max_leaf_nodes!r}."
            )

    def __check__max_childs(self) -> None:
        if (
            not (
                isinstance(self.__max_childs, int)
                or self.__max_childs == float("+inf")
            )
            or self.__max_childs < 2
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
        if isinstance(self.__hierarchy, dict):
            for key, value in self.__hierarchy.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "`hierarchy` must be a dictionary"
                        " {opening feature: opened feature / list of opened strings}."
                        f" Value {key!r} of opening feature isnt a string."
                    )
                if not isinstance(value, (str, list)):
                    raise ValueError(
                        "`hierarchy` must be a dictionary"
                        " {opening feature: opened feature / list of opened features}."
                        f" Value {value} of opened feature(s) isnt a string (list of"
                        " strings)."
                    )
                if isinstance(value, list):
                    for elem in value:
                        if not isinstance(elem, str):
                            raise ValueError(
                                "`hierarchy` must be a dictionary {opening feature:"
                                " opened feature / list of opened features}."
                                f" Value {elem} of opened feature isnt a string."
                            )
        elif self.__hierarchy is not None:
            raise ValueError(
                "`hierarchy` must be a dictionary"
                " {opening feature: opened feature / list of opened strings}."
                f" The current value of `hierarchy` is {self.__hierarchy!r}."
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
    def max_depth(self) -> int | None:
        return self.__max_depth

    @property
    def min_samples_split(self) -> int | float:
        return self.__min_samples_split

    @property
    def min_samples_leaf(self) -> int | float:
        return self.__min_samples_leaf

    @property
    def max_leaf_nodes(self) -> int | float:
        return self.__max_leaf_nodes

    @property
    def max_childs(self) -> int | float:
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
        if not self.__is_fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet."
                " Call `fit` with appropriate arguments before using this estimator."
            )

        return self.__feature_names

    @property
    def hierarchy(self) -> dict[str, str | list[str]]:
        return self.__hierarchy

    @property
    def numerical_nan_mode(self) -> Literal["include", "min", "max"]:
        return self.__numerical_nan_mode

    @property
    def categorical_nan_mode(self) -> Literal["include", "as_category"]:
        return self.__categorical_nan_mode

    @property
    def categorical_nan_filler(self) -> str:
        return self.__categorical_nan_filler


class SmartDecisionTreeClassifier(BaseSmartDecisionTree):
    """
    A decision tree classifier.

    Parameters:
        criterion: {'gini', 'entropy', 'log_loss'}, default='gini'
            The function to measure the quality of a split. Supported criteria
            are 'gini' for the Gini impurity and 'log_loss' and 'entropy' both
            for the Shannon information gain.

        max_depth: int, default=None
            The maximum depth of the tree. If None, then nodes are expanded
            until all leaves are pure or until all leaves contain less than
            min_samples_split samples.

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

            TODO: formula

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

        numerical_nan_mode: Literal['include', 'min', 'max'], default='include'
            The mode of handling missing values in a numerical feature.

            - If 'include': While training samples with missing values are
              included into all child nodes. While predicting decision is
              weighted mean of all decisions in child nodes.
            - If 'min', missing values are filled with minimum value of
              a numerical feature in training data.
            - If 'max', missing values are filled with maximum value of
              a numerical feature in training data.

        categorical_nan_mode: Literal['include', 'as_category'], default='include'
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

        verbose: Literal['critical', 'error', 'warning', 'info', 'debug'] or int, default=2
            Controls the level of decision tree verbosity.

            - If 'critical'
            - If 'error'
            - If 'warning'
            - If 'info'
            - If 'debug'

    Attributes:
        tree: The underlying Tree object.
        class_names: The sorted list of class names.
        feature_names: The list of all features in train data.
        numerical_feature_names: The list of all numerical features in train data.
        categorical_feature_names: The list of all categorical features in train data.
        rank_feature_names: The list of all rank features in train data.
        feature_importances: The dict {feature name: feature importance}.
    """
    @staticmethod
    def __check_init_params(verbose):
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

        if (
            not isinstance(verbose, (str, int))
            or (
                isinstance(verbose, str)
                and verbose not in ["critical", "error", "warning", "info", "debug"]
            )
        ):
            raise ValueError(
                "`verbose` must be an integer or"
                " Literal['critical', 'error', 'warning', 'info', 'debug']."
                f" The current value of `verbose` is {verbose!r}."
            )

    def __init__(
        self,
        *,
        criterion: Literal["gini", "entropy", "log_loss"] = "gini",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_leaf_nodes: int | float = float("+inf"),
        min_impurity_decrease: float = .0,
        max_childs: int | float = float("+inf"),
        numerical_feature_names: list[str] | str | None = None,
        categorical_feature_names: list[str] | str | None = None,
        rank_feature_names: dict[str, list] | None = None,
        hierarchy: dict[str, str | list[str]] | None = None,
        numerical_nan_mode: Literal["include", "min", "max"] = "min",
        categorical_nan_mode: Literal["include", "as_category"] = "include",
        categorical_nan_filler: str = "missing_value",
        verbose: Literal["critical", "error", "warning", "info", "debug"] | int = 2,
    ) -> None:

        super().__init__(
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_leaf_nodes,
            max_childs,
            numerical_feature_names,
            categorical_feature_names,
            rank_feature_names,
            hierarchy,
            numerical_nan_mode,
            categorical_nan_mode,
            categorical_nan_filler,
        )

        self.__criterion = criterion
        self.__min_impurity_decrease = min_impurity_decrease

        self.__check__criterion()
        self.__check__min_impurity_decrease()

        self.__check_init_params(verbose)

        match verbose:
            case "critical":
                logging_level = logging.CRITICAL
            case "error":
                logging_level = logging.ERROR
            case "warning":
                logging_level = logging.WARNING
            case "info":
                logging_level = logging.INFO
            case "debug":
                logging_level = logging.DEBUG
            case _:
                if verbose < 0:
                    logging_level = logging.CRITICAL
                elif verbose == 0:
                    logging_level = logging.ERROR
                elif verbose == 1:
                    logging_level = logging.WARNING
                elif verbose == 2:
                    logging_level = logging.INFO
                elif verbose > 2:
                    logging_level = logging.DEBUG

        logging.basicConfig(level=logging_level)

        match self.criterion:
            case "gini":
                self.__impurity = self.__gini_index
            case "entropy" | "log_loss":
                self.__impurity = self.__entropy

        # attributes that are open for reading
        self.__root = None
        self.__graph = None
        self.__class_names = None
        self.__feature_names = None
        self.__feature_importances = {}

        self.__fill_numerical_nan_values = {}

        self.__is_fitted = False

        self.__node_counter = 0
        self.__leaf_counter = 0

    def __check__criterion(self) -> None:
        if self.__criterion not in ["entropy", "gini", "log_loss"]:
            raise ValueError(
                "`criterion` mist be Literal['entropy', 'log_loss', 'gini']."
                f" The current value of `criterion` is {self.__criterion!r}."
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

    @property
    def criterion(self) -> Literal["entropy", "log_loss", "gini"]:
        return self.__criterion

    @property
    def min_impurity_decrease(self) -> float:
        return self.__min_impurity_decrease

    def __repr__(self):
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
        if self.max_leaf_nodes != float("+inf"):
            repr_.append(f"max_leaf_nodes={self.max_leaf_nodes}")
        if self.min_impurity_decrease != .0:
            repr_.append(f"min_impurity_decrease={self.min_impurity_decrease}")
        if self.max_childs != float("+inf"):
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

    @property
    def tree(self) -> TreeNode:
        if not self.__is_fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet."
                " Call `fit` with appropriate arguments before using this estimator."
            )

        return self.__root

    @property
    def class_names(self) -> list[str]:
        if not self.__is_fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet."
                " Call `fit` with appropriate arguments before using this estimator."
            )

        return self.__class_names

    @property
    def feature_importances(self) -> dict[str, float]:
        if not self.__is_fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet."
                " Call `fit` with appropriate arguments before using this estimator."
            )

        return self.__feature_importances

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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters:
            X: The training input samples.
            y: The target values.
        """
        self.__check_fit_data(X, y)

        # until the end of the training, we encapsulate X and y
        self.X = X.copy()
        self.y = y.copy()
        # technical attribute
        self.splittable_leaf_nodes = []

        self.__feature_names = X.columns.tolist()
        self.__class_names = sorted(y.unique())

        if isinstance(self.min_samples_split, float):
            self.min_samples_split = math.ceil(self.min_samples_split * X.shape[0])
        if isinstance(self.min_samples_leaf, float):
            self.min_samples_leaf = math.ceil(self.min_samples_leaf * X.shape[0])

        # initialize feature_importances with all the features and the default value of 0
        for feature_name in self.__feature_names:
            self.__feature_importances[feature_name] = 0

        # numerical_feature_names and categorical_feature_names extensions ##############
        unsetted_features_set = set(self.X.columns) - (
            set(self.numerical_feature_names) |
            set(self.categorical_feature_names) |
            set(self.rank_feature_names)
        )

        if unsetted_features_set:
            unsetted_num_features = (
                self.X[list(unsetted_features_set)]
                .select_dtypes("number").columns.tolist()
            )
            if unsetted_num_features:
                self.numerical_feature_names.extend(unsetted_num_features)
                logging.info(
                    f"[MultiSplitDecisionTree] [Info] {unsetted_num_features} are added"
                    " to `numerical_feature_names`."
                )
            unsetted_cat_features = (
                self.X[list(unsetted_features_set)]
                .select_dtypes(include=["category", "object"]).columns.tolist()
            )
            if unsetted_cat_features:
                self.categorical_feature_names.extend(unsetted_cat_features)
                logging.info(
                    f"[MultiSplitDecisionTree] [Info] {unsetted_cat_features} are added"
                    " to `categorical_feature_names`."
                )
        ################################################################################

        match self.numerical_nan_mode:
            case "min":
                for num_feature in self.numerical_feature_names:
                    fill_nan_value = X[num_feature].min()
                    self.__fill_numerical_nan_values[num_feature] = fill_nan_value
                    X[num_feature].fillna(fill_nan_value, inplace=True)
            case "max":
                for num_feature_name in self.numerical_feature_names:
                    fill_nan_value = X[num_feature_name].max()
                    self.__fill_numerical_nan_values[fill_nan_value] = fill_nan_value
                    X[num_feature_name].fillna(fill_nan_value, inplace=True)

        if self == "as_category":
            for cat_feature in self.categorical_feature_names:
                X[cat_feature].fillna(self.categorical_nan_filler, inplace=True)

        hierarchy = self.hierarchy.copy()
        available_feature_names = X.columns.tolist()
        # remove those features that cannot be considered yet
        for value in hierarchy.values():
            if isinstance(value, str):
                available_feature_names.remove(value)
            elif isinstance(value, list):
                for feature_name in value:
                    available_feature_names.remove(feature_name)
            else:
                assert False

        root_mask = y.apply(lambda x: True)
        self.__root = self.__create_node(
            mask=root_mask,
            hierarchy=hierarchy,
            available_feature_names=available_feature_names,
            depth=0,
        )

        if self.__is_splittable(self.__root):
            self.splittable_leaf_nodes.append(self.__root)

        while (
            len(self.splittable_leaf_nodes) > 0
            and self.__leaf_counter < self.max_leaf_nodes
        ):
            best_node = self.splittable_leaf_nodes.pop()
            (
                inf_gain,
                split_type,
                split_feature_name,
                feature_values,
                child_masks,
            ) = best_node._best_split

            self.__feature_importances[split_feature_name] += inf_gain

            for child_mask, feature_value in zip(child_masks, feature_values):
                # add opened features
                if split_feature_name in best_node._hierarchy:
                    value = best_node._hierarchy.pop(split_feature_name)
                    if isinstance(value, str):
                        best_node._available_feature_names.append(value)
                    elif isinstance(value, list):
                        best_node._available_feature_names.extend(value)
                    else:
                        assert False

                child_node = self.__create_node(
                    mask=child_mask,
                    hierarchy=best_node._hierarchy,
                    available_feature_names=best_node._available_feature_names,
                    depth=best_node._depth+1,
                )
                child_node.feature_value = feature_value
                self.__leaf_counter += 1

                best_node.childs.append(child_node)
                if self.__is_splittable(child_node):
                    bisect.insort(
                        self.splittable_leaf_nodes,
                        child_node,
                        key=lambda x: x._best_split[0],
                    )

            best_node.is_leaf = False
            best_node.split_type = split_type
            best_node.split_feature_name = split_feature_name
            self.__leaf_counter -= 1

        del self.X
        del self.y
        del self.splittable_leaf_nodes

        self.__is_fitted = True

    def __is_splittable(self, node: TreeNode) -> bool:
        """Checks whether a tree node can be split."""
        if self.max_depth and node._depth >= self.max_depth:
            return False

        if node.samples < self.min_samples_split:
            return False

        if node.impurity == 0:
            return False

        best_split_results = self.__find_best_split(node._mask, node._available_feature_names)
        inf_gain = best_split_results[0]
        if inf_gain < self.min_impurity_decrease:
            return False
        else:
            node._best_split = best_split_results
            return True

    def __create_node(
        self,
        mask: pd.Series,
        hierarchy: dict[str, str | list[str]],
        available_feature_names: list[str],
        depth: int,
    ) -> TreeNode:
        """Create a node of the tree."""
        hierarchy = hierarchy.copy()
        available_feature_names = available_feature_names.copy()

        samples = mask.sum()
        distribution = self.__distribution(mask)
        impurity = self.__impurity(mask)
        label = self.y[mask].value_counts().index[0]

        tree_node = TreeNode(
            self.__node_counter,
            samples,
            distribution,
            impurity,
            label,
            depth,
            mask,
            hierarchy,
            available_feature_names,
        )

        self.__node_counter += 1

        return tree_node

    def __find_best_split(
        self,
        parent_mask: pd.Series,
        available_feature_names: list[str],
    ) -> tuple[float, str | None, str | None, list[list[str]] | None, list[pd.Series] | None]:
        """
        Finds the best tree node split, if it exists.

        Parameters:
            parent_mask: boolean mask of the tree node.
            available_feature_names: the list of features available for splitting.

        Returns:
            Tuple `(inf_gain, split_type, split_feature_name, feature_values,
            child_masks)`.
              inf_gain: information gain of the split.
              split_type: split type.
              split_feature_name: the feature by which it is best to split the input set.
              feature_values: feature values corresponding to child nodes.
              child_masks: list of child masks.
        """
        best_inf_gain = float("-inf")
        best_split_type = None
        best_split_feature_name = None
        best_feature_values = None
        best_child_masks = None
        for split_feature_name in available_feature_names:
            if split_feature_name in self.numerical_feature_names:
                split_type = "numerical"
                (
                    inf_gain,
                    feature_values,
                    child_masks,
                ) = self.__num_split(parent_mask, split_feature_name)
            elif split_feature_name in self.categorical_feature_names:
                split_type = "categorical"
                (
                    inf_gain,
                    feature_values,
                    child_masks,
                ) = self.__best_cat_split(parent_mask, split_feature_name)
            elif split_feature_name in self.rank_feature_names:
                split_type = "rank"
                (
                    inf_gain,
                    feature_values,
                    child_masks,
                ) = self.__best_rank_split(parent_mask, split_feature_name)

            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_split_type = split_type
                best_split_feature_name = split_feature_name
                best_feature_values = feature_values
                best_child_masks = child_masks

        return (
            best_inf_gain,
            best_split_type,
            best_split_feature_name,
            best_feature_values,
            best_child_masks,
        )

    def __num_split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
    ) -> tuple[float, list[list[str]] | None, list[pd.Series] | None]:
        """
        Finds the best tree node split by set numerical feature, if it exists.

        Parameters:
            parent_mask: boolean mask of the tree node.
            split_feature_name: The name of the set numerical feature by which to find
              the best split.

        Returns:
            Tuple `(inf_gain, feature_values, child_masks)`.
              inf_gain: information gain of the split.
              feature_values: feature values corresponding to child nodes.
              child_masks: boolean masks of child nodes.
        """
        use_including_na = (
            self.numerical_nan_mode == "include"
            # and there are samples with missing values
            and (parent_mask & self.X[split_feature_name].isna()).sum()
        )

        if use_including_na:
            mask_notna = parent_mask & self.X[split_feature_name].notna()
            # if split by feature value is not possible
            if mask_notna.sum() <= 1:
                return float("-inf"), None, None
            mask_na = parent_mask & self.X[split_feature_name].isna()

            points = self.X.loc[mask_notna, split_feature_name].to_numpy()
        else:
            points = self.X.loc[parent_mask, split_feature_name].to_numpy()

        thresholds = get_thresholds(points)

        best_inf_gain = float("-inf")
        best_feature_values = None
        best_child_masks = None
        for threshold in thresholds:
            mask_less = parent_mask & (self.X[split_feature_name] <= threshold)
            mask_more = parent_mask & (self.X[split_feature_name] > threshold)

            if use_including_na:
                mask_less = mask_less | mask_na
                mask_more = mask_more | mask_na

            if (
                mask_less.sum() < self.min_samples_leaf
                or mask_more.sum() < self.min_samples_leaf
            ):
                continue

            child_masks = [mask_less, mask_more]

            inf_gain = self.__information_gain(
                parent_mask, child_masks, nan_mode=self.numerical_nan_mode)

            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                less_values = [f"<= {threshold}"]
                more_values = [f"> {threshold}"]
                best_feature_values = [less_values, more_values]
                best_child_masks = child_masks

        return best_inf_gain, best_feature_values, best_child_masks

    def __best_cat_split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
    ) -> tuple[float, list[list[str]] | None, list[pd.Series] | None]:
        """
        Split a node according to a categorical feature in the best way.

        Parameters:
            parent_mask: boolean mask of split node.
            split_feature_name: feature according to which node should be split.

        Returns:
            Tuple `(inf_gain, feature_values, child_masks)`.
              inf_gain: information gain of the split.
              feature_values: feature values corresponding to child nodes.
              child_masks: boolean masks of child nodes.
        """
        available_feature_values = self.X.loc[parent_mask, split_feature_name].unique()
        if (
            self.categorical_nan_mode == "include"
            and pd.isna(available_feature_values).any()  # if contains missing values
        ):
            available_feature_values = available_feature_values[~pd.isna(available_feature_values)]
        if len(available_feature_values) <= 1:
            return float("-inf"), None, None
        available_feature_values = sorted(available_feature_values)

        # get list of all possible partitions
        partitions = []
        for partition in cat_partitions(available_feature_values):
            # if partitions is not really partitions
            if len(partition) < 2:
                continue
            # limitation of branching
            if len(partition) > self.max_childs:
                continue
            # if the number of leaves exceeds the limit after splitting
            if self.__leaf_counter + len(partition) > self.max_leaf_nodes:
                continue

            partitions.append(partition)

        best_inf_gain = float("-inf")
        best_feature_values = None
        best_child_masks = None
        for feature_values in partitions:
            inf_gain, child_masks = \
                self.__cat_split(parent_mask, split_feature_name, feature_values)
            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_child_masks = child_masks
                best_feature_values = feature_values

        return best_inf_gain, best_feature_values, best_child_masks

    def __cat_split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
        feature_values: list[list],
    ) -> tuple[float, list[pd.Series] | None]:
        """
        Split a node according to a categorical feature according to the
        defined feature values.

        Parameters:
            parent_mask: boolean mask of split node.
            split_feature_name: feature according to which node should be split.
            feature_values: feature values corresponding to child nodes.

        Returns:
            Tuple `(inf_gain, child_masks)`.
              inf_gain: information gain of the split.
              child_masks: boolean masks of child nodes.
        """
        mask_na = parent_mask & self.X[split_feature_name].isna()

        child_masks = []
        for list_ in feature_values:
            child_mask = parent_mask & (self.X[split_feature_name].isin(list_) | mask_na)
            if child_mask.sum() < self.min_samples_leaf:
                return float("-inf"), None
            child_masks.append(child_mask)

        inf_gain = self.__information_gain(
            parent_mask, child_masks, nan_mode=self.categorical_nan_mode)

        return inf_gain, child_masks

    def __best_rank_split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
    ) -> tuple[float, list[list[str]], list[pd.Series]]:
        """Split a node according to a rank feature in the best way."""
        available_feature_values = self.rank_feature_names[split_feature_name]

        best_inf_gain = float("-inf")
        best_child_masks = None
        best_feature_values = None
        for feature_values in rank_partitions(available_feature_values):
            inf_gain, child_masks = \
                self.__rank_split(parent_mask, split_feature_name, feature_values)
            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_child_masks = child_masks
                best_feature_values = feature_values

        return best_inf_gain, best_feature_values, best_child_masks

    def __rank_split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
        feature_values: list[list[str]],
    ) -> tuple[float, list[pd.Series] | None]:
        """
        Splits a node according to a rank feature according to the defined feature
        values.
        """
        left_list_, right_list_ = feature_values

        mask_left = parent_mask & self.X[split_feature_name].isin(left_list_)
        mask_right = parent_mask & self.X[split_feature_name].isin(right_list_)

        if (
            mask_left.sum() < self.min_samples_leaf
            or mask_right.sum() < self.min_samples_leaf
        ):
            return float("-inf"), None

        child_masks = [mask_left, mask_right]

        inf_gain = self.__information_gain(parent_mask, child_masks)

        return inf_gain, child_masks

    def __information_gain(
        self,
        parent_mask: pd.Series,
        child_masks: list[pd.Series],
        nan_mode: str | None = None,
    ) -> float:
        """
        Calculates information gain of the split.

        Parameters:
            parent_mask: boolean mask of parent node.
            child_masks: list of boolean masks of child nodes.
            nan_mode: missing values handling node.
              If 'include', then turn on normalization of child nodes impurity.

        Returns:
            information gain.

        References:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

        Formula in LaTeX:
            \text{Information Gain} = \frac{N_{\text{parent}}}{N} * \Big(\text{impurity}_{\text{parent}} - \sum^C_{i=1}{\frac{N_{\text{child}_i}}{N_{\text{parent}}}} * \text{impurity}_{\text{child}_i} \Big)
            where
            \text{Information Gain} - information fain;
            N - the number of samples in the entire training set;
            N_{\text{parent}} - the number of samples in the parent node;
            \text{impurity}_{\text{parent}} - the parent node impurity;
            С - the number of child nodes;
            N_{\text{child}_i} - the number of samples in the child node;
            \text{impurity}_{\text{child}_i} - the child node impurity.
        """
        N = self.y.shape[0]
        N_parent = parent_mask.sum()

        impurity_parent = self.__impurity(parent_mask)

        weighted_impurity_childs = 0
        N_childs = 0
        for child_mask_i in child_masks:
            N_child_i = child_mask_i.sum()
            N_childs += N_child_i
            impurity_child_i = self.__impurity(child_mask_i)
            weighted_impurity_childs += (N_child_i / N_parent) * impurity_child_i

        if nan_mode == "include":
            norm_coef = N_parent / N_childs
            weighted_impurity_childs *= norm_coef

        local_information_gain = impurity_parent - weighted_impurity_childs

        information_gain = (N_parent / N) * local_information_gain

        return information_gain

    def __gini_index(self, mask: pd.Series) -> float:
        """
        Calculates Gini index in a tree node.

        Gini index formula in LaTeX:
            \text{Gini Index} = \sum^C_{i=1} p_i \times (1 - p_i)
            where
            \text{Gini Index} - Gini index;
            C - total number of classes;
            p_i - the probability of choosing a sample with class i.
        """
        N = mask.sum()

        gini_index = 0
        for label in self.__class_names:
            N_i = (mask & (self.y == label)).sum()
            p_i = N_i / N
            gini_index += p_i * (1 - p_i)

        return gini_index

    def __entropy(self, mask: pd.Series) -> float:
        """
        Calculates entropy in a tree node.

        Entropy formula in LaTeX:
        H = \log{\overline{N}} = \sum^N_{i=1} p_i \log{(1/p_i)} = -\sum^N_{i=1} p_i \log{p_i}
        where
        H - entropy;
        \overline{N} - effective number of states;
        p_i - probability of the i-th system state.
        """
        N = mask.sum()

        entropy = 0
        for label in self.__class_names:
            N_i = (mask & (self.y == label)).sum()
            if N_i != 0:
                p_i = N_i / N
                entropy -= p_i * math.log2(p_i)

        return entropy

    def __distribution(self, mask: pd.Series) -> list[int]:
        """Calculates the class distribution."""
        distribution = [
            (mask & (self.y == class_name)).sum()
            for class_name in self.__class_names
        ]

        return distribution

    def predict(self, X: pd.DataFrame | pd.Series) -> list[str] | str:
        """
        Predict class for samples in X.

        Parameters:
            X: The input samples.

        Returns:
            The predicted classes.
        """
        if not self.__is_fitted:
            raise NotFittedError(
                "This MultiSplitDecisionTree instance is not fitted yet."
                " Call `fit` with appropriate arguments before using this estimator."
            )

        y_pred_proba_s = self.predict_proba(X)
        y_pred = [self.__class_names[y_pred_proba.argmax()] for y_pred_proba in y_pred_proba_s]

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
        if not self.__is_fitted:
            raise NotFittedError(
                "This MultiSplitDecisionTree instance is not fitted yet."
                " Call `fit` with appropriate arguments before using this estimator."
            )

        # TODO: write __check_predict_proba_data()

        X = self.__preprocess(X)

        y_pred_proba = np.array([
            self.__predict_proba(self.__root, point)[0]
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
                X.fillna(self.__fill_numerical_nan_values[num_feature], inplace=True)

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

        fitted_features_set = set(self.__feature_names)
        X_features_set = set(X.columns)
        if fitted_features_set != X_features_set:
            message = (
                "The feature names should match those that were passed during fit.\n"
            )

            unexpected_names = sorted(X_features_set - fitted_features_set)
            missing_names = sorted(fitted_features_set - X_features_set)

            def add_names(names):
                output = ''
                max_n_names = 5
                for i, name in enumerate(names):
                    if i >= max_n_names:
                        output += "- ...\n"
                        break
                    output += f"- {name}\n"
                return output

            if unexpected_names:
                message += "Feature names unseen at fit time:\n"
                message += add_names(unexpected_names)

            if missing_names:
                message += "Feature names seen at fit time, yet now missing:\n"
                message += add_names(missing_names)

            # TODO: same order of features

            raise ValueError(message)

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> float:
        """Returns the accuracy metric."""
        if not self.__is_fitted:
            raise NotFittedError(
                "This MultiSplitDecisionTree instance is not fitted yet."
                " Call `fit` with appropriate arguments before using this estimator."
            )

        self.__check_score_data(X, y, sample_weight)

        score = accuracy_score(y, self.predict(X), sample_weight=sample_weight)

        return score

    def get_params(
        self,
        deep: bool = True,  # implemented for sklearn.model_selection.GridSearchCV
    ) -> dict:
        """Возвращает параметры этого классификатора."""
        return {
            "criterion": self.criterion,
            "max_depth": self.__max_depth,
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

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        # Simple optimization to gain speed (inspect is slow)
        if not params:
            return self

        valid_params = self.get_params(deep=True)

        for param, value in params.items():
            if param not in valid_params:
                raise ValueError(
                    f"Invalid parameter {param} for estimator {self}."
                    f" Valid parameters are: {valid_params}."
                )
            setattr(self, f'_{self.__class__.__name__}__{param}', value)

        return self

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
            rounded: whether to round the corners of the nodes
              (they are in the shape of a rectangle).
            show_impurity: whether to show the impurity of the node.
            show_num_samples: whether to show the number of samples in the node.
            show_distribution: whether to show the class distribution.
            show_label: whether to show the class to which the node belongs.
            **kwargs: arguments for graphviz.Digraph.render.

        Returns:
            An object of the Digraph class containing a description of the graph
            structure of the tree for visualization.
        """
        if not self.__is_fitted:
            raise NotFittedError(
                "This MultiSplitDecisionTree instance is not fitted yet."
                " Call `fit` with appropriate arguments before using this estimator."
            )

        if self.__graph is None:
            self.__create_graph(
                rounded, show_impurity, show_num_samples, show_distribution, show_label)
        if kwargs:
            self.__graph.render(**kwargs)

        return self.__graph

    def __create_graph(
        self,
        rounded: bool,
        show_impurity: bool,
        show_num_samples: bool,
        show_distribution: bool,
        show_label: bool,
    ) -> None:
        """
        Creates an object of the Digraph class containing a description of the graph
        structure of the tree for visualization.
        """
        node_attr = {"shape": "box"}
        if rounded:
            node_attr["style"] = "rounded"

        self.__graph = Digraph(name="decision tree", node_attr=node_attr)
        self.__add_node(
            node=self.__root,
            parent_name=None,
            show_impurity=show_impurity,
            show_num_samples=show_num_samples,
            show_distribution=show_distribution,
            show_label=show_label,
        )

    def __add_node(
        self,
        node: TreeNode,
        parent_name: str | None,
        show_impurity: bool,
        show_num_samples: bool,
        show_distribution: bool,
        show_label: bool,
    ) -> None:
        """
        Recursively adds a description of the node and its relationship to the parent
        node (if available).
        """
        node_name = f"node {node.number}"

        node_content = [f"Node {node.number}"]
        if node.split_feature_name:
            node_content.append(f"{node.split_feature_name}")
        if show_impurity:
            node_content.append(f"{self.criterion} = {node.impurity:.2f}")
        if show_num_samples:
            node_content.append(f"samples = {node.samples}")
        if show_distribution:
            node_content.append(f"distribution = {node.distribution}")
        if show_label:
            node_content.append(f"label = {node.label}")
        node_content = "\n".join(node_content)

        self.__graph.node(name=node_name, label=node_content)

        if parent_name:
            edge_label = "\n".join([str(fv) for fv in node.feature_value])
            self.__graph.edge(
                tail_name=parent_name,
                head_name=node_name,
                label=edge_label,
            )

        for child in node.childs:
            self.__add_node(
                node=child,
                parent_name=node_name,
                show_impurity=show_impurity,
                show_num_samples=show_num_samples,
                show_distribution=show_distribution,
                show_label=show_label,
            )
