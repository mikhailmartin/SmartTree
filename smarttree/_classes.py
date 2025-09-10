"""Custom realization of Decision Tree which can handle categorical features."""
import logging
import math
from abc import abstractmethod
from functools import lru_cache
from typing import Self

import numpy as np
import pandas as pd
from graphviz import Digraph
from sklearn.metrics import accuracy_score

from ._builder import Builder
from ._check import check__data, check__params
from ._exceptions import NotFittedError
from ._node_splitter import NodeSplitter
from ._renderer import Renderer
from ._tree_node import TreeNode
from ._types import (
    CategoricalNaModeType,
    ClassificationCriterionType,
    NumericalNaModeType,
    VerboseType,
)


class BaseSmartDecisionTree:
    """Base class for smart decision trees."""

    def __init__(
        self,
        *,
        criterion: ClassificationCriterionType = "gini",
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
        numerical_na_mode: NumericalNaModeType = "min",
        categorical_na_mode: CategoricalNaModeType = "as_category",
        categorical_na_filler: str = "missing_value",
        verbose: VerboseType = "WARNING",
    ) -> None:

        check__params(
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
            numerical_na_mode=numerical_na_mode,
            categorical_na_mode=categorical_na_mode,
            categorical_na_filler=categorical_na_filler,
        )

        self.__criterion = criterion
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__max_leaf_nodes = max_leaf_nodes
        self.__min_impurity_decrease = min_impurity_decrease
        self.__max_childs = max_childs
        self.__hierarchy = dict() if hierarchy is None else hierarchy

        if numerical_feature_names is None:
            self.__numerical_feature_names = []
        elif isinstance(numerical_feature_names, str):
            self.__numerical_feature_names = [numerical_feature_names]
        else:
            self.__numerical_feature_names = numerical_feature_names

        if categorical_feature_names is None:
            self.__categorical_feature_names = []
        elif isinstance(categorical_feature_names, str):
            self.__categorical_feature_names = [categorical_feature_names]
        else:
            self.__categorical_feature_names = categorical_feature_names

        if rank_feature_names is None:
            self.__rank_feature_names = dict()
        else:
            self.__rank_feature_names = rank_feature_names

        self._all_feature_names: list[str] = []
        self.__numerical_na_mode = numerical_na_mode
        self.__categorical_na_mode = categorical_na_mode
        self.__categorical_na_filler = categorical_na_filler

        self.logger = logging.getLogger()
        self.logger.setLevel(verbose)

        self._is_fitted: bool = False
        self._root: TreeNode | None = None
        self._feature_importances: dict = dict()
        self._feature_na_mode: dict[str, NumericalNaModeType | CategoricalNaModeType | None] = dict()
        self._numerical_na_filler: dict[str, int | float] = dict()

    @property
    def criterion(self) -> ClassificationCriterionType:
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
    def all_feature_names(self) -> list[str]:
        self._check_is_fitted()
        return self._all_feature_names

    @property
    def hierarchy(self) -> dict[str, str | list[str]]:
        return self.__hierarchy

    @property
    def numerical_na_mode(self) -> NumericalNaModeType:
        return self.__numerical_na_mode

    @property
    def categorical_na_mode(self) -> CategoricalNaModeType:
        return self.__categorical_na_mode

    @property
    def categorical_na_filler(self) -> str:
        return self.__categorical_na_filler

    @property
    def tree(self) -> TreeNode:
        self._check_is_fitted()
        assert self._root is not None
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
    def predict(self, X: pd.DataFrame) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def score(self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> float | np.floating:
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
            "numerical_na_mode": self.numerical_na_mode,
            "categorical_na_mode": self.categorical_na_mode,
            "categorical_na_filler": self.categorical_na_filler,
        }

    def set_params(self, **params) -> Self:
        """Set the parameters of this estimator instance."""
        # Simple optimization to gain speed (inspect is slow)
        if not params:
            return self

        valid_params = self.get_params(deep=True)

        for param, value in params.items():
            if param not in valid_params:
                raise ValueError(
                    f"Invalid parameter `{param}` for estimator {self.__class__.__name__}."
                    f" Valid parameters are: {', '.join(valid_params.keys())}."
                )
        check__params(**params)

        for param, value in params.items():
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
        criterion: {"gini", "entropy", "log_loss"}, default="gini"
          The function to measure the quality of a split. Supported criteria are
          "gini" for the Gini impurity and "log_loss" and "entropy" both for
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

          The weighted impurity decrease equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

          where ``N`` is the total number of samples, ``N_t`` is the number of
          samples at the current node, ``N_t_L`` is the number of samples in the
          left child, and ``N_t_R`` is the number of samples in the right child.

          ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
          if ``sample_weight`` is passed.

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

        hierarchy: dict[str, str | list[str]], default=None
          A hierarchical dependency between features that determines the order
          in which they can be used for splitting nodes in the decision tree.
          If provided, the algorithm will respect these dependencies when
          selecting features for splits.

        numerical_na_mode: {"min", "max", "include_all", "include_best"},
                           default="min"
          The mode of handling missing values in a numerical feature.

          - If "min", then missing values are filled with minimum value of
            a numerical feature in training data.
          - If "max", then missing values are filled with maximum value of
            a numerical feature in training data.
          - If "include_all", then while training samples with missing values
            are included into all child nodes. While predicting decision is
            weighted mean of all decisions in child nodes.
          - If "include_best", then while training and prediction samples with
            missing values are included into the best child node according to
            information gain.

        categorical_na_mode: {"as_category", "include_all", "include_best"},
                             default="as_category"
          The mode of handling missing values in a categorical feature.

          - If "as_category", then while training and predicting missing values
            will be filled with `categorical_na_filler`.
          - If "include_all", then while training samples with missing values
            are included into all child nodes. While predicting decision is
            weighted mean of all decisions in child nodes.
          - If "include_best", then while training and prediction samples with
            missing values are included into the best child node according to
            information gain.

        categorical_na_filler: str, default="missing_value"
          If `categorical_na_mode` is set to "as_category", then during
          training and predicting missing values will be filled with
          `categorical_na_filler`.

        verbose: {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"} or int,
                 default="WARNING"
          Controls the level of decision tree verbosity.
    """

    def __init__(
        self,
        *,
        criterion: ClassificationCriterionType = "gini",
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
        numerical_na_mode: NumericalNaModeType = "min",
        categorical_na_mode: CategoricalNaModeType = "as_category",
        categorical_na_filler: str = "missing_value",
        verbose: VerboseType = "WARNING",
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
            numerical_na_mode=numerical_na_mode,
            categorical_na_mode=categorical_na_mode,
            categorical_na_filler=categorical_na_filler,
            verbose=verbose,
        )
        self.__classes: list[str] = []

    @property
    def classes_(self) -> list[str]:  # TODO: -> np.ndarray
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
        if self.numerical_na_mode != "min":
            repr_.append(f"numerical_na_mode={self.numerical_na_mode!r}")
        if self.categorical_na_mode != "as_category":
            repr_.append(f"categorical_na_mode={self.categorical_na_mode!r}")
        if self.categorical_na_filler != "missing_value":
            repr_.append(f"categorical_na_filler={self.categorical_na_filler!r}")

        return (
            f"{self.__class__.__name__}({', '.join(repr_)})"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters:
            X: pd.DataFrame
              The training input samples.
            y: pd.Series
              The target values.
        """
        check__data(
            X=X,
            y=y,
            numerical_feature_names=self.numerical_feature_names,
            categorical_feature_names=self.categorical_feature_names,
            rank_feature_names=self.rank_feature_names,
        )

        ################################################################################
        max_depth = float("+inf") if self.max_depth is None else self.max_depth

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

        max_childs = float("+inf") if self.max_childs is None else self.max_childs

        known_feature_names = (
            self.numerical_feature_names
            + self.categorical_feature_names
            + list(self.rank_feature_names.keys())
        )
        unknown_num_feature_names = (
            X.drop(columns=known_feature_names).select_dtypes("number").columns.to_list()
        )
        unknown_cat_feature_names = (
            X.drop(columns=known_feature_names).select_dtypes(include=["category", "object"]).columns.to_list()
        )
        if unknown_num_feature_names:
            self.numerical_feature_names.extend(unknown_num_feature_names)
            self.logger.info(
                f"[{self.__class__.__name__}] [Info] {unknown_num_feature_names} are"
                " added to `numerical_feature_names`."
            )
        if unknown_cat_feature_names:
            self.categorical_feature_names.extend(unknown_cat_feature_names)
            self.logger.info(
                f"[{self.__class__.__name__}] [Info] {unknown_cat_feature_names} are"
                " added to `categorical_feature_names`."
            )

        self._all_feature_names = X.columns.tolist()
        self.__classes = sorted(y.unique())

        if self.numerical_na_mode in ("min", "max"):
            for numerical_feature_name in self.numerical_feature_names:
                if self.numerical_na_mode == "min":
                    na_filler = X[numerical_feature_name].min()
                else:  # max
                    na_filler = X[numerical_feature_name].max()
                self._numerical_na_filler[numerical_feature_name] = na_filler

        for numerical_feature_name in self.numerical_feature_names:
            self._feature_na_mode[numerical_feature_name] = self.numerical_na_mode
        for categorical_feature_name in self.categorical_feature_names:
            self._feature_na_mode[categorical_feature_name] = self.categorical_na_mode
        for rank_feature_name in self.rank_feature_names:
            self._feature_na_mode[rank_feature_name] = None

        X = self.__preprocess(X)

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
            numerical_feature_names=self.numerical_feature_names,
            categorical_feature_names=self.categorical_feature_names,
            rank_feature_names=self.rank_feature_names,
            numerical_na_mode=self.numerical_na_mode,
            categorical_na_mode=self.categorical_na_mode,
        )

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

    def predict(self, X: pd.DataFrame) -> list[str]:
        """
        Predict class for samples in X.

        Parameters:
            X: pd.DataFrame
              The input samples.

        Returns:
            list[str]: The predicted classes.
        """
        y_pred_proba_s = self.predict_proba(X)
        y_pred = [
            self.__classes[y_pred_proba.argmax()] for y_pred_proba in y_pred_proba_s
        ]

        return y_pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of
        the same class in a leaf.

        Parameters:
            X: pd.DataFrame
              The input samples.

        Returns:
            ndarray: The class probabilities of the input samples. The order of
            the classes corresponds to that in the attribute :term:`class_names`.
        """
        check__data(X=X, all_feature_names=self.all_feature_names)

        X = self.__preprocess(X)

        distributions = [
            self.__get_distribution(self.tree, point) for _, point in X.iterrows()
        ]
        y_pred_proba = np.array([
            distribution / distribution.sum() for distribution in distributions
        ])

        return y_pred_proba

    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class log-probabilities of the input samples X.

        Parameters:
            X: pd.DataFrame
              The input samples.

        Returns:
            ndarray: The class log-probabilities of the input samples. The order
            of the classes corresponds to that in the attribute :term:`classes_`.
        """
        y_pred_proba = self.predict_proba(X)
        for i in range(len(y_pred_proba)):
            y_pred_proba[i] = np.log(y_pred_proba[i])

        return y_pred_proba

    def __preprocess(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.copy()

        if self.numerical_na_mode in ("min", "max"):
            for num_feature in self.numerical_feature_names:
                na_filler = self._numerical_na_filler[num_feature]
                X[num_feature].fillna(na_filler, inplace=True)

        if self.categorical_na_mode == "as_category":
            for cat_feature in self.categorical_feature_names:
                X[cat_feature].fillna(self.categorical_na_filler, inplace=True)

        return X

    def __get_distribution(self, node: TreeNode, point: pd.Series) -> np.ndarray:

        if node.is_leaf:
            return node.distribution

        else:
            if pd.isna(point[node.split_feature_name]):
                distribution = np.array([0, 0, 0], dtype="int")
                for child in node.childs:
                    distribution += self.__get_distribution(child, point)
                return distribution

            elif node.split_type == "numerical":
                threshold = float(node.childs[0].feature_value[0][3:])
                if point[node.split_feature_name] <= threshold:
                    return self.__get_distribution(node.childs[0], point)
                else:
                    return self.__get_distribution(node.childs[1], point)

            elif node.split_type in ("categorical", "rank"):
                for child in node.childs:
                    if point[node.split_feature_name] in child.feature_value:
                        return self.__get_distribution(child, point)
                else:
                    # if there is no such branch TODO
                    return node.distribution

            else:
                assert False

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> float | np.floating:
        """Returns the accuracy metric."""
        check__data(X=X, y=y, all_feature_names=self.all_feature_names)

        score = accuracy_score(y, self.predict(X), sample_weight=sample_weight)

        return score

    @lru_cache
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
        renderer = Renderer(criterion=self.criterion, rounded=rounded)
        graph = renderer.render(
            tree=self.tree,
            show_impurity=show_impurity,
            show_num_samples=show_num_samples,
            show_distribution=show_distribution,
            show_label=show_label,
            **kwargs,
        )

        return graph
