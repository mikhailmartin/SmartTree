"""Custom realization of Decision Tree which can handle categorical features."""
import logging
import math
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Self

import numpy as np
import pandas as pd
from graphviz import Digraph
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score

from ._builder import Builder
from ._check import check__data, check__params
from ._exceptions import NotFittedError
from ._node_splitter import NodeSplitter
from ._renderer import Renderer
from ._tree import Tree, TreeNode
from ._types import (
    CatNaModeType,
    ClassificationCriterionType,
    CommonNaModeType,
    NaModeType,
    NumNaModeType,
    VerboseType,
)


class BaseSmartDecisionTree(ABC):
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
        num_features: list[str] | str | None = None,
        cat_features: list[str] | str | None = None,
        rank_features: dict[str, list] | None = None,
        hierarchy: dict[str, str | list[str]] | None = None,
        na_mode: CommonNaModeType = "include_best",
        num_na_mode: NumNaModeType | None = None,
        cat_na_mode: CatNaModeType | None = None,
        cat_na_filler: str = "missing_value",
        rank_na_mode: CommonNaModeType | None = None,
        feature_na_mode: dict[str, NaModeType] | None = None,
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
            num_features=num_features,
            cat_features=cat_features,
            rank_features=rank_features,
            hierarchy=hierarchy,
            na_mode=na_mode,
            num_na_mode=num_na_mode,
            cat_na_mode=cat_na_mode,
            cat_na_filler=cat_na_filler,
            rank_na_mode=rank_na_mode,
            feature_na_mode=feature_na_mode,
        )

        self.__criterion = criterion
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__max_leaf_nodes = max_leaf_nodes
        self.__min_impurity_decrease = min_impurity_decrease
        self.__max_childs = max_childs
        self.__hierarchy = hierarchy or dict()

        if num_features is None:
            self.__num_features = []
        elif isinstance(num_features, str):
            self.__num_features = [num_features]
        else:
            self.__num_features = num_features

        if cat_features is None:
            self.__cat_features = []
        elif isinstance(cat_features, str):
            self.__cat_features = [cat_features]
        else:
            self.__cat_features = cat_features

        self.__rank_features = rank_features or dict()

        self._all_features: list[str] = []

        self.__na_mode = na_mode
        self.__num_na_mode = num_na_mode
        self.__cat_na_mode = cat_na_mode
        self.__cat_na_filler = cat_na_filler
        self.__rank_na_mode = rank_na_mode
        self.__feature_na_mode: dict[str, NaModeType] = feature_na_mode or dict()

        self.logger = logging.getLogger()
        self.logger.setLevel(verbose)

        self._is_fitted: bool = False
        self._tree: Tree | None = None
        self._feature_importances: dict = dict()
        self._feature_na_filler: dict[str, int | float | str] = dict()

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
    def num_features(self) -> list[str]:
        return self.__num_features

    @property
    def cat_features(self) -> list[str]:
        return self.__cat_features

    @property
    def rank_features(self) -> dict[str, list]:
        return self.__rank_features

    @property
    def all_features(self) -> list[str]:
        self._check_is_fitted()
        return self._all_features

    @property
    def hierarchy(self) -> dict[str, str | list[str]]:
        return self.__hierarchy

    @property
    def na_mode(self) -> CommonNaModeType:
        return self.__na_mode

    @property
    def num_na_mode(self) -> NumNaModeType | None:
        return self.__num_na_mode

    @property
    def cat_na_mode(self) -> CatNaModeType | None:
        return self.__cat_na_mode

    @property
    def cat_na_filler(self) -> str:
        return self.__cat_na_filler

    @property
    def rank_na_mode(self) -> CommonNaModeType | None:
        return self.__rank_na_mode

    @property
    def feature_na_mode(self) -> dict[str, NaModeType]:
        return self.__feature_na_mode

    @property
    def tree_(self) -> Tree:
        self._check_is_fitted()
        assert self._tree is not None
        return self._tree

    def get_n_leaves(self) -> int:
        return self.tree_.leaf_counter

    def get_depth(self) -> int:
        return self.tree_.max_depth

    @property
    def feature_importances_(self) -> dict[str, float]:
        self._check_is_fitted()
        return self.tree_.compute_feature_importances()

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        raise NotImplementedError

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet."
                " Call `fit` with appropriate arguments before using this estimator."
            )

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> NDArray:
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
            "num_features": self.num_features,
            "cat_features": self.cat_features,
            "rank_features": self.rank_features,
            "hierarchy": self.hierarchy,
            "na_mode": self.na_mode,
            "num_na_mode": self.num_na_mode,
            "cat_na_mode": self.cat_na_mode,
            "cat_na_filler": self.cat_na_filler,
            "rank_na_mode": self.rank_na_mode,
            "feature_na_mode": self.feature_na_mode,
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

        num_features: list[str] or str, default=None
          List of numerical feature names. If None `numerical_features` will be
          set from unset feature names in X while .fit().

        cat_features: list[str] or str, default=None
          List of categorical feature names. If None `categorical_features`
          will be set from unset feature names in X while .fit().

        rank_features: list[str] or str, default=None
          List of rank feature names.

        hierarchy: dict[str, str | list[str]], default=None
          A hierarchical dependency between features that determines the order
          in which they can be used for splitting nodes in the decision tree.
          If provided, the algorithm will respect these dependencies when
          selecting features for splits.

        na_mode: {"include_all", "include_best"}, default="include_best"
          The mode of handling missing values in a feature.

          - If "include_all", then while training samples with missing values
            are included into all child nodes. While predicting decision is
            weighted mean of all decisions in child nodes.
          - If "include_best", then while training and prediction samples with
            missing values are included into the best child node according to
            information gain.

        num_na_mode: {"min", "max", "include_all", "include_best"}, default=None
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

        cat_na_mode: {"as_category", "include_all", "include_best"}, default=None
          The mode of handling missing values in a categorical feature.

          - If "as_category", then while training and predicting missing values
            will be filled with `categorical_na_filler`.
          - If "include_all", then while training samples with missing values
            are included into all child nodes. While predicting decision is
            weighted mean of all decisions in child nodes.
          - If "include_best", then while training and prediction samples with
            missing values are included into the best child node according to
            information gain.

        cat_na_filler: str, default="missing_value"
          If `cat_na_mode` is set to "as_category", then during training and
          predicting missing values will be filled with `cat_na_filler`.

        rank_na_mode: {"include_all", "include_best"}, default=None
          The mode of handling missing values in a rank feature.

          - If "include_all", then while training samples with missing values
            are included into all child nodes. While predicting decision is
            weighted mean of all decisions in child nodes.
          - If "include_best", then while training and prediction samples with
            missing values are included into the best child node according to
            information gain.

        feature_na_mode: dict[str, {"min", "max", "as_category", "include_all", "include_best"}],
                         default=None
          The mode of handling missing values in a feature.

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
        num_features: list[str] | str | None = None,
        cat_features: list[str] | str | None = None,
        rank_features: dict[str, list] | None = None,
        hierarchy: dict[str, str | list[str]] | None = None,
        na_mode: CommonNaModeType = "include_best",
        num_na_mode: NumNaModeType | None = None,
        cat_na_mode: CatNaModeType | None = None,
        cat_na_filler: str = "missing_value",
        rank_na_mode: CommonNaModeType | None = None,
        feature_na_mode: dict[str, NaModeType] | None = None,
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
            num_features=num_features,
            cat_features=cat_features,
            rank_features=rank_features,
            hierarchy=hierarchy,
            na_mode=na_mode,
            num_na_mode=num_na_mode,
            cat_na_mode=cat_na_mode,
            cat_na_filler=cat_na_filler,
            rank_na_mode=rank_na_mode,
            feature_na_mode=feature_na_mode,
            verbose=verbose,
        )
        self.__classes: NDArray = np.array([])

    @property
    def classes_(self) -> NDArray:
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
        if self.num_features:
            repr_.append(f"num_features={self.num_features}")
        if self.cat_features:
            repr_.append(f"cat_features={self.cat_features}")
        if self.rank_features:
            repr_.append(f"rank_features={self.rank_features}")
        if self.hierarchy:
            repr_.append(f"hierarchy={self.hierarchy}")
        if self.na_mode != "include_best":
            repr_.append(f"na_mode={self.na_mode!r}")
        if self.num_na_mode:
            repr_.append(f"num_na_mode={self.num_na_mode!r}")
        if self.cat_na_mode:
            repr_.append(f"cat_na_mode={self.cat_na_mode!r}")
        if self.cat_na_filler != "missing_value":
            repr_.append(f"cat_na_filler={self.cat_na_filler!r}")
        if self.rank_na_mode:
            repr_.append(f"rank_na_mode={self.rank_na_mode!r}")

        return (
            f"{self.__class__.__name__}({', '.join(repr_)})"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters:
            X: pd.DataFrame
              The training input samples.
            y: pd.Series
              The target values.
        """
        X, y = check__data(
            X=X,
            y=y,
            num_features=self.num_features,
            cat_features=self.cat_features,
            rank_features=self.rank_features,
        )

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

        known_features = (
            self.num_features + self.cat_features + list(self.rank_features.keys())
        )
        unknown_num_features = (
            X.drop(columns=known_features).select_dtypes("number").columns.to_list()
        )
        unknown_cat_features = (
            X.drop(columns=known_features)
            .select_dtypes(include=["category", "object"]).columns.to_list()
        )
        if unknown_num_features:
            self.num_features.extend(unknown_num_features)
            self.logger.info(
                f"[{self.__class__.__name__}] [Info] {unknown_num_features} are"
                " added to `num_features`."
            )
        if unknown_cat_features:
            self.cat_features.extend(unknown_cat_features)
            self.logger.info(
                f"[{self.__class__.__name__}] [Info] {unknown_cat_features} are"
                " added to `cat_features`."
            )

        self._all_features = X.columns.to_list()

        temp_feature_na_mode = self.feature_na_mode.copy()
        self.feature_na_mode.update({f: self.na_mode for f in self._all_features})
        if self.num_na_mode is not None:
            self.feature_na_mode.update({f: self.num_na_mode for f in self.num_features})
        if self.cat_na_mode is not None:
            self.feature_na_mode.update({f: self.cat_na_mode for f in self.cat_features})
        if self.rank_na_mode is not None:
            self.feature_na_mode.update({f: self.rank_na_mode for f in self.rank_features})
        self.feature_na_mode.update(temp_feature_na_mode)

        self.__classes = np.sort(y.unique())

        for feature, na_mode in self.feature_na_mode.items():
            if na_mode == "min":
                na_filler = X[feature].min()
            elif na_mode == "max":
                na_filler = X[feature].max()
            elif na_mode == "as_category":
                na_filler = self.cat_na_filler
            else:
                continue
            self._feature_na_filler[feature] = na_filler

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
            num_features=self.num_features,
            cat_features=self.cat_features,
            rank_features=self.rank_features,
            feature_na_mode=self.feature_na_mode,
        )

        self._tree = Tree()

        builder = Builder(
            X=X,
            y=y,
            criterion=self.criterion,
            splitter=splitter,
            max_leaf_nodes=max_leaf_nodes,
            hierarchy=self.hierarchy,
        )
        builder.build(self._tree)

        self._is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> NDArray:
        """
        Predict class for samples in X.

        Parameters:
            X: pd.DataFrame
              The input samples.

        Returns:
            ndarray: The predicted classes.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.floating]:
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
        X = check__data(X=X, all_features=self.all_features)

        X = self.__preprocess(X)

        distributions = np.array([
            self.__get_distribution(self.tree_.root, point) for _, point in X.iterrows()
        ])

        return distributions / distributions.sum(axis=1, keepdims=True)

    def predict_log_proba(self, X: pd.DataFrame) -> NDArray[np.floating]:
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
        epsilon = 1e-10
        return np.log(y_pred_proba + epsilon)

    def __preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna(self._feature_na_filler)

    def __get_distribution(
        self,
        node: TreeNode,
        point: pd.Series,
    ) -> NDArray[np.integer]:

        if node.is_leaf:
            return node.distribution

        if pd.isna(point[node.split_feature]):
            if self.feature_na_mode[node.split_feature] == "include_all":
                distribution = np.array([0, 0, 0], dtype="int")
                for child in node.childs:
                    distribution += self.__get_distribution(child, point)
                return distribution
            else:  # "include_best"
                child = node.childs[node.child_na_index]
                return self.__get_distribution(child, point)

        elif node.split_type == "numerical":
            threshold = float(node.childs[0].feature_value[0][3:])
            if point[node.split_feature] <= threshold:
                return self.__get_distribution(node.childs[0], point)
            else:
                return self.__get_distribution(node.childs[1], point)

        else:  # "categorical" | "rank"
            for child in node.childs:
                if point[node.split_feature] in child.feature_value:
                    return self.__get_distribution(child, point)
            else:
                distribution = np.array([0, 0, 0], dtype="int")
                for child in node.childs:
                    distribution += self.__get_distribution(child, point)
                return distribution

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> float | np.floating:
        """Returns the accuracy metric."""
        X, y = check__data(X=X, y=y, all_features=self.all_features)
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

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
            root=self.tree_.root,
            show_impurity=show_impurity,
            show_num_samples=show_num_samples,
            show_distribution=show_distribution,
            show_label=show_label,
            **kwargs,
        )

        return graph
