"""
Кастомная реализация дерева решений, которая может работать с категориальными и
численными признаками.
"""
# TODO:
# Алгоритм предварительной сортировки
# Дополнение numerical_feature_names
# logging
# доделать 'as_category'
# описать листья дерева через правила и предиктить по ним
# поменять rank_feature_names как numerical
# feature_value в numerical_node
# узлы через именованные кортежи? (оптимизация по оперативке)
# raises
# None в аннотация типов
# раскрасить визуализацию дерева
# cat_nan_mod = 'include' and 'as_category'
# модульные, юнит тесты тесты
# min_weight_fraction_leaf
# совместимость с GridSearchCV (нужна picklable)

import math
from typing import Literal

from graphviz import Digraph
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from multi_split_decision_tree._checkers import (
    _check_init_params, _check_fit_params, _check_score_params)
from multi_split_decision_tree._tree_node import TreeNode
from multi_split_decision_tree._utils import (
    cat_partitions, get_thresholds, rank_partitions)


class MultiSplitDecisionTreeClassifier:
    """
    Дерево решений.

    Attributes:
        tree: базовый древовидный объект.
        class_names: отсортированный список классов.
        feature_names: список всех признаков, находившихся в обучающих данных.
        numerical_feature_names: список численных признаков.
        categorical_feature_names: список категориальных признаков.
    """
    def __init__(
        self,
        *,
        criterion: Literal['gini', 'entropy', 'log_loss'] = 'gini',
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_leaf_nodes: int | float = float('+inf'),
        min_impurity_decrease: float = .0,
        max_childs: int | float = float('+inf'),
        numerical_feature_names: list[str] | str | None = None,
        categorical_feature_names: list[str] | str | None = None,
        rank_feature_names: dict[str, list] | None = None,
        hierarchy: dict[str, str | list[str]] | None = None,
        numerical_nan_mode: Literal['include', 'min', 'max'] = 'min',
        categorical_nan_mode: Literal['include'] = 'include',
    ) -> None:
        _check_init_params(
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_leaf_nodes,
            min_impurity_decrease,
            max_childs,
            numerical_feature_names,
            categorical_feature_names,
            rank_feature_names,
            hierarchy,
            numerical_nan_mode,
            categorical_nan_mode,
        )

        self.__criterion = criterion

        match criterion:
            case 'gini':
                self.__impurity = self.__gini_index
            case 'entropy' | 'log_loss':
                self.__impurity = self.__entropy

        # критерии остановки ветвления
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__max_leaf_nodes = max_leaf_nodes
        self.__min_impurity_decrease = min_impurity_decrease
        # критерий ограничения ветвления
        self.__max_childs = max_childs

        # открытые для чтения атрибуты
        self.__root = None
        self.__graph = None
        self.__class_names = None
        self.__feature_names = None
        self.__feature_importances = {}

        # TODO: not DRY
        if numerical_feature_names is None:
            self.__numerical_feature_names = []
        elif isinstance(numerical_feature_names, str):
            self.__numerical_feature_names = [numerical_feature_names]
        elif isinstance(numerical_feature_names, list):
            self.__numerical_feature_names = numerical_feature_names

        if categorical_feature_names is None:
            self.__categorical_feature_names = []
        elif isinstance(categorical_feature_names, str):
            self.__categorical_feature_names = [categorical_feature_names]
        elif isinstance(categorical_feature_names, list):
            self.__categorical_feature_names = categorical_feature_names

        if rank_feature_names is None:
            self.__rank_feature_names = {}
        elif isinstance(rank_feature_names, dict):
            self.__rank_feature_names = rank_feature_names

        self.__hierarchy = hierarchy if hierarchy else {}

        self.__numerical_nan_mode = numerical_nan_mode
        self.__categorical_nan_mode = categorical_nan_mode

        self.__is_fitted = False

        self.__node_counter = 0
        self.__leaf_counter = 0

    def __repr__(self):
        repr_ = []

        # если значение параметра отличается от того, что задано по умолчанию, то
        # добавляем его в репрезентацию
        if self.__criterion != 'gini':
            repr_.append(f'criterion={self.__criterion}')
        if self.__max_depth:
            repr_.append(f'max_depth={self.__max_depth}')
        if self.__min_samples_split != 2:
            repr_.append(f'min_samples_split={self.__min_samples_split}')
        if self.__min_samples_leaf != 1:
            repr_.append(f'min_samples_split={self.__min_samples_split}')
        if self.__max_leaf_nodes != float('+inf'):
            repr_.append(f'max_leaf_nodes={self.__max_leaf_nodes}')
        if self.__min_impurity_decrease != .0:
            repr_.append(f'min_impurity_decrease={self.__min_impurity_decrease}')
        if self.__max_childs != float('+inf'):
            repr_.append(f'max_childs={self.__max_childs}')
        if self.__numerical_feature_names:
            repr_.append(f'numerical_feature_names={self.__numerical_feature_names}')
        if self.__categorical_feature_names:
            repr_.append(f'categorical_feature_names={self.__categorical_feature_names}')
        if self.__rank_feature_names:
            repr_.append(f'rank_feature_names={self.__rank_feature_names}')
        if self.__hierarchy:
            repr_.append(f'hierarchy={self.__hierarchy}')
        if self.__numerical_nan_mode != 'min':
            repr_.append(f'numerical_nan_mode={self.__numerical_nan_mode}')
        if self.__categorical_nan_mode != 'include':
            repr_.append(f'categorical_nan_mode={self.__categorical_nan_mode}')

        return (
            f'{self.__class__.__name__}({", ".join(repr_)})'
        )

    @property
    def tree(self) -> TreeNode:
        if not self.__is_fitted:
            raise BaseException

        return self.__root

    @property
    def class_names(self) -> list[str]:
        if not self.__is_fitted:
            raise BaseException

        return self.__class_names

    @property
    def feature_names(self) -> list[str]:
        if not self.__is_fitted:
            raise BaseException

        return self.__feature_names

    @property
    def numerical_feature_names(self) -> list[str]:
        return self.__numerical_feature_names

    @property
    def categorical_feature_names(self) -> list[str]:
        return self.__categorical_feature_names

    @property
    def rank_feature_names(self) -> dict[str, list[str]]:
        return self.__rank_feature_names

    @property
    def feature_importances(self) -> dict[str, float]:
        return self.__feature_importances

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Обучает дерево решений.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
        """
        _check_fit_params(X, y, self)

        # до конца обучения инкапсулируем X и y
        self.X = X.copy()
        self.y = y.copy()
        # технический атрибут
        self.splittable_leaf_nodes = []

        self.__feature_names = X.columns.tolist()
        self.__class_names = sorted(y.unique())

        if isinstance(self.__min_samples_split, float):
            self.__min_samples_split = math.ceil(self.__min_samples_split * X.shape[0])
        if isinstance(self.__min_samples_leaf, float):
            self.__min_samples_leaf = math.ceil(self.__min_samples_leaf * X.shape[0])

        # инициализируем feature_importances всеми признаками и дефолтным значением 0
        for feature_name in self.__feature_names:
            self.__feature_importances[feature_name] = 0

        if not self.__numerical_feature_names:
            self.__numerical_feature_names = X.select_dtypes('number').columns.tolist()
        if not self.__categorical_feature_names:
            self.__categorical_feature_names = (
                X.select_dtypes(include=['object', 'category']).columns.tolist())

        match self.__numerical_feature_names:
            case 'min':
                for num_feature_name in self.__numerical_feature_names:
                    X[num_feature_name].fillna(X[num_feature_name].min(), inplace=True)
            case 'max':
                for num_feature_name in self.__numerical_feature_names:
                    X[num_feature_name].fillna(X[num_feature_name].max(), inplace=True)
            case 'as_category':
                for num_feature_name in self.__numerical_feature_names:
                    # если в признаке есть пропуски
                    if X[num_feature_name].isna().sum():
                        miss_feature_name = f'miss_{num_feature_name}'
                        X[miss_feature_name] = X[num_feature_name].isna()
                        # добавляем новое правило разбиений
                        self.__hierarchy[miss_feature_name] = num_feature_name

        hierarchy = self.__hierarchy.copy()
        available_feature_names = X.columns.tolist()
        # удаляем те признаки, которые пока не могут рассматриваться
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
            and self.__leaf_counter < self.__max_leaf_nodes
        ):

            # сортируем по листья по убыванию прироста информативности при лучшем
            # разбиении листа
            self.splittable_leaf_nodes = sorted(
                self.splittable_leaf_nodes,
                key=lambda x: x._best_split[0],
                reverse=True,
            )

            best_node = self.splittable_leaf_nodes.pop(0)
            (
                inf_gain,
                split_type,
                split_feature_name,
                feature_values,
                child_masks,
            ) = best_node._best_split

            self.__feature_importances[split_feature_name] += inf_gain

            for child_mask, feature_value in zip(child_masks, feature_values):
                # добавляем открывшиеся признаки
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
                    self.splittable_leaf_nodes.append(child_node)

            best_node.is_leaf = False
            best_node.split_type = split_type
            best_node.split_feature_name = split_feature_name
            self.__leaf_counter -= 1

        del self.X
        del self.y
        del self.splittable_leaf_nodes

        self.__is_fitted = True

    def __is_splittable(self, node: TreeNode) -> bool:
        """Проверяет, может ли узел дерева быть разделён."""
        if self.__max_depth and node._depth >= self.__max_depth:
            return False

        if node.samples < self.__min_samples_split:
            return False

        if node.impurity == 0:
            return False

        best_split_results = self.__find_best_split(node._mask, node._available_feature_names)
        inf_gain = best_split_results[0]
        if inf_gain < self.__min_impurity_decrease:
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
        """Создаёт узел дерева."""
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
        Находит лучшее разделение узла дерева, если оно существует.

        Args:
            parent_mask: булевая маска узла дерева.
            available_feature_names: список признаков, по которым допустимы разбиения.

        Returns:
            Кортеж `(inf_gain, split_type, split_feature_name, feature_values,
            child_masks)`.
              inf_gain: прирост информативности после разбиения.
              split_type: тип разбиения.
              split_feature_name: признак, по которому лучше всего разбивать входное
                множество.
              feature_values: значения признаков, соответствующие дочерним
                подмножествам.
              child_masks: булевые маски дочерних узлов.
        """
        best_inf_gain = float('-inf')
        best_split_type = None
        best_split_feature_name = None
        best_feature_values = None
        best_child_masks = None
        for split_feature_name in available_feature_names:
            if split_feature_name in self.__numerical_feature_names:
                split_type = 'numerical'
                (
                    inf_gain,
                    feature_values,
                    child_masks,
                ) = self.__num_split(parent_mask, split_feature_name)
            elif split_feature_name in self.__categorical_feature_names:
                split_type = 'categorical'
                (
                    inf_gain,
                    feature_values,
                    child_masks,
                ) = self.__best_cat_split(parent_mask, split_feature_name)
            elif split_feature_name in self.__rank_feature_names:
                split_type = 'rank'
                (
                    inf_gain,
                    feature_values,
                    child_masks,
                ) = self.__best_rank_split(parent_mask, split_feature_name)
            else:
                print(self.__numerical_feature_names)
                print(self.__categorical_feature_names)
                print(self.__rank_feature_names)
                print(split_feature_name)
                assert False

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
        Находит лучшее разделение узла дерева по заданному численному признаку, если оно
        существует.

        Args:
            parent_mask: булевая маска узла дерева.
            split_feature_name: название заданного численного признака, по которому
              нужно найти лучшее разделение.

        Returns:
            Кортеж `(inf_gain, feature_values, child_masks)`.
              inf_gain: прирост информативности при разделении.
              feature_values: значения признаков, соответствующие дочерним
                подмножествам.
              child_masks: булевые маски дочерних узлов.
        """
        use_including_na = (
            self.__numerical_nan_mode == 'include'
            # и есть примеры с пропусками
            and (parent_mask & self.X[split_feature_name].isna()).sum()
        )

        if use_including_na:
            mask_notna = parent_mask & self.X[split_feature_name].notna()
            # если невозможно разделение по значению признака
            if mask_notna.sum() <= 1:
                return float('-inf'), None, None
            mask_na = parent_mask & self.X[split_feature_name].isna()

            points = self.X.loc[mask_notna, split_feature_name].to_numpy()
        else:
            points = self.X.loc[parent_mask, split_feature_name].to_numpy()

        thresholds = get_thresholds(points)

        best_inf_gain = float('-inf')
        best_feature_values = None
        best_child_masks = None
        for threshold in thresholds:
            mask_less = parent_mask & (self.X[split_feature_name] <= threshold)
            mask_more = parent_mask & (self.X[split_feature_name] > threshold)

            if use_including_na:
                mask_less = mask_less | mask_na
                mask_more = mask_more | mask_na

            if (
                mask_less.sum() < self.__min_samples_leaf
                or mask_more.sum() < self.__min_samples_leaf
            ):
                continue

            child_masks = [mask_less, mask_more]

            inf_gain = self.__information_gain(
                parent_mask, child_masks, nan_mode=self.__numerical_nan_mode)

            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                less_values = [f'<= {threshold}']
                more_values = [f'> {threshold}']
                best_feature_values = [less_values, more_values]
                best_child_masks = child_masks

        return best_inf_gain, best_feature_values, best_child_masks

    def __best_cat_split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
    ) -> tuple[float, list[list[str]] | None, list[pd.Series] | None]:
        """
        Разделяет входное множество по категориальному признаку наилучшим образом.

        Args:
            parent_mask: булева маска родительского узла.
            split_feature_name: признак, по которому нужно разделить входное множество.

        Returns:
            Кортеж `(inf_gain, feature_values, child_masks)`.
              inf_gain: прирост информативности при разделении.
              feature_values: значения признаков, соответствующие дочерним
                подмножествам.
              child_masks: булевы маски дочерних узлов.
        """
        available_feature_values = self.X.loc[parent_mask, split_feature_name].unique()
        # TODO nan_mode
        # если содержит NaN (float('nan'))
        if pd.isna(available_feature_values).any():
            available_feature_values = available_feature_values[~pd.isna(available_feature_values)]
        if len(available_feature_values) <= 1:
            return float('-inf'), None, None
        available_feature_values = sorted(available_feature_values)

        # получаем список всех возможных разбиений
        partitions = []
        for partition in cat_partitions(available_feature_values):
            # если разбиение - на самом деле не разбиение
            if len(partition) < 2:
                continue
            # ограничение ветвления
            if len(partition) > self.__max_childs:
                continue
            # если после разбиения количество листьев превысит ограничение
            if self.__leaf_counter + len(partition) > self.__max_leaf_nodes:
                continue

            partitions.append(partition)

        best_inf_gain = float('-inf')
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
        Разделяет входное множество по категориальному признаку согласно заданным
        значениям.

        Args:
            parent_mask: булевая маска родительского узла.
            split_feature_name: признак, по которому нужно разделить входное множество.
            feature_values: значения признаков, соответствующие дочерним подмножествам.

        Returns:
            Кортеж `(inf_gain, child_masks)`.
              inf_gain: прирост информативности при разделении.
              child_masks: булевые маски дочерних узлов.
        """
        mask_na = parent_mask & self.X[split_feature_name].isna()

        child_masks = []
        for list_ in feature_values:
            child_mask = parent_mask & (self.X[split_feature_name].isin(list_) | mask_na)
            if child_mask.sum() < self.__min_samples_leaf:
                return float('-inf'), None
            child_masks.append(child_mask)

        inf_gain = self.__information_gain(
            parent_mask, child_masks, nan_mode=self.__categorical_nan_mode)

        return inf_gain, child_masks

    def __best_rank_split(
        self,
        parent_mask: pd.Series,
        split_feature_name: str,
    ) -> tuple[float, list[list[str]], list[pd.Series]]:
        """Разделяет входное множество по ранговому признаку наилучшим образом."""
        available_feature_values = self.__rank_feature_names[split_feature_name]

        best_inf_gain = float('-inf')
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
        Разделяет входное множество по ранговому признаку согласно заданным значениям.
        """
        left_list_, right_list_ = feature_values

        mask_left = parent_mask & self.X[split_feature_name].isin(left_list_)
        mask_right = parent_mask & self.X[split_feature_name].isin(right_list_)

        if (
            mask_left.sum() < self.__min_samples_leaf
            or mask_right.sum() < self.__min_samples_leaf
        ):
            return float('-inf'), None

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
        Считает прирост информативности.

        Args:
            parent_mask: булевая маска родительского узла.
            child_masks: список булевых масок дочерних узлов.
            nan_mode: режим обработки пропусков.
              Если 'include', то подрубает нормализацию дочерних загрязнений.

        Returns:
            прирост информативности.

        References:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

        Формула в LaTeX:
            \text{Information Gain} = \frac{N_{\text{parent}}}{N} * \Big(\text{impurity}_{\text{parent}} - \sum^C_{i=1}{\frac{N_{\text{child}_i}}{N_{\text{parent}}}} * \text{impurity}_{\text{child}_i} \Big)
            где
            \text{Information Gain} - собственно прирост информативности;
            N - количество примеров во всём обучающем наборе;
            N_{\text{parent}} - количество примеров в родительском узле;
            \text{impurity}_{\text{parent}} - загрязнённость родительского узла;
            С - количество дочерних узлов;
            N_{\text{child}_i} - количество примеров в дочернем узле;
            \text{impurity}_{\text{child}_i} - загрязнённость дочернего узла.
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

        if nan_mode == 'include':
            norm_coef = N_parent / N_childs
            weighted_impurity_childs *= norm_coef

        local_information_gain = impurity_parent - weighted_impurity_childs

        information_gain = (N_parent / N) * local_information_gain

        return information_gain

    def __gini_index(self, mask: pd.Series) -> float:
        """
        Считает индекс джини в узле дерева.

        Формула индекса Джини в LaTeX:
            \text{Gini Index} = \sum^C_{i=1} p_i \times (1 - p_i)
            где
            \text{Gini Index} - собственно индекс Джини;
            C - общее количество классов;
            p_i - вероятность выбора примера с классом i.
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
        Считает энтропию в узле дерева.

        Формула энтропии в LaTeX:
        H = \log{\overline{N}} = \sum^N_{i=1} p_i \log{(1/p_i)} = -\sum^N_{i=1} p_i \log{p_i}
        где
        H - энтропия;
        \overline{N} - эффективное количество состояний;
        p_i - вероятность состояния системы.
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
        """Подсчитывает распределение точек данных по классам."""
        distribution = [
            (mask & (self.y == class_name)).sum()
            for class_name in self.__class_names
        ]

        return distribution

    def predict(self, X: pd.DataFrame | pd.Series) -> list[str] | str:
        """Предсказывает метки классов для точек данных в X."""
        if not self.__is_fitted:
            raise BaseException

        if isinstance(X, pd.DataFrame):
            y_pred = [self.predict(point) for _, point in X.iterrows()]
        elif isinstance(X, pd.Series):
            y_pred_proba, samples = self.__predict_proba(self.__root, X)
            y_pred = self.__class_names[y_pred_proba.argmax()]
        else:
            raise ValueError('X должен представлять собой pd.DataFrame или pd.Series.')

        return y_pred

    def predict_proba(self, X: pd.DataFrame | pd.Series) -> np.array:
        """TODO."""
        if not self.__is_fitted:
            raise BaseException

        if isinstance(X, pd.DataFrame):
            y_pred_proba = [self.predict_proba(point) for _, point in X.iterrows()]
        elif isinstance(X, pd.Series):
            y_pred_proba, samples = self.__predict_proba(self.__root, X)
        else:
            assert False

        return y_pred_proba

    def __predict_proba(
        self,
        node: TreeNode,
        point: pd.Series,
    ) -> tuple[np.ndarray, int]:
        """Предсказывает метку класса для точки данных."""
        # Если мы не дошли до листа
        if not node.is_leaf:
            # но оказались в узле, в котором правило разделения задано по некоторому
            # признаку, а точка данных в этом признаке содержит пропуск
            if pd.isna(point[node.split_feature_name]):
                # то идём в дочерние узлы за предсказаниями, а потом их взвешенно
                # усредняем.
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

            elif node.split_feature_name in self.__numerical_feature_names:
                # ищем ту ветвь, по которой нужно идти
                threshold = float(node.childs[0].feature_value[0][3:])
                if point[node.split_feature_name] <= threshold:
                    y_pred_proba, samples = self.__predict_proba(node.childs[0], point)
                elif point[node.split_feature_name] > threshold:
                    y_pred_proba, samples = self.__predict_proba(node.childs[1], point)
                else:
                    assert False

            elif (
                node.split_feature_name in self.__categorical_feature_names
                or node.split_feature_name in self.__rank_feature_names
            ):
                # ищем ту ветвь, по которой нужно идти
                for child in node.childs:
                    # если нашли
                    if child.feature_value == point[node.split_feature_name]:
                        y_pred_proba, samples = self.__predict_proba(child, point)
                        # то можно заканчивать пропуск
                        break
                else:
                    # если такой ветви нет TODO
                    distribution = np.array(node.distribution)
                    y_pred_proba = distribution / distribution.sum()
                    samples = node.samples

            else:
                assert False

        # Если мы дошли до листа
        else:
            distribution = np.array(node.distribution)
            y_pred_proba = distribution / distribution.sum()
            samples = node.samples

        return y_pred_proba, samples

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> float:
        """Возвращает метрику accuracy."""
        if not self.__is_fitted:
            raise BaseException

        _check_score_params(self, X, y)

        score = accuracy_score(y, self.predict(X), sample_weight=sample_weight)

        return score

    def get_params(
        self,
        deep: bool = True,  # реализован для sklearn.model_selection.GridSearchCV
    ) -> dict:
        """Возвращает параметры этого классификатора."""
        return {
            'criterion': self.__criterion,
            'max_depth': self.__max_depth,
            'min_samples_split': self.__min_samples_split,
            'min_samples_leaf': self.__min_samples_leaf,
            'max_leaf_nodes': self.__max_leaf_nodes,
            'min_impurity_decrease': self.__min_impurity_decrease,
            'max_childs': self.__max_childs,
            'numerical_feature_names': self.__numerical_feature_names,
            'categorical_feature_names': self.__categorical_feature_names,
            'rank_feature_names': self.__rank_feature_names,
            'hierarchy': self.__hierarchy,
            'numerical_nan_mode': self.__numerical_nan_mode,
            'categorical_nan_mode': self.__categorical_nan_mode,
        }

    def set_params(self, **params):
        """Задаёт параметры этому классификатору."""
        valid_params = self.get_params(deep=True)

        for param, value in params.items():
            if param not in valid_params:
                raise ValueError(
                    f'Недопустимый параметр {param} для дерева {self}. Проверьте список'
                    ' доступных параметров с помощью `estimator.get_params().keys()`.'
                )
            # TODO: _check_params
            # TODO: работает пока совпадают параметры и приватные атрибуты
            # TODO: почему через setattr()?
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
        Визуализирует дерево решений.

        Если указаны именованные параметры, сохраняет визуализацию в виде файла(ов).

        Args:
            rounded: скруглять ли углы у узлов (они в форме прямоугольника).
            show_impurity: показывать ли загрязнённость узла.
            show_num_samples: показывать ли количество точек в узле.
            show_distribution: показывать ли распределение точек по классам.
            show_label: показывать ли класс, к которому относится узел.
            **kwargs: аргументы для graphviz.Digraph.render.

        Returns:
            Объект класса Digraph, содержащий описание графовой структуры дерева для
            визуализации.
        """
        if not self.__is_fitted:
            raise BaseException

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
        Создаёт объект класса Digraph, содержащий описание графовой структуры дерева для
        визуализации.
        """
        node_attr = {'shape': 'box'}
        if rounded:
            node_attr['style'] = 'rounded'

        self.__graph = Digraph(name='дерево решений', node_attr=node_attr)
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
        Рекурсивно добавляет описание узла и его связь с родительским узлом
        (если имеется).
        """
        node_name = f'node {node.number}'

        node_content = [f'Узел {node.number}']
        if node.split_feature_name:
            node_content.append(f'{node.split_feature_name}')
        if show_impurity:
            node_content.append(f'{self.__criterion} = {node.impurity:.2f}')
        if show_num_samples:
            node_content.append(f'samples = {node.samples}')
        if show_distribution:
            node_content.append(f'distribution = {node.distribution}')
        if show_label:
            node_content.append(f'label = {node.label}')
        node_content = '\n'.join(node_content)

        self.__graph.node(name=node_name, label=node_content)

        if parent_name:
            # TODO
            edge_label = '\n'.join([str(i) for i in node.feature_value])
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
