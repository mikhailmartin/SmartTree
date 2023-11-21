"""
Кастомная реализация дерева решений, которая может работать с категориальными и
численными признаками.
"""
from __future__ import annotations

import logging
from typing import Literal

import functools
import math

from graphviz import Digraph
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


logging.basicConfig(level=logging.DEBUG, filename='log.log', filemode='w')


def counter(function):
    """Декоратор-счётчик."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        return function(*args, **kwargs)
    wrapper.count = 0
    return wrapper


def categorical_partition(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in categorical_partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        # put `first` in its own subset
        yield [[first]] + smaller


def rank_partition(collection):
    for i in range(1, len(collection)):
        yield collection[:i], collection[i:]


def _check_init_params(
    max_depth,
    criterion,
    min_samples_split,
    min_samples_leaf,
    min_impurity_decrease,
    max_childs,
):
    if max_depth is not None and not isinstance(max_depth, int):
        raise ValueError('`max_depth` должен представлять собой int.')

    if criterion not in ['entropy', 'gini']:
        raise ValueError('Для `criterion` доступны значения "entropy" и "gini".')

    if not isinstance(min_samples_split, int) or min_samples_split <= 1:
        raise ValueError(
            '`min_samples_split` должен представлять собой int и '
            'быть строго больше 1.'
        )

    if not isinstance(min_samples_leaf, int) or min_samples_leaf <= 0:
        raise ValueError(
            '`min_samples_leaf` должен представлять собой int и '
            'быть строго больше 0.'
        )

    if not isinstance(min_impurity_decrease, float) or min_impurity_decrease < 0:
        raise ValueError(
            '`min_impurity_decrease` должен представлять собой float '
            'и быть неотрицательным.'
        )

    if min_samples_split < 2 * min_samples_leaf:
        raise ValueError(
            '`min_samples_split` должен быть строго в '
            '2 раза больше `min_samples_leaf`.'
        )

    if max_childs is not None and not isinstance(max_childs, int) or \
            isinstance(max_childs, int) and max_childs < 2:
        raise ValueError(
            '`max_childs` должен представлять собой int и быть строго больше 1.'
        )


def _check_fit_params(
    X, y,
    categorical_feature_names, rank_feature_names, numerical_feature_names,
    special_cases,
):
    if not isinstance(X, pd.DataFrame):
        raise ValueError('X должен представлять собой pd.DataFrame.')

    if not isinstance(y, pd.Series):
        raise ValueError('y должен представлять собой pd.Series.')

    if X.shape[0] != y.shape[0]:
        raise ValueError('X и y должны быть одной длины.')

    if not any((categorical_feature_names, rank_feature_names, numerical_feature_names)):
        raise ValueError(
            'Признаки должны быть отнесены хотя бы к одной из возможных групп '
            '(categorical_feature_names, rank_feature_names и numerical_feature_names).'
        )

    if categorical_feature_names is not None:
        if not isinstance(categorical_feature_names, dict):
            raise ValueError(
                '`categorical_feature_names` должен представлять собой словарь '
                '{название категориального признака: список возможных его значений}.'
            )
        for categorical_feature_name, value_list in categorical_feature_names.items():
            if not isinstance(categorical_feature_name, str):
                raise ValueError(
                    'Ключи в `categorical_feature_names` должны представлять собой'
                    f' строки. `{categorical_feature_names}` - не строка.'
                )
            if not isinstance(value_list, list):
                raise ValueError(
                    'Значения в `categorical_feature_names` должны представлять собой'
                    f' списки строк. Значение `{categorical_feature_name}:'
                    f' {value_list}` - не список.'
                )
            for value in value_list:
                if not isinstance(value, str):
                    raise ValueError(
                        'Значения в `categorical_feature_names` должны представлять'
                        f' собой списки строк. Значение `{categorical_feature_name}:'
                        f' {value_list}` - не список строк, `{value}` - не строка.'
                    )

    if rank_feature_names is not None:
        if not isinstance(rank_feature_names, dict):
            raise ValueError(
                '`rank_feature_names` должен представлять собой словарь '
                '{название рангового признака: упорядоченный список его значений}.'
            )
        for rank_feature_name, value_list in rank_feature_names.items():
            if not isinstance(rank_feature_name, str):
                raise ValueError(
                    'Ключи в `rank_feature_names` должны представлять собой строки. '
                    f'`{rank_feature_name}` - не строка.'
                )
            if not isinstance(value_list, list):
                raise ValueError(
                    'Значения в `rank_feature_names` должны представлять собой списки. '
                    f'Значение `{rank_feature_name}: {value_list}` - не список.'
                )

    if numerical_feature_names is not None:
        if not isinstance(numerical_feature_names, list):
            raise ValueError(
                '`numerical_feature_names` должен представлять собой список строк.'
            )
        for numerical_feature_name in numerical_feature_names:
            if not isinstance(numerical_feature_name, str):
                raise ValueError(
                    '`numerical_feature_names` должен представлять собой список строк. '
                    f'`{numerical_feature_name}` - не строка.'
                )

    if special_cases is not None:
        if not isinstance(special_cases, dict):
            raise ValueError(
                '`special_cases` должен представлять собой словарь '
                '{строки: либо строки, либо списки строк}.'
            )
        for key, value in special_cases.items():
            if not isinstance(key, str):
                raise ValueError(
                    'special_cases должен представлять собой словарь '
                    '{строки: либо строки, либо списки строк}.'
                )
            if not isinstance(value, (str, list)):
                raise ValueError(
                    'special_cases должен представлять собой словарь '
                    '{строки: либо строки, либо списки строк}.'
                )
            if isinstance(value, list):
                for elem in value:
                    if not isinstance(elem, str):
                        raise ValueError(
                            'special_cases должен представлять собой словарь '
                            '{строки: либо строки, либо списки строк}.'
                        )

    setted_feature_names = []
    if categorical_feature_names:
        for feature_name in categorical_feature_names:
            if feature_name not in X.columns:
                raise ValueError(
                    f'`categorical_feature_names` содержит признак {feature_name}, '
                    'которого нет в обучающих данных.'
                )
        setted_feature_names += categorical_feature_names
    if rank_feature_names:
        for feature_name in rank_feature_names.keys():
            if feature_name not in X.columns:
                raise ValueError(
                    f'`rank_feature_names` содержит признак {feature_name}, '
                    'которого нет в обучающих данных.'
                )
        setted_feature_names += list(rank_feature_names.keys())
    if numerical_feature_names:
        for feature_name in numerical_feature_names:
            if feature_name not in X.columns:
                raise ValueError(
                    f'`numerical_feature_names` содержит признак {feature_name}, '
                    'которого нет в обучающих данных.'
                )
        setted_feature_names += numerical_feature_names
    for feature_name in X.columns:
        if feature_name not in setted_feature_names:
            raise ValueError(
                f'Обучающие данные содержат признак `{feature_name}`, который не '
                'определён ни в `categorical_feature_names`, ни в'
                ' `rank_feature_names`, ни в `numerical_feature_names`.'
            )


class MultiSplitDecisionTreeClassifier:
    """
    Дерево решений.

    Attributes:
        feature_names: список всех признаков, находившихся в обучающих данных.
        class_names: отсортированный список классов.
        categorical_feature_names: список всех категориальных признаков, находившихся в
          обучающих данных.
        rank_feature_names: словарь {название рангового признака: упорядоченный список
          его значений}.
        numerical_feature_names: список всех численных признаков, находившихся в
          обучающих данных.
        feature_importances: словарь {название признака: его нормализованная
          значимость}.
    """
    def __init__(
        self,
        *,
        max_depth: int | None = None,
        criterion: Literal['gini', 'entropy'] = 'gini',
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = .0,
        max_childs: int | None = None,
    ) -> None:
        _check_init_params(
            max_depth,
            criterion,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            max_childs,
        )

        self.__max_depth = max_depth
        self.__criterion = criterion
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__min_impurity_decrease = min_impurity_decrease
        self.__max_childs = max_childs

        self.__feature_names = None
        self.__class_names = None
        self.__cat_feature_names = {}
        self.__rank_feature_names = {}
        self.__num_feature_names = []
        self.__tree = None
        self.__graph = None
        self.__feature_importances = None
        self.__total_samples_num = None

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def class_names(self):
        return self.__class_names

    @property
    def categorical_feature_names(self) -> dict[str, list[str]]:
        return self.__cat_feature_names

    @property
    def rank_feature_names(self) -> dict[str, list]:
        return self.__rank_feature_names

    @property
    def numerical_feature_names(self) -> list[str]:
        return self.__num_feature_names

    @property
    def feature_importances(self):
        total = 0
        for importance in self.__feature_importances.values():
            total += importance
        for feature in self.__feature_importances.keys():
            self.__feature_importances[feature] /= total

        return self.__feature_importances

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        categorical_feature_names: dict[str, list[str]] | None = None,
        rank_feature_names: dict[str, list] | None = None,
        numerical_feature_names: list | None = None,
        special_cases: dict[str, str | dict] | None = None,
    ) -> None:
        """
        Обучает дерево решений.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
            categorical_feature_names: словарь {название категориального признака:
              список возможных его значений}.
            rank_feature_names: словарь {название рангового признака: упорядоченный
              список его значений}.
            numerical_feature_names: список численных признаков.
            special_cases: словарь {признак, который должен быть первым: признак или
              список признаков, которые могут быть после}.
        """
        _check_fit_params(
            X, y,
            categorical_feature_names, rank_feature_names, numerical_feature_names,
            special_cases,
        )

        self.__feature_names = list(X.columns)
        self.__class_names = sorted(y.unique())
        if categorical_feature_names:
            self.__cat_feature_names = categorical_feature_names
        if rank_feature_names:
            self.__rank_feature_names = rank_feature_names
        if numerical_feature_names:
            self.__num_feature_names = numerical_feature_names

        self.__total_samples_num = X.shape[0]
        self.__feature_importances = dict.fromkeys(self.__feature_names, 0)

        available_feature_names = self.__feature_names.copy()
        # удаляем те признаки, которые пока не могут рассматриваться
        if special_cases:
            for value in special_cases.values():
                if isinstance(value, str):
                    available_feature_names.remove(value)
                elif isinstance(value, list):
                    for feature_name in value:
                        available_feature_names.remove(feature_name)
                else:
                    assert False

        self.__tree = self.__generate_node(
            X, y,
            parent_mask=y.apply(lambda x: True),
            feature_value=None,
            depth=1,
            available_feature_names=available_feature_names,
            special_cases=special_cases,
        )

    @counter
    def __generate_node(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        parent_mask: pd.Series,
        feature_value: list[str],
        depth: int,
        available_feature_names: list[str],
        special_cases: dict[str, str | dict] | None = None,
    ) -> Node:
        """
        Рекурсивная функция создания узлов дерева.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
            parent_mask: булевая маска родительского узла.
            feature_value: значение признака, по которому этот узел был сформирован.
            depth: глубина узла.
            available_feature_names: список доступных признаков для разбиения входного
              множества.
            special_cases: словарь {признак, который должен быть первым: признак,
              который может быть после}.

        Returns:
            узел дерева.
        """
        impurity = self.__impurity(y, parent_mask)
        samples_num = parent_mask.sum()
        distribution = self.__distribution(y, parent_mask)
        label = y[parent_mask].value_counts().index[0]

        childs = []
        feature = None
        if samples_num >= self.__min_samples_split and \
                (not self.__max_depth or (depth <= self.__max_depth)):
            feature, masks, feature_values, inf_gain = self.__split(
                X, y, parent_mask, available_feature_names)

            if feature:
                available_feature_names = available_feature_names.copy()

                self.__feature_importances[feature] += \
                    (samples_num / self.__total_samples_num) * inf_gain

                # добавление открывшихся признаков
                if special_cases:
                    special_cases = special_cases.copy()
                    if feature in special_cases.keys():
                        if isinstance(special_cases[feature], str):
                            available_feature_names.append(special_cases[feature])
                        elif isinstance(special_cases[feature], list):
                            available_feature_names.extend(special_cases[feature])
                        else:
                            assert False
                        special_cases.pop(feature)

                # рекурсивное создание потомков
                for mask, fv in zip(masks, feature_values):
                    child = self.__generate_node(
                        X, y, mask, fv, depth+1, available_feature_names, special_cases
                    )
                    childs.append(child)

        assert label is not None, 'label is None'

        logging.debug(
            f'feature: {feature};'
            f' feature_value: {feature_value}'
            f' impurity: {impurity}'
            f' samples_num: {samples_num}'
            f' distribution: {distribution}'
            f' label: {label}'
            f' childs: {childs}'
        )

        node = Node(
            feature, feature_value, impurity, samples_num, distribution, label, childs)

        return node

    def __distribution(self, y: pd.Series, mask: pd.Series) -> list[int]:
        """Подсчитывает распределение точек данных по классам."""
        distribution = [
            (mask & (y == class_name)).sum()
            for class_name in self.__class_names
        ]

        return distribution

    def __impurity(self, y: pd.Series, mask: pd.Series) -> float:
        """Считает загрязнённость для множества."""
        match self.__criterion:
            case 'entropy':
                impurity = self.__entropy(y, mask)
            case 'gini':
                impurity = self.__gini(y, mask)
            case _:
                assert False

        return impurity

    def __entropy(self, y: pd.Series, mask: pd.Series) -> float:
        """
        Считает энтропию в множестве.

        Формула энтропии в LaTeX:
        H = \log{\overline{N}} = \sum^N_{i=1} p_i \log{(1/p_i)} = -\sum^N_{i=1} p_i \log{p_i}
        где
        H - энтропия;
        \overline{N} - эффективное количество состояний;
        p_i - вероятность состояния системы.
        """
        n = mask.sum()  # количество точек в множестве

        entropy = 0
        for label in self.__class_names:  # перебор по классам
            m_i = (mask & (y == label)).sum()
            if m_i != 0:
                entropy -= (m_i / n) * math.log2(m_i / n)

        return entropy

    def __gini(self, y: pd.Series, mask: pd.Series) -> float:
        """
        Считает индекс Джини в множестве.

        Формула индекса Джини в LaTeX:
        G = \sum^C_{i=1} p_i \times (1 - p_i)
        где
        G - индекс Джини;
        C - общее количество классов;
        p_i - вероятность выбора точки с классом i.
        """
        n = mask.sum()  # количество точек в множестве

        gini = 0
        for label in self.__class_names:  # перебор по классам
            m_i = (mask & (y == label)).sum()
            p_i = m_i / n
            gini += p_i * (1 - p_i)

        return gini

    def __split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        parent_mask: pd.Series,
        available_feature_names: list[str],
    ) -> tuple[str, list[pd.Series], tuple, float]:
        """
        Разделяет входное множество наилучшим образом.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
            parent_mask: булевая маска родительского узла.
            available_feature_names: список доступных признаков для разбиения входного
              множества.

        Returns:
            Кортеж `(feature_name, masks, feature_values, inf_gain)`.
              feature_name: признак, по которому лучше всего разбивать входное
                множество.
              masks: булевые маски дочерних узлов.
              feature_values: значения признаков, соответствующие дочерним
                подмножествам.
              inf_gain: прирост информативности после разбиения.
        """
        best_feature_name = None
        best_masks = []
        best_feature_values = tuple()
        best_inf_gain = 0
        for feature_name in available_feature_names:
            if feature_name in self.__cat_feature_names:
                inf_gain, masks, feature_values = self.__best_categorical_split(
                    X, y, parent_mask, feature_name)
            elif feature_name in self.__rank_feature_names:
                inf_gain, masks, feature_values = self.__best_rank_split(
                    X, y, parent_mask, feature_name)
            elif feature_name in self.__num_feature_names:
                inf_gain, masks, feature_values = self.__numerical_split(
                    X, y, parent_mask, feature_name)
            else:
                assert False

            if inf_gain >= self.__min_impurity_decrease and inf_gain > best_inf_gain:
                best_feature_name = feature_name
                best_masks = masks
                best_feature_values = feature_values
                best_inf_gain = inf_gain

        for list_ in best_feature_values:
            assert isinstance(list_, list)

        return best_feature_name, best_masks, best_feature_values, best_inf_gain

    def __best_categorical_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        parent_mask: pd.Series,
        feature_name: str,
    ) -> tuple[float, list[pd.Series], tuple]:
        """
        Разделяет входное множество по категориальному признаку наилучшим образом.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
            parent_mask: булевая маска родительского узла.
            feature_name: признак, по которому нужно разделить входное множество.

        Returns:
            Кортеж `(inf_gain, masks, feature_values)`.
              inf_gain: прирост информативности при разделении.
              masks: булевые маски дочерних узлов.
              feature_values: значения признаков, соответствующие дочерним
                подмножествам.
        """
        best_inf_gain = 0
        best_child_masks = []
        best_feature_values = tuple()

        available_feature_values = set(X.loc[parent_mask, feature_name].tolist())
        if np.NaN in available_feature_values:
            available_feature_values.remove(np.NaN)
        if len(available_feature_values) <= 1:
            return best_inf_gain, best_child_masks, best_feature_values
        available_feature_values = sorted(list(available_feature_values))

        assert len(available_feature_values) != 0

        # получаем список всех возможных разбиений
        partitions = [tuple(i) for i in categorical_partition(available_feature_values)]
        partitions = partitions[1:]  # убираем вариант без разбиения
        partitions = sorted(partitions, key=len)
        if self.__max_childs:
            partitions = list(filter(lambda x: len(x) <= self.__max_childs, partitions))

        for feature_values in partitions:
            inf_gain, child_masks = self.__categorical_split(
                X, y, parent_mask, feature_name, feature_values)
            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_child_masks = child_masks
                best_feature_values = feature_values

        return best_inf_gain, best_child_masks, best_feature_values

    def __categorical_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        parent_mask: pd.Series,
        feature_name: str,
        feature_values: tuple,
    ) -> tuple[float, list[pd.Series]]:
        """
        Разделяет входное множество по категориальному признаку согласно заданным
        значениям.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
            parent_mask: булевая маска родительского узла.
            feature_name: признак, по которому нужно разделить входное множество.
            feature_values: значения признаков, соответствующие дочерним подмножествам.

        Returns:
            Кортеж `(inf_gain, masks)`.
              inf_gain: прирост информативности при разделении.
              masks: булевые маски дочерних узлов.
        """
        mask_na = parent_mask & X[feature_name].isna()

        child_masks = []
        for list_ in feature_values:
            child_mask = parent_mask & (X[feature_name].isin(list_) | mask_na)
            if child_mask.sum() < self.__min_samples_leaf:
                return 0, []
            child_masks.append(child_mask)

        inf_gain = self.__information_gain(y, parent_mask, child_masks)

        return inf_gain, child_masks

    def __best_rank_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        parent_mask: pd.Series,
        feature_name: str,
    ) -> tuple[float, list[pd.Series], tuple]:
        """Разделяет входное множество по ранговому признаку наилучшим образом."""
        available_feature_values = self.__rank_feature_names[feature_name]

        best_inf_gain = 0
        best_child_masks = []
        best_feature_values = tuple()
        for feature_values in rank_partition(available_feature_values):
            inf_gain, child_masks = self.__rank_split(
                X, y, parent_mask, feature_name, feature_values)
            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_child_masks = child_masks
                best_feature_values = feature_values

        return best_inf_gain, best_child_masks, best_feature_values

    def __rank_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        parent_mask: pd.Series,
        feature_name: str,
        feature_values: tuple[list[str], list[str]],
    ) -> tuple[float, list[pd.Series]]:
        """
        Разделяет входное множество по ранговому признаку согласно заданным значениям.
        """
        left_list_, right_list_ = feature_values

        mask_na = parent_mask & X[feature_name].isna()

        mask_left = parent_mask & X[feature_name].isin(left_list_)
        mask_right = parent_mask & X[feature_name].isin(right_list_)

        mask_left_na = mask_left | mask_na
        mask_right_na = mask_right | mask_na

        if mask_left_na.sum() < self.__min_samples_leaf or \
                mask_right_na.sum() < self.__min_samples_leaf:
            return 0, []

        child_masks = [mask_left_na, mask_right_na]

        inf_gain = self.__information_gain(y, parent_mask, child_masks)

        return inf_gain, child_masks

    def __numerical_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        parent_mask: pd.Series,
        feature_name: str,
    ) -> tuple[float, list[pd.Series], tuple]:
        """
        Разделяет входное множество по численному признаку, выбирая наилучший порог.

        Args:
            X: pd.DataFrame с точками данных.
            y: pd.Series с соответствующими метками.
            parent_mask: булевая маска родительского узла.
            feature_name: признак, по которому нужно разделить входное множество.

        Returns:
            Кортеж `(inf_gain, masks, feature_values)`.
              inf_gain: прирост информативности при разделении.
              masks: булевые маски дочерних узлов.
              feature_values: значения признаков, соответствующие дочерним
                подмножествам.
        """
        mask_na = parent_mask & X[feature_name].isna()
        mask_notna = parent_mask & X[feature_name].notna()

        points = sorted(X.loc[mask_notna, feature_name].tolist())
        thresholds = [(points[i] + points[i+1]) / 2 for i in range(len(points) - 1)]

        best_inf_gain = 0
        best_child_masks = []
        best_feature_values = tuple()
        for threshold in thresholds:
            mask_less = parent_mask & (X[feature_name] <= threshold)
            mask_more = parent_mask & (X[feature_name] > threshold)

            mask_less_na = mask_less | mask_na
            mask_more_na = mask_more | mask_na

            if mask_less_na.sum() < self.__min_samples_leaf or \
                    mask_more_na.sum() < self.__min_samples_leaf:
                continue

            child_masks = [mask_less_na, mask_more_na]

            inf_gain = self.__information_gain(y, parent_mask, child_masks)

            if best_inf_gain < inf_gain:
                best_inf_gain = inf_gain
                best_child_masks = child_masks
                best_feature_values = ([f'<= {threshold}'], [f'> {threshold}'])

        return best_inf_gain, best_child_masks, best_feature_values

    def __information_gain(
        self,
        y: pd.Series,
        parent_mask: pd.Series,
        child_masks: list[pd.Series],
    ) -> float:
        """
        Считает прирост информативности.

        Формула в LaTeX:
        Gain(A, Q) = H(A, S) -\sum\limits^q_{i=1} \frac{|A_i|}{|A|} H(A_i, S),
        где
        H - функция энтропии;
        A - множество точек данных;
        Q - метка данных;
        q - множество значений метки, т.е. классы;
        S - признак;
        A_i - множество элементов A, на которых Q имеет значение i.

        Args:
            y: pd.Series с метками родительского множества.
            parent_mask: булевая маска родительского узла.
            child_masks: булевые маски дочерних узлов.

        Returns:
            прирост информативности.
        """
        A = parent_mask.sum()

        second_term = 0
        for child_mask in child_masks:
            A_i = child_mask.sum()
            second_term += (A_i / A) * self.__impurity(y, child_mask)

        inf_gain = self.__impurity(y, parent_mask) - second_term

        assert isinstance(inf_gain, float)

        return inf_gain

    def get_params(
        self,
        deep: bool = True,  # реализован для sklearn.model_selection.GridSearchCV
    ) -> dict:
        """Возвращает параметры этого классификатора."""
        return {
            'max_depth': self.__max_depth,
            'criterion': self.__criterion,
            'min_samples_split': self.__min_samples_split,
            'min_samples_leaf': self.__min_samples_leaf,
            'min_impurity_decrease': self.__min_impurity_decrease,
        }

    def set_params(self, **params):
        """Задаёт параметры этому классификатору."""
        if not params:
            return self
        valid_params = self.get_params(deep=True)

        for param, value in params.items():
            if param not in valid_params:
                raise ValueError(
                    f'Недопустимый параметр {param} для дерева {self}. Проверьте список '
                    'доступных параметров с помощью `estimator.get_params().keys()`.'
                )

            setattr(self, param, value)
            valid_params[param] = value

        return self

    def predict(self, X: pd.DataFrame | pd.Series) -> list[str] | str:
        """Предсказывает метки классов для точек данных в X."""
        if isinstance(X, pd.DataFrame):
            y_pred = [self.predict(point) for _, point in X.iterrows()]
        elif isinstance(X, pd.Series):
            y_pred = self.__predict(self.__tree, X)
        else:
            raise ValueError('X должен представлять собой pd.DataFrame или pd.Series.')

        assert y_pred is not None, 'предсказывает None'

        return y_pred

    def __predict(self, node: Node, point: pd.Series) -> str:
        """Предсказывает метку класса для точки данных."""
        Y = None
        # если мы дошли до листа
        if node.feature is None:
            Y = node.label
            assert Y is not None, 'label оказался None'
        elif node.feature in self.__cat_feature_names | self.__rank_feature_names:
            # ищем ту ветвь, по которой нужно идти
            for child in node.childs:
                if child.feature_value == point[node.feature]:
                    Y = self.__predict(child, point)
                    break
            else:
                # если такой ветви нет
                if Y is None:
                    Y = node.label
        elif node.feature in self.__num_feature_names:
            # ищем ту ветвь, по которой нужно идти
            threshold = float(node.childs[0].feature_value[0][3:])
            if point[node.feature] <= threshold:
                Y = self.__predict(node.childs[0], point)
            elif point[node.feature] > threshold:
                Y = self.__predict(node.childs[1], point)
            else:
                assert False
        else:
            assert False, (
                'node.split_feature и не None, и не в `categorical_feature_names` и не в'
                '`numerical_feature_names`'
            )

        assert Y is not None, 'Y is None'

        return Y

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
            self.__tree,
            None,
            show_impurity,
            show_num_samples,
            show_distribution,
            show_label,
        )

    @counter
    def __add_node(
        self,
        node: Node,
        parent_name: str,
        show_impurity: bool,
        show_num_samples: bool,
        show_distribution: bool,
        show_label: bool,
    ) -> None:
        """
        Рекурсивно добавляет описание узла и его связь с родительским узлом
        (если имеется).
        """
        node_name = f'node{self.__add_node.count}'

        node_content = []
        if node.feature:
            node_content.append(f'{node.feature}')
        if show_impurity:
            node_content.append(f'{self.__criterion} = {node.impurity:.3f}')
        if show_num_samples:
            node_content.append(f'samples = {node.samples}')
        if show_distribution:
            node_content.append(f'distribution: {node.distribution}')
        if show_label:
            node_content.append(f'label = {node.label}')
        node_content = '\n'.join(node_content)

        self.__graph.node(name=node_name, label=node_content)
        if parent_name:
            if isinstance(node.feature_value, list):
                a = [str(i) for i in node.feature_value]
                node_label = '\n'.join(a)
            else:
                assert False, 'пришли сюда'
            self.__graph.edge(parent_name, node_name, label=node_label)

        for child in node.childs:
            self.__add_node(
                child,
                node_name,
                show_impurity,
                show_num_samples,
                show_distribution,
                show_label,
            )

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Возвращает точность по заданным тестовым данным и меткам."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X должен представлять собой pd.DataFrame.')

        if not isinstance(y, pd.Series):
            raise ValueError('y должен представлять собой pd.Series.')

        if X.shape[0] != y.shape[0]:
            raise ValueError('X и y должны быть одной длины.')

        score = accuracy_score(y, self.predict(X))

        return score


class Node:
    """Узел дерева решений."""
    def __init__(
        self,
        feature,
        feature_value,
        impurity,
        samples,
        distribution,
        label,
        childs,
    ):
        self.feature = feature
        self.feature_value = feature_value
        self.impurity = impurity
        self.samples = samples
        self.distribution = distribution
        self.label = label
        self.childs = childs
