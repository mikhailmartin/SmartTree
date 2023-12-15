# TODO:
# подумать над шаблоном current_value_message
# Текущее значение = string_type
# Стараться проверить сразу всё
# проверка y в __check_score_params


import pandas as pd


def _check_init_params(
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_leaf_nodes=float('+inf'),
    min_impurity_decrease=.0,
    max_childs=float('+inf'),
    numerical_feature_names=None,
    categorical_feature_names=None,
    rank_feature_names=None,
    hierarchy=None,
    numerical_nan_mode='min',
    categorical_nan_mode='include',
) -> None:
    """Проверяет параметры инициализации экземпляра класса дерева."""
    if criterion not in ['entropy', 'gini', 'log_loss']:
        raise ValueError(
            'Для `criterion` доступны значения "entropy", "gini" и "log_loss".'
            f' Текущее значение `criterion` = {criterion}.'
        )

    if max_depth:
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError(
                '`max_depth` должен представлять собой int и быть строго больше 0'
                f' Текущее значение `max_depth` = {max_depth}.'
            )

    if (
        not isinstance(min_samples_split, (int, float))
        or (isinstance(min_samples_split, int) and min_samples_split < 2)
        or (
            isinstance(min_samples_split, float)
            and (min_samples_split <= 0 or min_samples_split >= 1)
        )
    ):
        raise ValueError(
            '`min_samples_split` должен представлять собой либо int и лежать в'
            ' диапазоне [2, +inf), либо float и лежать в диапазоне (0, 1).'
            f' Текущее значение `min_samples_split` = {min_samples_split}.'
        )

    if (
        not isinstance(min_samples_leaf, (int, float))
        or (isinstance(min_samples_leaf, int) and min_samples_leaf < 1)
        or (
            isinstance(min_samples_leaf, float)
            and (min_samples_leaf <= 0 or min_samples_leaf >= 1)
        )
    ):
        raise ValueError(
            '`min_samples_leaf` должен представлять собой либо int и лежать в'
            ' диапазоне [1, +inf), либо float и лежать в диапазоне (0, 1).'
            f' Текущее значение `min_samples_leaf` = {min_samples_leaf}.'
        )

    if (
        not (isinstance(max_leaf_nodes, int) or max_leaf_nodes == float('+inf'))
        or max_leaf_nodes < 2
    ):
        raise ValueError(
            '`max_leaf_nodes` должен представлять собой int и быть строго больше 2.'
            f' Текущее значение `max_leaf_nodes` = {max_leaf_nodes}.'
        )

    if not isinstance(min_impurity_decrease, float) or min_impurity_decrease < 0:
        raise ValueError(
            '`min_impurity_decrease` должен представлять собой float'
            ' и быть неотрицательным.'
            f' Текущее значение `min_impurity_decrease` = {min_impurity_decrease}.'
        )

    if (
        (isinstance(min_samples_split, int) and isinstance(min_samples_leaf, int))
        and min_samples_split < 2 * min_samples_leaf
    ):
        raise ValueError(
            '`min_samples_split` должен быть строго в 2 раза больше `min_samples_leaf`.'
            f' Текущее значение `min_samples_split` = {min_samples_split},'
            f' `min_samples_leaf` = {min_samples_leaf}.'
        )

    if (
        not (isinstance(max_childs, int) or max_childs == float('+inf'))
        or max_childs < 2
    ):
        raise ValueError(
            '`max_childs` должен представлять собой int и быть строго больше 2.'
            f' Текущее значение `max_childs` = {max_childs}.'
        )

    if numerical_feature_names:
        if not isinstance(numerical_feature_names, (list, str)):
            raise ValueError(
                '`numerical_feature_names` должен представлять собой список строк'
                ' либо строку.'
                f' Текущее значение `numerical_feature_names` = {numerical_feature_names}.'
            )
        for numerical_feature_name in numerical_feature_names:
            if not isinstance(numerical_feature_name, str):
                raise ValueError(
                    'Если `numerical_feature_names` представляет собой список,'
                    ' то должен содержать строки.'
                    f' Элемент списка {numerical_feature_name} - не строка.'
                )

    if categorical_feature_names:
        if not isinstance(categorical_feature_names, (list, str)):
            raise ValueError(
                '`categorical_feature_names` должен представлять собой список строк'
                ' либо строку.'
                f' Текущее значение `categorical_feature_names` = {categorical_feature_names}.'
            )
        for categorical_feature_name in categorical_feature_names:
            if not isinstance(categorical_feature_name, str):
                raise ValueError(
                    'Если `categorical_feature_names` представляет собой список,'
                    ' то должен содержать строки.'
                    f' Элемент списка {categorical_feature_name} - не строка.'
                )

    if rank_feature_names:
        if not isinstance(rank_feature_names, dict):
            raise ValueError(
                '`rank_feature_names` должен представлять собой словарь'
                ' {название рангового признака: упорядоченный список его значений}.'
            )
        for rank_feature_name, value_list in rank_feature_names.items():
            if not isinstance(rank_feature_name, str):
                raise ValueError(
                    'Ключи в `rank_feature_names` должны представлять собой строки.'
                    f' {rank_feature_name} - не строка.'
                )
            if not isinstance(value_list, list):
                raise ValueError(
                    'Значения в `rank_feature_names` должны представлять собой списки.'
                    f' Значение {rank_feature_name} = {value_list} - не список.'
                )

    if hierarchy:
        if not isinstance(hierarchy, dict):
            raise ValueError(
                '`hierarchy` должен представлять собой словарь {открывающий признак:'
                ' открывающийся признак / список открывающихся признаков}.'
                f' Текущее значение `hierarchy` = {hierarchy}.'
            )
        for key, value in hierarchy.items():
            if not isinstance(key, str):
                raise ValueError(
                    '`hierarchy` должен представлять собой словарь {открывающий признак:'
                    ' открывающийся признак / список открывающихся признаков}.'
                    f' Значение открывающего признака {key} - не строка.'
                )
            if not isinstance(value, (str, list)):
                raise ValueError(
                    '`hierarchy` должен представлять собой словарь {открывающий признак:'
                    ' открывающийся признак / список открывающихся признаков}.'
                    f' Значение открывающегося признака(ов) {value} - не строка (список строк).'
                )
            if isinstance(value, list):
                for elem in value:
                    if not isinstance(elem, str):
                        raise ValueError(
                            '`hierarchy` должен представлять собой словарь {открывающий признак:'
                            ' открывающийся признак / список открывающихся признаков}.'
                            f' Значение открывающегося признака(ов) {value} - не строка (список строк).'
                        )

    if numerical_nan_mode not in ['include', 'min', 'max']:
        raise ValueError(
            'Для `numerical_nan_mode` доступны значения "include", "min" и "max".'
            f' Текущее значение `numerical_nan_mode` = {numerical_nan_mode}.'
        )

    if categorical_nan_mode not in ['include']:
        raise ValueError(
            'Для `categorical_nan_mode` доступно значение "include".'
            f' Текущее значение `categorical_nan_mode` = {categorical_nan_mode}.'
        )


def _check_fit_params(X, y, tree):
    if not isinstance(X, pd.DataFrame):
        raise ValueError('X должен представлять собой pd.DataFrame.')

    if not isinstance(y, pd.Series):
        raise ValueError('y должен представлять собой pd.Series.')

    if X.shape[0] != y.shape[0]:
        raise ValueError('X и y должны быть одной длины.')

    setted_feature_names = []
    if tree.categorical_feature_names:
        for cat_feature_name in tree.categorical_feature_names:
            if cat_feature_name not in X.columns:
                raise ValueError(
                    f'`categorical_feature_names` содержит признак {cat_feature_name},'
                    ' которого нет в обучающих данных.'
                )
        setted_feature_names += tree.categorical_feature_names
    if tree.rank_feature_names:
        for rank_feature_name in tree.rank_feature_names.keys():
            if rank_feature_name not in X.columns:
                raise ValueError(
                    f'`rank_feature_names` содержит признак {rank_feature_name},'
                    ' которого нет в обучающих данных.'
                )
        setted_feature_names += list(tree.rank_feature_names.keys())
    if tree.numerical_feature_names:
        for num_feature_name in tree.numerical_feature_names:
            if num_feature_name not in X.columns:
                raise ValueError(
                    f'`numerical_feature_names` содержит признак {num_feature_name},'
                    ' которого нет в обучающих данных.'
                )
        setted_feature_names += tree.numerical_feature_names
    # for feature_name in X.columns:
    #     if feature_name not in setted_feature_names:
    #         raise ValueError(
    #             f'Обучающие данные содержат признак `{feature_name}`, который не'
    #             ' определён ни в `categorical_feature_names`, ни в'
    #             ' `rank_feature_names`, ни в `numerical_feature_names`.'
    #         )

def _check_score_params(tree, X, y):
    if not isinstance(X, pd.DataFrame):
        raise ValueError('X должен представлять собой pd.DataFrame.')

    if not isinstance(y, pd.Series):
        raise ValueError('y должен представлять собой pd.Series.')

    if X.shape[0] != y.shape[0]:
        raise ValueError('X и y должны быть одной длины.')

    fitted_feature_names = tree.feature_names
    X_feature_names = X.columns
    if len(fitted_feature_names) != len(X_feature_names):
        message = (
            'Названия признаков должны совпадать с теми,'
            ' что были переданы во время обучения.\n'
        )
        fitted_feature_names_set = set(fitted_feature_names)
        X_feature_names_set = set(X_feature_names)

        unexpected_names = sorted(X_feature_names_set - fitted_feature_names_set)
        missing_names = sorted(fitted_feature_names_set - X_feature_names_set)

        def add_names(names):
            output = ''
            max_n_names = 5
            for i, name in enumerate(names):
                if i >= max_n_names:
                    output += '- ...\n'
                    break
                output += f'- {name}\n'
            return output

        if unexpected_names:
            message += 'Названия признаков, что не были переданы во время обучения:\n'
            message += add_names(unexpected_names)

        if missing_names:
            message += (
                'Названия признаков, что были переданы во время обучения,'
                ' но сейчас отсутствуют:\n'
            )
            message += add_names(missing_names)

        raise ValueError(message)

    # TODO: проверка y
