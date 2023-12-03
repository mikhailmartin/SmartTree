# TODO:
# подумать над шаблоном current_value_message


import pandas as pd


def _check_init_params(
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
) -> None:
    """Проверяет параметры инициализации экземпляра класса дерева."""
    if criterion not in ['entropy', 'gini']:
        raise ValueError(
            'Для `criterion` доступны значения "entropy" и "gini".'
            f' Текущее значение `criterion` = {criterion}.'
        )

    if max_depth:
        if not isinstance(max_depth, int):
            raise ValueError(
                '`max_depth` должен представлять собой int.'
                f' Текущее значение `max_depth` = {max_depth}.'
            )

    if (
        not (isinstance(min_samples_split, (int, float)))
        or (isinstance(min_samples_split, int) and min_samples_split < 2)
        or (isinstance(min_samples_split, float) and 0 >= min_samples_split >= 1)
    ):
        raise ValueError(
            '`min_samples_split` должен представлять собой либо int и лежать в'
            ' диапазоне [2, +inf), либо float и лежать в диапазоне (0, 1).'
            f' Текущее значение `min_samples_split` = {min_samples_split}.'
        )

    if (
        not isinstance(min_samples_leaf, (int, float))
        or (isinstance(min_samples_leaf, int) and min_samples_leaf < 1)
        or (isinstance(min_samples_leaf, float) and 0 >= min_samples_leaf >= 1)
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

    if min_samples_split < 2 * min_samples_leaf:
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
            )
        for numerical_feature_name in numerical_feature_names:
            if not isinstance(numerical_feature_name, str):
                raise ValueError(
                    '`numerical_feature_names` должен представлять собой список строк.'
                    f' `{numerical_feature_name}` - не строка.'
                )

    if categorical_feature_names:
        if not isinstance(categorical_feature_names, list):
            raise ValueError(
                '`categorical_feature_names` должен представлять собой список строк.'
            )
        for categorical_feature_name in categorical_feature_names:
            if not isinstance(categorical_feature_name, str):
                raise ValueError(
                    '`categorical_feature_names` должен представлять собой список'
                    f' строк. `{categorical_feature_name}` - не строка.'
                )

    if rank_feature_names is not None:
        if not isinstance(rank_feature_names, dict):
            raise ValueError(
                '`rank_feature_names` должен представлять собой словарь'
                ' {название рангового признака: упорядоченный список его значений}.'
            )
        for rank_feature_name, value_list in rank_feature_names.items():
            if not isinstance(rank_feature_name, str):
                raise ValueError(
                    'Ключи в `rank_feature_names` должны представлять собой строки.'
                    f' `{rank_feature_name}` - не строка.'
                )
            if not isinstance(value_list, list):
                raise ValueError(
                    'Значения в `rank_feature_names` должны представлять собой списки.'
                    f' Значение `{rank_feature_name}: {value_list}` - не список.'
                )

    if hierarchy:
        if not isinstance(hierarchy, dict):
            raise ValueError(
                '`hierarchy` должен представлять собой словарь '
                '{строки: либо строки, либо списки строк}.'
            )
        for key, value in hierarchy.items():
            if not isinstance(key, str):
                raise ValueError(
                    '`hierarchy` должен представлять собой словарь '
                    '{строки: либо строки, либо списки строк}.'
                )
            if not isinstance(value, (str, list)):
                raise ValueError(
                    '`hierarchy` должен представлять собой словарь '
                    '{строки: либо строки, либо списки строк}.'
                )
            if isinstance(value, list):
                for elem in value:
                    if not isinstance(elem, str):
                        raise ValueError(
                            '`hierarchy` должен представлять собой словарь '
                            '{строки: либо строки, либо списки строк}.'
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
