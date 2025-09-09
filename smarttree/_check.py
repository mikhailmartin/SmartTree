import pandas as pd


def check__params(
    *,
    criterion=None,
    max_depth=None,
    min_samples_split=None,
    min_samples_leaf=None,
    max_leaf_nodes=None,
    min_impurity_decrease=None,
    max_childs=None,
    numerical_feature_names=None,
    categorical_feature_names=None,
    rank_feature_names=None,
    hierarchy=None,
    numerical_na_mode=None,
    categorical_na_mode=None,
    categorical_na_filler=None,
):
    if criterion is not None:
        _check__criterion(criterion)

    if max_depth is not None:
        _check__max_depth(max_depth)

    if min_samples_split is not None:
        _check__min_samples_split(min_samples_split)

    if min_samples_leaf is not None:
        _check__min_samples_leaf(min_samples_leaf)

    if min_samples_split is not None and min_samples_leaf is not None:
        _check__min_samples_split__and__min_samples_leaf(
            min_samples_split, min_samples_leaf
        )

    if max_leaf_nodes is not None:
        _check__max_leaf_nodes(max_leaf_nodes)

    if min_impurity_decrease is not None:
        _check__min_impurity_decrease(min_impurity_decrease)

    if max_childs is not None:
        _check__max_childs(max_childs)

    if numerical_feature_names is not None:
        _check__numerical_feature_names(numerical_feature_names)

    if categorical_feature_names is not None:
        _check__categorical_feature_names(categorical_feature_names)

    if rank_feature_names is not None:
        _check__rank_feature_names(rank_feature_names)

    if hierarchy is not None:
        _check__hierarchy(hierarchy)

    if numerical_na_mode is not None:
        _check__numerical_na_mode(numerical_na_mode)

    if categorical_na_mode is not None:
        _check__categorical_na_mode(categorical_na_mode)

    if categorical_na_filler is not None:
        _check__categorical_na_filler(categorical_na_filler)


def _check__criterion(criterion):
    if criterion not in ("entropy", "gini", "log_loss"):
        raise ValueError(
            "`criterion` mist be Literal['entropy', 'log_loss', 'gini']."
            f" The current value of `criterion` is {criterion!r}."
        )


def _check__max_depth(max_depth):
    if not isinstance(max_depth, int) or max_depth <= 0:
        raise ValueError(
            "`max_depth` must be an integer and strictly greater than 0."
            f" The current value of `max_depth` is {max_depth!r}."
        )


def _check__min_samples_split(min_samples_split):
    if (
        not isinstance(min_samples_split, (int, float))
        or (
            isinstance(min_samples_split, int)
            and min_samples_split < 2
        )
        or (
            isinstance(min_samples_split, float)
            and (min_samples_split <= 0 or min_samples_split >= 1)
        )
    ):
        raise ValueError(
            "`min_samples_split` must be an integer and lie in the range [2, +inf),"
            " or float and lie in the range (0, 1)."
            f" The current value of `min_samples_split` is {min_samples_split!r}."
        )


def _check__min_samples_leaf(min_samples_leaf):
    if (
        not isinstance(min_samples_leaf, (int, float))
        or (isinstance(min_samples_leaf, int) and min_samples_leaf < 1)
        or (
            isinstance(min_samples_leaf, float)
            and (min_samples_leaf <= 0 or min_samples_leaf >= 1)
        )
    ):
        raise ValueError(
            "`min_samples_leaf` must be an integer and lie in the range [1, +inf),"
            " or float and lie in the range (0, 1)."
            f" The current value of `min_samples_leaf` is {min_samples_leaf!r}."
        )


def _check__min_samples_split__and__min_samples_leaf(
    min_samples_split, min_samples_leaf
) -> None:
    same_type = (
        (isinstance(min_samples_split, int) and isinstance(min_samples_leaf, int))
        or
        (isinstance(min_samples_split, float) and isinstance(min_samples_leaf, float))
    )

    if same_type and min_samples_split < 2 * min_samples_leaf:
        raise ValueError(
            "`min_samples_split` must be strictly 2 times greater than"
            " `min_samples_leaf`. Current values of `min_samples_split` is"
            f" {min_samples_split}, of `min_samples_leaf` is {min_samples_leaf}."
        )


def _check__max_leaf_nodes(max_leaf_nodes):
    if not isinstance(max_leaf_nodes, int) or max_leaf_nodes < 2:
        raise ValueError(
            "`max_leaf_nodes` must be an integer and strictly greater than 2."
            f" The current value of `max_leaf_nodes` is {max_leaf_nodes!r}."
        )


def _check__min_impurity_decrease(min_impurity_decrease):
    # TODO: could impurity_decrease be greater 1?
    if not isinstance(min_impurity_decrease, float) or min_impurity_decrease < 0:
        raise ValueError(
            "`min_impurity_decrease` must be float and non-negative."
            f" The current value of `min_impurity_decrease` is {min_impurity_decrease!r}."
        )


def _check__max_childs(max_childs) -> None:
    if not isinstance(max_childs, int) or max_childs < 2:
        raise ValueError(
            "`max_childs` must be integer and strictly greater than 2."
            f" The current value of `max_childs` is {max_childs!r}."
        )


def _check__numerical_feature_names(numerical_feature_names):
    if isinstance(numerical_feature_names, list):
        for numerical_feature_name in numerical_feature_names:
            if not isinstance(numerical_feature_name, str):
                raise ValueError(
                    "If `numerical_feature_names` is a list, it must consists of"
                    " strings."
                    f" The element {numerical_feature_name} of the list isnt a"
                    " string."
                )
    elif not isinstance(numerical_feature_names, str):
        raise ValueError(
            "`numerical_feature_names` must be a string or list of strings."
            f" The current value of `numerical_feature_names` is"
            f" {numerical_feature_names!r}."
        )


def _check__categorical_feature_names(categorical_feature_names):
    if isinstance(categorical_feature_names, list):
        for categorical_feature_name in categorical_feature_names:
            if not isinstance(categorical_feature_name, str):
                raise ValueError(
                    "If `categorical_feature_names` is a list, it must consists of"
                    " strings."
                    f" The element {categorical_feature_name} of the list isnt a"
                    " string."
                )
    elif not isinstance(categorical_feature_names, str):
        raise ValueError(
            "`categorical_feature_names` must be a string or list of strings."
            f" The current value of `categorical_feature_names` is"
            f" {categorical_feature_names!r}."
        )


def _check__rank_feature_names(rank_feature_names):
    if isinstance(rank_feature_names, dict):
        for rank_feature_name, value_list in rank_feature_names.items():
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
    else:
        raise ValueError(
            "`rank_feature_names` must be a dictionary"
            " {rang feature name: list of its ordered values}."
        )


def _check__hierarchy(hierarchy):
    common_message = (
        "`hierarchy` must be a dictionary"
        " {opening feature: opened feature / list of opened features}."
    )

    if isinstance(hierarchy, dict):
        for key, value in hierarchy.items():
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
    else:
        raise ValueError(
            common_message
            + f" The current value of `hierarchy` is {hierarchy!r}."
        )


def _check__numerical_na_mode(numerical_na_mode):
    if numerical_na_mode not in ("min", "max", "include_all", "include_best"):
        raise ValueError(
            "`numerical_na_mode` must be Literal['min', 'max', 'include_all', 'include_best']."
            f" The current value of `numerical_na_mode` is {numerical_na_mode!r}."
        )


def _check__categorical_na_mode(categorical_na_mode):
    if categorical_na_mode not in ("as_category", "include_all", "include_best"):
        raise ValueError(
            "`categorical_na_mode` must be Literal['as_category', 'include_all', 'include_best']."
            f" The current value of `categorical_na_mode` is {categorical_na_mode!r}."
        )


def _check__categorical_na_filler(categorical_na_filler):
    if not isinstance(categorical_na_filler, str):
        raise ValueError(
            "`categorical_na_filler` must be a string."
            f" The current value of `categorical_na_filler` is {categorical_na_filler!r}."
        )


def check__data(
    *,
    X=None,
    y=None,
    numerical_feature_names=None,
    categorical_feature_names=None,
    rank_feature_names=None,
    all_feature_names=None,
):
    if X is not None:
        _check__X(X)

    if y is not None:
        _check__y(y)

    if X is not None and y is not None:
        _check__X_and_y(X, y)

    if numerical_feature_names is not None:
        _check__numerical_feature_names_in(X, numerical_feature_names)

    if categorical_feature_names is not None:
        _check__categorical_feature_names_in(X, categorical_feature_names)

    if rank_feature_names is not None:
        _check__rank_feature_names_in(X, rank_feature_names)

    if all_feature_names is not None:
        _check_all_feature_names_in(X, all_feature_names)


def _check__X(X):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas.DataFrame.")


def _check__y(y):
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas.Series.")

    if y.isna().any():
        raise ValueError("y must not contain NA.")


def _check__X_and_y(X, y):
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must be the equal length.")


def _check__numerical_feature_names_in(X, numerical_feature_names):
    for numerical_feature_name in numerical_feature_names:
        if numerical_feature_name not in X.columns:
            raise ValueError(
                f"`numerical_feature_names` contain feature {numerical_feature_name!r},"
                " which isnt present in the training data."
            )


def _check__categorical_feature_names_in(X, categorical_feature_names):
    for categorical_feature_name in categorical_feature_names:
        if categorical_feature_name not in X.columns:
            raise ValueError(
                f"`categorical_feature_names` contain feature {categorical_feature_name!r},"
                " which isnt present in the training data."
            )


def _check__rank_feature_names_in(X, rank_feature_names):
    for rank_feature_name in rank_feature_names.keys():
        if rank_feature_name not in X.columns:
            raise ValueError(
                f"`rank_feature_names` contain feature {rank_feature_name!r},"
                " which isnt present in the training data."
            )


def _check_all_feature_names_in(X, all_feature_names):

    fitted_features_set = set(all_feature_names)
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
                output.append(f"- {name!r}")
            return "\n".join(output)

        if unexpected_names:
            message.append("Feature names unseen at fit time:")
            message.append(add_names(unexpected_names))

        if missing_names:
            message.append("Feature names seen at fit time, yet now missing:")
            message.append(add_names(missing_names))

        # TODO: same order of features

        raise ValueError("\n".join(message))
