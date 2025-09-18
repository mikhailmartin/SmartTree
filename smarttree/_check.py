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
    num_features=None,
    cat_features=None,
    rank_features=None,
    hierarchy=None,
    na_mode=None,
    num_na_mode=None,
    cat_na_mode=None,
    cat_na_filler=None,
    rank_na_mode=None,
    feature_na_mode=None,
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

    if num_features is not None:
        _check__num_features(num_features)
        param_name = "num_features"
        _check__features_contain_duplicates(param_name, num_features)

    if cat_features is not None:
        _check__cat_features(cat_features)
        param_name = "cat_features"
        _check__features_contain_duplicates(param_name, cat_features)

    if rank_features is not None:
        _check__rank_features(rank_features)

    if num_features is not None and cat_features is not None:
        param_name1, param_name2 = "num_features", "cat_features"
        _check__ambiguous(num_features, cat_features, param_name1, param_name2)

    if num_features is not None and rank_features is not None:
        param_name1, param_name2 = "num_features", "rank_features"
        _check__ambiguous(num_features, list(rank_features.keys()), param_name1, param_name2)

    if cat_features is not None and rank_features is not None:
        param_name1, param_name2 = "cat_features", "rank_features"
        _check__ambiguous(cat_features, list(rank_features.keys()), param_name1, param_name2)

    if hierarchy is not None:
        _check__hierarchy(hierarchy)

    if na_mode is not None:
        _check_na_mode(na_mode)

    if num_na_mode is not None:
        _check__num_na_mode(num_na_mode)

    if cat_na_mode is not None:
        _check__cat_na_mode(cat_na_mode)

    if cat_na_filler is not None:
        _check__cat_na_filler(cat_na_filler)

    if rank_na_mode is not None:
        _check__rank_na_mode(rank_na_mode)

    if feature_na_mode is not None:
        _check__feature_na_mode(feature_na_mode)


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


def _check__num_features(num_features):
    if isinstance(num_features, list):
        for num_feature in num_features:
            if not isinstance(num_feature, str):
                raise ValueError(
                    "If `num_features` is a list, it must consists of strings."
                    f" The element {num_feature} of the list isnt a string."
                )
    elif not isinstance(num_features, str):
        raise ValueError(
            "`num_features` must be a string or list of strings."
            f" The current value of `num_features` is {num_features!r}."
        )


def _check__cat_features(cat_features):
    if isinstance(cat_features, list):
        for cat_feature in cat_features:
            if not isinstance(cat_feature, str):
                raise ValueError(
                    "If `cat_features` is a list, it must consists of strings."
                    f" The element {cat_feature} of the list isnt a string."
                )
    elif not isinstance(cat_features, str):
        raise ValueError(
            "`cat_features` must be a string or list of strings."
            f" The current value of `cat_features` is {cat_features!r}."
        )


def _check__rank_features(rank_features):
    if not isinstance(rank_features, dict):
        raise ValueError(
            "`rank_features` must be a dictionary"
            " {rang feature name: list of its ordered values}."
        )
    for rank_feature, value_list in rank_features.items():
        if not isinstance(rank_feature, str):
            raise ValueError(
                "Keys in `rank_features` must be a strings."
                f" The key {rank_feature} isnt a string."
            )
        if not isinstance(value_list, list):
            raise ValueError(
                "Values in `rank_features` must be lists."
                f" The value {value_list} of the key {rank_feature} isnt a list."
            )


def _check__features_contain_duplicates(param_name, features):
    if isinstance(features, list) and len(features) != len(set(features)):
        raise ValueError(f"`{param_name}` contains duplicates.")


def _check__ambiguous(features1, features2, param_name1, param_name2):
    set_features1 = {features1} if isinstance(features1, str) else set(features1)
    set_features2 = {features2} if isinstance(features2, str) else set(features2)
    intersection = set_features1 & set_features2
    if intersection:
        raise ValueError(
            "Following feature names are ambiguous, they are defined in both"
            f" '{param_name1}' and '{param_name2}': {intersection}."
        )


def _check__hierarchy(hierarchy):
    common_message = (
        "`hierarchy` must be a dictionary"
        " {opening feature: opened feature / list of opened features}."
    )

    if not isinstance(hierarchy, dict):
        raise ValueError(
            f"{common_message} The current value of `hierarchy` is {hierarchy!r}."
        )
    for key, value in hierarchy.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{common_message} Value {key!r} of opening feature isnt a string."
            )
        if not isinstance(value, (str, list)):
            raise ValueError(
                f"{common_message} Value {value} of opened feature(s) isnt a string"
                " (list of strings)."
            )
        if isinstance(value, list):
            for elem in value:
                if not isinstance(elem, str):
                    raise ValueError(
                        f"{common_message} Value {elem} of opened feature isnt a string."
                    )


def _check_na_mode(na_mode):
    if na_mode not in ("include_all", "include_best"):
        raise ValueError(
            "`na_mode` must be Literal['include_all', 'include_best']."
            f" The current value of `na_mode` is {na_mode!r}."
        )


def _check__num_na_mode(num_na_mode):
    if num_na_mode not in ("min", "max", "include_all", "include_best"):
        raise ValueError(
            "`num_na_mode` must be Literal['min', 'max', 'include_all', 'include_best']."
            f" The current value of `num_na_mode` is {num_na_mode!r}."
        )


def _check__cat_na_mode(cat_na_mode):
    if cat_na_mode not in ("as_category", "include_all", "include_best"):
        raise ValueError(
            "`cat_na_mode` must be Literal['as_category', 'include_all', 'include_best']."
            f" The current value of `cat_na_mode` is {cat_na_mode!r}."
        )


def _check__cat_na_filler(cat_na_filler):
    if not isinstance(cat_na_filler, str):
        raise ValueError(
            "`cat_na_filler` must be a string."
            f" The current value of `cat_na_filler` is {cat_na_filler!r}."
        )


def _check__rank_na_mode(rank_na_mode):
    if rank_na_mode not in ("include_all", "include_best"):
        raise ValueError(
            "`rank_na_mode` must be Literal['include_all', 'include_best']."
            f" The current value of `rank_na_mode` is {rank_na_mode!r}."
        )


def _check__feature_na_mode(feature_na_mode):
    if not isinstance(feature_na_mode, dict):
        raise ValueError(
            "`feature_na_mode` must be a dictionary {feature name: NA mode}."
            f" The current value of `feature_na_mode` is {feature_na_mode!r}."
        )
    for key, value in feature_na_mode.items():
        if not isinstance(key, str):
            raise ValueError(
                "Keys in `feature_na_mode` must be a strings."
                f" The key {key} isnt a string."
            )
        if value not in ("min", "max", "as_category", "include_all", "include_best"):
            raise ValueError(
                "Values in `feature_na_mode` must be "
                "Literal['min', 'max', 'as_category' 'include_all', 'include_best']."
                f" The current value of `na_mode` for `feature` {key!r} is {value!r}."
            )


def check__data(
    X,
    *,
    y=None,
    num_features=None,
    cat_features=None,
    rank_features=None,
    all_features=None,
):
    _check__X(X)
    X = X.copy()

    if y is not None:
        _check__y(y)

    if X is not None and y is not None:
        _check__X_and_y(X, y)

    if num_features is not None:
        _check__num_features_in(X, num_features)

    if cat_features is not None:
        _check__cat_features_in(X, cat_features)

    if rank_features is not None:
        _check__rank_features_in(X, rank_features)

    if all_features is not None:
        _check__all_features_in(X, all_features)

    if y is not None:
        return X, y
    else:
        return X


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


def _check__num_features_in(X, num_features):
    for num_feature in num_features:
        if num_feature not in X.columns:
            raise ValueError(
                f"`num_features` contain feature '{num_feature}',"
                " which isnt present in the training data."
            )
        if not pd.api.types.is_numeric_dtype(X[num_feature]):
            try:
                X[num_feature] = pd.to_numeric(X[num_feature])
            except ValueError:
                raise ValueError(
                    f"`num_features` contain feature '{num_feature}',"
                    " which isnt numeric or convertable to numeric."
                )


def _check__cat_features_in(X, cat_features):
    for cat_feature in cat_features:
        if cat_feature not in X.columns:
            raise ValueError(
                f"`cat_features` contain feature {cat_feature!r},"
                " which isnt present in the training data."
            )


def _check__rank_features_in(X, rank_features):
    for rank_feature in rank_features.keys():
        if rank_feature not in X.columns:
            raise ValueError(
                f"`rank_features` contain feature {rank_feature!r},"
                " which isnt present in the training data."
            )


def _check__all_features_in(X, all_features):

    fitted_features_set = set(all_features)
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
