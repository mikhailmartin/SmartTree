class TreeNode:
    """Decision Tree Node."""
    def __init__(
        self,
        number: int,
        samples: int,
        distribution: list[int],
        impurity: float,
        label: str,
        # technical attributes
        _depth,
        _mask,
        _hierarchy,
        _available_feature_names,

        is_leaf: bool = True,
        split_type: str | None = None,
        split_feature_name: str | None = None,
        feature_value=None,
        childs: list | None = None,
    ) -> None:
        self.number = number
        self.is_leaf = is_leaf
        self.split_type = split_type
        self.split_feature_name = split_feature_name
        self.feature_value = feature_value
        if childs is None:
            self.childs = []
        else:
            self.childs = childs
        self.samples = samples
        self.distribution = distribution
        self.impurity = impurity
        self.label = label
        # technical attributes
        self._depth = _depth
        self._mask = _mask
        self._hierarchy = _hierarchy
        self._available_feature_names = _available_feature_names

    def __repr__(self) -> str:
        representation = [
            f'node_number={self.number}',
            f'samples={self.samples}',
            f'distribution={self.distribution}',
            f'impurity={self.impurity}',
            f'label={self.label!r}',
        ]

        return f'{self.__class__.__name__}({", ".join(representation)})'
