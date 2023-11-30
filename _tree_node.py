class TreeNode:
    """Узел дерева решений."""
    def __init__(
        self,
        number: int,
        samples: int,
        distribution: list[int],
        impurity: float,
        label: str,
        # технические атрибуты
        _mask,
        _hierarchy,
        _available_feature_names,
    ) -> None:
        self.number = number
        self.is_leaf = True
        self.split_type = None
        self.split_feature_name = None
        self.feature_value = None
        self.childs = []
        self.samples = samples
        self.distribution = distribution
        self.impurity = impurity
        self.label = label
        # технические атрибуты
        self._mask = _mask
        self._hierarchy = _hierarchy
        self._available_feature_names = _available_feature_names

    def __repr__(self) -> str:
        representation = [
            f'node_number={self.number}',
            f'samples={self.samples}',
            f'distribution={self.distribution}',
            f'impurity={self.impurity}',
            f'label={self.label}',
        ]

        return f'{self.__class__.__name__}({", ".join(representation)})'
