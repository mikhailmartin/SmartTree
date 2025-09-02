class TreeNode:
    """Decision Tree Node."""
    def __init__(
        self,
        number: int,
        num_samples: int,
        distribution: list[int] = None,
        impurity: float = None,
        label: str = None,
        # technical attributes
        depth=None,
        mask=None,
        hierarchy=None,
        available_feature_names=None,

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
        self.num_samples = num_samples
        self.distribution = distribution
        self.impurity = impurity
        self.label = label
        # technical attributes
        self.depth = depth
        self.mask = mask
        self.hierarchy = hierarchy
        self.available_feature_names = available_feature_names

    def __repr__(self) -> str:
        representation = [
            f"node_number={self.number}",
            f"num_samples={self.num_samples}",
            f"distribution={self.distribution}",
            f"impurity={self.impurity}",
            f"label={self.label!r}",
        ]

        return f"{self.__class__.__name__}({', '.join(representation)})"
