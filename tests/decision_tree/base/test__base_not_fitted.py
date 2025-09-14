import pytest

from smarttree._exceptions import NotFittedError


@pytest.mark.parametrize(
    "property_name",
    ["tree_", "all_features", "feature_importances_"],
    ids=lambda param: str(param),
)
def test__not_fitted__property(concrete_smart_tree, property_name):
    with pytest.raises(NotFittedError):
        property_ = getattr(concrete_smart_tree, property_name)
