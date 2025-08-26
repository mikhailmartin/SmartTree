from smarttree import TreeNode


def test_repr_tree_node():
    tn = TreeNode(
        number=0, samples=0, distribution=[1, 1, 1], impurity=1., label="test",
        depth=0, mask=None, hierarchy=None, available_feature_names=None,
    )

    assert repr(tn) == (
        "TreeNode(node_number=0, samples=0, distribution=[1, 1, 1], impurity=1.0,"
        " label='test')"
    )
