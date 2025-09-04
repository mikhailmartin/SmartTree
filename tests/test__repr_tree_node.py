def test_repr_tree_node(root_node):
    assert repr(root_node) == (
        "TreeNode(node_number=0, num_samples=595, distribution=[199, 199, 197],"
        " impurity=0.37, label='доброкачественная опухоль')"
    )
