from smarttree import SmartDecisionTreeClassifier


def test__render(X, y):
    tree = SmartDecisionTreeClassifier()
    tree.fit(X, y)
    tree.render(
        rounded=True,
        show_impurity=True,
        show_num_samples=True,
        show_distribution=True,
        show_label=True,
    )
