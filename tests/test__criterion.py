import numpy as np

from smarttree._criterion import Entropy, Gini


def test__gini(dataset):
    gini_criterion = Gini(dataset)
    mask = np.ones(dataset.y.shape, dtype=np.int8)
    gini_index = gini_criterion.impurity(mask)
    assert gini_index == 0.6666591342419322


def test__entropy(dataset):
    entropy_criterion = Entropy(dataset)
    mask = np.ones(dataset.y.shape, dtype=np.int8)
    entropy = entropy_criterion.impurity(mask)
    assert entropy == 1.584946181877191
