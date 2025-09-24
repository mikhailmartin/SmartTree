import numpy as np

from smarttree._cy_column_splitter import CyBaseColumnSplitter


def test__gini_index(dataset):

    cy_base_column_splitter = CyBaseColumnSplitter(dataset=dataset, criterion="gini")

    mask = dataset.y.apply(lambda x: True).values.astype(np.int8)

    gini_index = cy_base_column_splitter.gini_index(mask)
    assert gini_index == 0.6666591342419322


def test__entropy(dataset):

    cy_base_column_splitter = CyBaseColumnSplitter(dataset=dataset, criterion="entropy")

    mask = dataset.y.apply(lambda x: True).values.astype(np.int8)

    gini_index = cy_base_column_splitter.entropy(mask)
    assert gini_index == 1.584946181877191
