import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._dataset import Dataset
from ._types import ClassificationCriterionType

class CyBaseColumnSplitter:

    criterion: ClassificationCriterionType

    def __init__(
        self,
        dataset: Dataset,
        criterion: ClassificationCriterionType,
    ) -> None:
        ...

    def information_gain(
        self,
        parent_mask: pd.Series,
        child_masks: list[pd.Series],
        normalize: bool = False,
    ) -> float:
        r"""
        Calculates information gain of the split.

        Parameters:
            parent_mask: pd.Series
              boolean mask of parent node.
            child_masks: pd.Series
              list of boolean masks of child nodes.
            normalize: bool, default=False
              if True, normalizes information gain by split factor to handle
              unbalanced splits. Uses child node counts for normalization.

        Returns:
            float: information gain.

        Formula in LaTeX:
            \begin{align*}
            \text{Information Gain} =
            \frac{N_{\text{parent}}}{N} \cdot
            \Biggl( & \text{impurity}_{\text{parent}} - \\
            & \sum^C_{i=1} \frac{N_{\text{child}_i}}{N_{\text{parent}}}
            \cdot \text{impurity}_{\text{child}_i} \Biggr)
            \end{align*}
            where:
            \begin{itemize}
                \item $\text{Information Gain}$ — information gain;
                \item $N$ — number of samples in entire training set;
                \item $N_{\text{parent}}$ — number of samples in parent node;
                \item $\text{impurity}_{\text{parent}}$ — parent node impurity;
                \item $C$ — number of child nodes;
                \item $N_{\text{child}_i}$ — number of samples in child node;
                \item $\text{impurity}_{\text{child}_i}$ — child node impurity.
            \end{itemize}
        """
        ...

    def gini_index(self, mask: NDArray[np.int8]) -> float:
        r"""
        Calculates Gini index in a tree node.

        Gini index formula in LaTeX:
            \text{Gini Index} = 1 - \sum^C_{i=1} p_i^2
            where
            C - total number of classes;
            p_i - the probability of choosing a sample with class i.
        """
        ...

    def entropy(self, mask: NDArray[np.int8]) -> float:
        r"""
        Calculates entropy in a tree node.

        Entropy formula in LaTeX:
        H = \log{\overline{N}} = \sum^N_{i=1} p_i \log{(1/p_i)} = -\sum^N_{i=1} p_i \log{p_i}
        where
        H - entropy;
        \overline{N} - effective number of states;
        p_i - probability of the i-th system state.
        """
        ...
