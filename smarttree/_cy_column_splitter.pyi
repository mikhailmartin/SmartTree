import pandas as pd

from ._dataset import Dataset
from ._types import Criterion

class CyBaseColumnSplitter:

    def __init__(self, dataset: Dataset, criterion: Criterion) -> None:
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
