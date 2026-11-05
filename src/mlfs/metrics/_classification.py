"""Classification metrics."""

import numpy as np

from mlfs._typing import ArrayLike

__all__ = ["accuracy_score"]


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Fraction of predictions that exactly match the true labels.

    .. math:: \mathrm{accuracy} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[y_i = \hat{y}_i]

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels. Any comparable dtype (ints, strings, ...).
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    accuracy : float
        Value in ``[0, 1]``.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must be 1D with the same shape, got "
            f"{true.shape} and {pred.shape}."
        )
    if true.shape[0] == 0:
        raise ValueError("y_true contains 0 samples; at least 1 is required.")
    return float(np.mean(true == pred))
