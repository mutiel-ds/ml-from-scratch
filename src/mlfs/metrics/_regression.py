"""Regression metrics."""

import numpy as np

from mlfs._typing import ArrayLike, FloatArray
from mlfs.utils.validation import check_array

__all__ = ["mean_squared_error", "r2_score"]


def _check_targets(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    true = check_array(y_true, ensure_2d=False)
    pred = check_array(y_pred, ensure_2d=False)
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred have different shapes: {true.shape} != {pred.shape}."
        )
    return true, pred


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Mean squared error (MSE).

    .. math:: \mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    mse : float
        Non-negative; 0 means a perfect fit.
    """
    true, pred = _check_targets(y_true, y_pred)
    return float(np.mean((true - pred) ** 2))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Coefficient of determination, :math:`R^2`.

    .. math::
        R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    r2 : float
        1.0 is a perfect fit, 0.0 matches always predicting the mean, and it
        can be negative for models worse than that. If ``y_true`` is constant,
        returns 1.0 for a perfect prediction and 0.0 otherwise (the same
        convention as scikit-learn).
    """
    true, pred = _check_targets(y_true, y_pred)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot
