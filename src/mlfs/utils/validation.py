"""Input validation helpers shared by every estimator.

Every public ``fit``/``predict``/``transform`` method validates its inputs with
these functions instead of trusting the caller, so all estimators reject bad
input in the same way and with the same error messages.
"""

from typing import Any, Literal, overload

import numpy as np

from mlfs._typing import ArrayLike, FloatArray, NDArray, RandomState

__all__ = [
    "check_X_y",
    "check_array",
    "check_n_features",
    "check_random_state",
]


def check_array(X: ArrayLike, *, ensure_2d: bool = True) -> FloatArray:
    """Convert ``X`` to a finite float64 array and check its shape.

    Parameters
    ----------
    X : array-like
        Input data, for example a nested list or a NumPy array.
    ensure_2d : bool, default=True
        If True, ``X`` must have shape ``(n_samples, n_features)``. If False,
        ``X`` must be one-dimensional.

    Returns
    -------
    X_checked : ndarray of float64
        A validated copy of (or view on) the input.

    Raises
    ------
    ValueError
        If ``X`` cannot be converted to floats, has the wrong number of
        dimensions, is empty, or contains NaN or infinite values.
    """
    try:
        X_checked = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Input could not be converted to float64: {exc}") from exc

    expected_ndim = 2 if ensure_2d else 1
    if X_checked.ndim != expected_ndim:
        raise ValueError(
            f"Expected a {expected_ndim}D array, got a {X_checked.ndim}D array "
            f"with shape {X_checked.shape}."
        )
    if X_checked.shape[0] == 0:
        raise ValueError("Input contains 0 samples; at least 1 is required.")
    if ensure_2d and X_checked.shape[1] == 0:
        raise ValueError("Input contains 0 features; at least 1 is required.")
    if not np.all(np.isfinite(X_checked)):
        raise ValueError("Input contains NaN or infinite values.")
    return X_checked


@overload
def check_X_y(
    X: ArrayLike, y: ArrayLike, *, y_numeric: Literal[True] = ...
) -> tuple[FloatArray, FloatArray]: ...
@overload
def check_X_y(
    X: ArrayLike, y: ArrayLike, *, y_numeric: Literal[False]
) -> tuple[FloatArray, NDArray[Any]]: ...
@overload
def check_X_y(
    X: ArrayLike, y: ArrayLike, *, y_numeric: bool
) -> tuple[FloatArray, NDArray[Any]]: ...
def check_X_y(
    X: ArrayLike, y: ArrayLike, *, y_numeric: bool = True
) -> tuple[FloatArray, NDArray[Any]]:
    """Validate a feature matrix and its target vector together.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target values.
    y_numeric : bool, default=True
        If True (regression), ``y`` is converted to finite float64. If False
        (classification), ``y`` keeps its dtype so labels such as strings
        are allowed.

    Returns
    -------
    X_checked : ndarray of float64 of shape (n_samples, n_features)
        The validated features.
    y_checked : ndarray of shape (n_samples,)
        The validated target.

    Raises
    ------
    ValueError
        If either input is invalid or they have different numbers of samples.
    """
    X_checked = check_array(X, ensure_2d=True)
    if y_numeric:
        y_checked: NDArray[Any] = check_array(y, ensure_2d=False)
    else:
        y_checked = np.asarray(y)
        if y_checked.ndim != 1:
            raise ValueError(
                f"Expected y to be a 1D array, got shape {y_checked.shape}."
            )

    if X_checked.shape[0] != y_checked.shape[0]:
        raise ValueError(
            f"X and y have inconsistent numbers of samples: "
            f"{X_checked.shape[0]} != {y_checked.shape[0]}."
        )
    return X_checked, y_checked


def check_n_features(X: FloatArray, n_features_in: int) -> None:
    """Check that ``X`` has as many features as the data seen during ``fit``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Already validated input data.
    n_features_in : int
        Number of features seen during ``fit``.

    Raises
    ------
    ValueError
        If the number of columns of ``X`` differs from ``n_features_in``.
    """
    if X.shape[1] != n_features_in:
        raise ValueError(
            f"X has {X.shape[1]} features, but the estimator was fitted with "
            f"{n_features_in} features."
        )


def check_random_state(seed: RandomState) -> np.random.Generator:
    """Turn a ``random_state`` hyperparameter into a NumPy ``Generator``.

    Parameters
    ----------
    seed : int, numpy.random.Generator or None
        ``None`` gives fresh OS entropy, an int gives a reproducible generator,
        and an existing ``Generator`` is returned unchanged.

    Returns
    -------
    rng : numpy.random.Generator
        The random number generator to use.
    """
    if seed is None or isinstance(seed, int | np.integer):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        f"random_state must be None, an int or a numpy.random.Generator, "
        f"got {type(seed).__name__}."
    )
