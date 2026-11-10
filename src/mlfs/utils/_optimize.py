"""Batch gradient descent shared by the iterative estimators.

Private module: estimators import it as ``mlfs.utils._optimize``. Each
estimator supplies only the gradient of its own loss; the stopping rule, the
divergence check and the convergence warning live here, so every iterative
solver in the package behaves the same way.
"""

import warnings
from collections.abc import Callable

import numpy as np

from mlfs._typing import FloatArray
from mlfs.exceptions import ConvergenceWarning

__all__ = ["check_gradient_descent_params", "gradient_descent"]


def check_gradient_descent_params(
    learning_rate: float, max_iter: int, tol: float
) -> None:
    """Validate the hyperparameters of :func:`gradient_descent`.

    Parameters
    ----------
    learning_rate : float
        Step size; must be ``> 0``.
    max_iter : int
        Maximum number of updates; must be ``>= 1``.
    tol : float
        Gradient-norm threshold; must be ``>= 0``.

    Raises
    ------
    ValueError
        If any of the values is out of range.
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}.")


def gradient_descent(
    gradient: Callable[[FloatArray], FloatArray],
    initial: FloatArray,
    *,
    learning_rate: float,
    max_iter: int,
    tol: float,
) -> tuple[FloatArray, int]:
    r"""Minimize a differentiable function by batch gradient descent.

    Starting from ``initial``, repeat the update

    .. math::

        \theta \leftarrow \theta - \eta \, \nabla L(\theta)

    until :math:`\lVert \nabla L(\theta) \rVert_2 <` ``tol`` or ``max_iter``
    updates have been made. The gradient is evaluated ``max_iter + 1`` times,
    so the parameters reached by the last update are also checked against
    ``tol``.

    Call it directly from the estimator's ``fit``: the warning uses
    ``stacklevel=3`` so that it points at the user's ``fit(...)`` line.

    Parameters
    ----------
    gradient : callable
        Maps parameters of shape ``(n_params,)`` to the gradient of the loss
        at those parameters, with the same shape.
    initial : ndarray of shape (n_params,)
        Starting parameters. Not modified.
    learning_rate : float
        Step size :math:`\eta`.
    max_iter : int
        Maximum number of updates.
    tol : float
        Stop when the Euclidean norm of the gradient falls below this value.

    Returns
    -------
    params : ndarray of shape (n_params,)
        The final parameters.
    n_iter : int
        The number of updates performed.

    Raises
    ------
    ValueError
        If the parameters stop being finite (the learning rate is too large).

    Warns
    -----
    ConvergenceWarning
        If ``max_iter`` updates were not enough to reach ``tol``.

    Examples
    --------
    Minimize :math:`L(\theta) = \frac{1}{2}(\theta - 3)^2`, whose gradient is
    :math:`\theta - 3`. With a step size of 1, one update reaches the minimum:

    >>> import numpy as np
    >>> from mlfs.utils._optimize import gradient_descent
    >>> gradient_descent(
    ...     lambda theta: theta - 3,
    ...     np.zeros(1),
    ...     learning_rate=1.0,
    ...     max_iter=10,
    ...     tol=1e-6,
    ... )
    (array([3.]), 1)
    """
    params = initial.copy()

    # A learning rate that is too large makes the parameters overflow to inf
    # and then NaN. That is reported below as an error, so NumPy's own
    # overflow warnings would only be noise.
    with np.errstate(over="ignore", invalid="ignore"):
        n_iter = 0
        while True:
            grad = gradient(params)
            if np.linalg.norm(grad) < tol:
                return params, n_iter
            if n_iter == max_iter:
                break  # out of updates; the last parameters were just checked
            params = params - learning_rate * grad
            n_iter += 1
            if not np.all(np.isfinite(params)):
                raise ValueError(
                    "Gradient descent diverged: the weights are not finite. "
                    "Use a smaller learning_rate or scale the features."
                )

    warnings.warn(
        f"Gradient descent did not converge within max_iter={max_iter} "
        "iterations. Increase max_iter or learning_rate, or scale the "
        "features.",
        ConvergenceWarning,
        stacklevel=3,
    )
    return params, max_iter
