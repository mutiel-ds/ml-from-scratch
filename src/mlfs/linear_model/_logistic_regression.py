"""Binary logistic regression (L2-regularized, fitted by gradient descent)."""

from typing import Any, Self

import numpy as np

from mlfs._typing import ArrayLike, NDArray
from mlfs.base import BaseClassifier
from mlfs.utils._optimize import check_gradient_descent_params

__all__ = ["LogisticRegression"]


class LogisticRegression(BaseClassifier):
    r"""Binary logistic regression with L2 regularization.

    Models the probability of the positive class (the second of the sorted
    class labels) as

    .. math::

        P(y = 1 \mid x) = \sigma(x^\top w + b),
        \qquad \sigma(z) = \frac{1}{1 + e^{-z}} .

    The parameters are found by minimizing the average log-loss (binary
    cross-entropy) plus an L2 penalty on the weights:

    .. math::

        L(w, b) = \frac{1}{n} \left[
        -\sum_{i=1}^{n} y_i \log p_i + (1 - y_i) \log (1 - p_i)
        + \frac{\alpha}{2} \lVert w \rVert_2^2
        \right],
        \qquad p_i = \sigma(x_i^\top w + b).

    The intercept :math:`b` is deliberately not penalized. The loss is convex
    (its Hessian is :math:`\frac{1}{n} X^\top \mathrm{diag}(p_i (1 - p_i)) X
    + \frac{\alpha}{n} I`, which is positive semidefinite), so gradient
    descent reaches the global minimum, but there is no closed form because
    setting the gradient to zero is a nonlinear equation in :math:`w`.

    Unlike in linear regression, centering the data cannot remove the
    intercept, so :math:`(w, b)` is optimized as a single parameter vector.

    On linearly separable data the unpenalized loss has no minimizer (the
    weights grow without bound), so keep ``alpha > 0`` there. Gradient descent
    is also sensitive to the scale of the features: standardize them first.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength, finite and ``>= 0``. It weights the penalty
        next to the *sum* of the per-sample losses, as in
        :class:`~mlfs.linear_model.Ridge`. It equals ``1 / C`` in
        scikit-learn's parametrization. ``0`` disables the penalty.
    fit_intercept : bool, default=True
        Whether to fit an intercept. If False, the decision boundary passes
        through the origin.
    learning_rate : float, default=0.1
        Step size of gradient descent.
    max_iter : int, default=1000
        Maximum number of gradient descent steps.
    tol : float, default=1e-6
        Gradient descent stops when the norm of the gradient falls below this
        value.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def _check_params(self) -> None:
        """Raise ``ValueError`` if a hyperparameter has an invalid value."""
        # Checking isfinite first also rejects NaN, which would slip through a
        # plain ``alpha < 0`` comparison.
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError(f"alpha must be finite and >= 0, got {self.alpha}.")
        check_gradient_descent_params(self.learning_rate, self.max_iter, self.tol)

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the model (work in progress)."""
        self._check_params()
        raise NotImplementedError

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict class labels (work in progress)."""
        raise NotImplementedError
