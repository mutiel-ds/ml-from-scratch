"""Ordinary least squares linear regression."""

from typing import Literal, Self

import numpy as np

from mlfs._typing import ArrayLike, FloatArray
from mlfs.base import BaseRegressor, check_is_fitted
from mlfs.utils._optimize import check_gradient_descent_params, gradient_descent
from mlfs.utils.validation import check_array, check_n_features, check_X_y

__all__ = ["LinearRegression"]

_SOLVERS = ("lstsq", "gradient_descent")


class LinearRegression(BaseRegressor):
    r"""Ordinary least squares linear regression.

    Fits a linear model :math:`\hat{y} = Xw + b` by minimizing the sum of
    squared residuals

    .. math::

        \min_{w, b} \; \lVert Xw + b - y \rVert_2^2 .

    Setting the gradient to zero gives the normal equation
    :math:`X^\top X w = X^\top y`. It is never solved by inverting
    :math:`X^\top X`: forming that product squares the condition number of
    ``X``, and the matrix is singular when features are collinear. The
    ``"lstsq"`` solver instead uses ``numpy.linalg.lstsq`` (an SVD of ``X``),
    which also returns the minimum-norm solution for rank-deficient problems.

    The intercept is handled by centering: with column means
    :math:`\bar{x}` and :math:`\bar{y}`, the weights are fitted on
    :math:`X - \bar{x}` and :math:`y - \bar{y}`, and then
    :math:`b = \bar{y} - \bar{x}^\top w`.

    The ``"gradient_descent"`` solver minimizes the loss
    :math:`\frac{1}{2n} \lVert Xw - y \rVert_2^2` (on centered data) with
    steps :math:`w \leftarrow w - \eta \nabla_w`, where
    :math:`\nabla_w = \frac{1}{n} X^\top (Xw - y)`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept. If False, the model passes through the
        origin and the data is not centered.
    solver : {"lstsq", "gradient_descent"}, default="lstsq"
        ``"lstsq"`` solves the problem in closed form. ``"gradient_descent"``
        solves it iteratively.
    learning_rate : float, default=0.01
        Step size :math:`\eta` of gradient descent. Ignored by ``"lstsq"``.
    max_iter : int, default=1000
        Maximum number of gradient descent steps. Ignored by ``"lstsq"``.
    tol : float, default=1e-6
        Gradient descent stops when the norm of the gradient falls below this
        value. Ignored by ``"lstsq"``.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated weights.
    intercept_ : float
        Estimated intercept; 0.0 if ``fit_intercept=False``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_iter_ : int
        Number of gradient descent steps run. Only set by
        ``solver="gradient_descent"``.

    Examples
    --------
    >>> from mlfs.linear_model import LinearRegression
    >>> model = LinearRegression().fit([[0], [1], [2]], [1, 3, 5])
    >>> model.predict([[3]])
    array([7.])
    """

    coef_: FloatArray
    intercept_: float
    n_features_in_: int
    n_iter_: int

    def __init__(
        self,
        fit_intercept: bool = True,
        solver: Literal["lstsq", "gradient_descent"] = "lstsq",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def _check_params(self) -> None:
        """Raise ``ValueError`` if a hyperparameter has an invalid value."""
        if self.solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {self.solver!r}.")
        check_gradient_descent_params(self.learning_rate, self.max_iter, self.tol)

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearRegression
            The fitted estimator.

        Raises
        ------
        ValueError
            If a hyperparameter or the data is invalid.
        """
        self._check_params()
        X_checked, y_checked = check_X_y(X, y)

        # Centering gives the intercept for free and never touches the
        # caller's arrays: the subtractions below create new ones.
        if self.fit_intercept:
            X_offset = X_checked.mean(axis=0)
            y_offset = float(y_checked.mean())
        else:
            X_offset = np.zeros(X_checked.shape[1])
            y_offset = 0.0
        X_centered = X_checked - X_offset
        y_centered = y_checked - y_offset

        if self.solver == "lstsq":
            # SVD-based solve: stable, and the minimum-norm solution when
            # X is rank-deficient. rcond=None uses the modern default cutoff.
            coef = np.linalg.lstsq(X_centered, y_centered, rcond=None)[0]
            # A previous gradient descent fit of this object must not leave a
            # stale iteration count behind.
            vars(self).pop("n_iter_", None)
        else:
            n_samples = X_centered.shape[0]
            # Called directly from fit so the ConvergenceWarning points at the
            # user's fit(...) line.
            coef, self.n_iter_ = gradient_descent(
                lambda w: X_centered.T @ (X_centered @ w - y_centered) / n_samples,
                np.zeros(X_centered.shape[1]),
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
            )

        self.coef_ = coef
        self.intercept_ = float(y_offset - X_offset @ coef)
        self.n_features_in_ = X_checked.shape[1]
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict targets for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values :math:`Xw + b`.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        ValueError
            If ``X`` is invalid or has a different number of features than
            the data seen during ``fit``.
        """
        check_is_fitted(self)
        X_checked = check_array(X)
        check_n_features(X_checked, self.n_features_in_)
        return X_checked @ self.coef_ + self.intercept_
