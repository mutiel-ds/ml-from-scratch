"""Ordinary least squares linear regression."""

from typing import Literal, Self

from mlfs._typing import ArrayLike, FloatArray
from mlfs.base import BaseRegressor, check_is_fitted
from mlfs.utils.validation import check_array, check_n_features

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
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol}.")

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the model. Not implemented yet (added in the next step)."""
        self._check_params()
        raise NotImplementedError

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
