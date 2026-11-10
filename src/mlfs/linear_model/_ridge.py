"""Ridge regression (L2-regularized least squares)."""

from typing import Literal, Self

import numpy as np

from mlfs._typing import ArrayLike, FloatArray
from mlfs.base import BaseRegressor

__all__ = ["Ridge"]

_SOLVERS = ("solve", "lstsq", "gradient_descent")


class Ridge(BaseRegressor):
    r"""Linear least squares with L2 regularization (ridge regression).

    Fits a linear model :math:`\hat{y} = Xw + b` by minimizing

    .. math::

        \min_{w, b} \; \lVert y - Xw - b \rVert_2^2
        + \alpha \lVert w \rVert_2^2 .

    The penalty shrinks the weights towards zero. The intercept :math:`b` is
    deliberately not penalized: otherwise adding a constant to ``y`` would
    change the slopes. Setting the gradient with respect to :math:`b` to zero
    gives :math:`b = \bar{y} - \bar{x}^\top w`, which is the same as fitting
    the weights on centered data :math:`X_c = X - \bar{x}` and
    :math:`y_c = y - \bar{y}`.

    Setting the gradient with respect to :math:`w` to zero gives the
    regularized normal equation

    .. math::

        (X_c^\top X_c + \alpha I)\, w = X_c^\top y_c .

    For :math:`\alpha > 0` the matrix is symmetric positive definite (its
    eigenvalues are :math:`\lambda_i + \alpha`), so the system always has a
    unique solution, even with collinear features. For :math:`\alpha = 0`
    the problem reduces to ordinary least squares. In the basis of the
    singular value decomposition of :math:`X_c`, each component of the
    solution is multiplied by :math:`\sigma_i^2 / (\sigma_i^2 + \alpha)`, so
    directions with small singular values are shrunk the most.

    Three solvers are available. None of them inverts a matrix explicitly.

    * ``"solve"`` solves the regularized normal equation with
      ``numpy.linalg.solve``. It is fast, but forming :math:`X_c^\top X_c`
      squares the condition number, and it fails for :math:`\alpha = 0` with
      collinear features.
    * ``"lstsq"`` solves the equivalent augmented least squares problem

      .. math::

          \min_w \left\lVert
          \begin{pmatrix} X_c \\ \sqrt{\alpha} I \end{pmatrix} w
          - \begin{pmatrix} y_c \\ 0 \end{pmatrix}
          \right\rVert_2^2

      with ``numpy.linalg.lstsq`` (SVD based). It is more stable and also
      handles :math:`\alpha = 0`.
    * ``"gradient_descent"`` minimizes
      :math:`\frac{1}{2n}(\lVert X_c w - y_c \rVert_2^2 + \alpha \lVert w
      \rVert_2^2)`, whose gradient is
      :math:`\frac{1}{n}(X_c^\top (X_c w - y_c) + \alpha w)`. The factor
      :math:`1/n` does not move the minimizer, so all solvers agree.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength, finite and ``>= 0``. Larger values shrink the
        weights more; ``0`` is ordinary least squares.
    fit_intercept : bool, default=True
        Whether to fit an intercept. If False, the model passes through the
        origin and the data is not centered.
    solver : {"solve", "lstsq", "gradient_descent"}, default="solve"
        How the weights are computed; see above.
    learning_rate : float, default=0.01
        Step size of gradient descent. Ignored by the other solvers.
    max_iter : int, default=1000
        Maximum number of gradient descent steps. Ignored by the other solvers.
    tol : float, default=1e-6
        Gradient descent stops when the norm of the gradient falls below this
        value. Ignored by the other solvers.

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
    """

    coef_: FloatArray
    intercept_: float
    n_features_in_: int
    n_iter_: int

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: Literal["solve", "lstsq", "gradient_descent"] = "solve",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def _check_params(self) -> None:
        """Raise ``ValueError`` if a hyperparameter has an invalid value."""
        # Checking isfinite first also rejects NaN, which would slip through a
        # plain ``alpha < 0`` comparison.
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError(f"alpha must be finite and >= 0, got {self.alpha}.")
        if self.solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {self.solver!r}.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol}.")

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the ridge model (not implemented yet)."""
        raise NotImplementedError

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict targets for ``X`` (not implemented yet)."""
        raise NotImplementedError
