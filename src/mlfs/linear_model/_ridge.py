"""Ridge regression (L2-regularized least squares)."""

import warnings
from typing import Literal, Self

import numpy as np

from mlfs._typing import ArrayLike, FloatArray
from mlfs.base import BaseRegressor, check_is_fitted
from mlfs.exceptions import ConvergenceWarning
from mlfs.utils.validation import check_array, check_n_features, check_X_y

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

    def _solve(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Solve the regularized normal equation with ``np.linalg.solve``."""
        n_features = X.shape[1]
        gram = X.T @ X + self.alpha * np.eye(n_features)
        try:
            coef: FloatArray = np.linalg.solve(gram, X.T @ y)
        except np.linalg.LinAlgError as exc:
            # Only reachable for alpha=0 (or tiny alpha) with collinear features.
            raise ValueError(
                "The regularized normal equation is singular. Use alpha > 0 or "
                "solver='lstsq'."
            ) from exc
        return coef

    def _lstsq(self, X: FloatArray, y: FloatArray) -> FloatArray:
        r"""Solve the augmented least squares problem with SVD.

        Stacking :math:`\sqrt{\alpha} I` under ``X`` and zeros under ``y``
        turns the penalty into extra residuals, so the ridge solution is the
        ordinary least squares solution of the stacked system. It avoids
        forming :math:`X^\top X` and also works for ``alpha=0``.
        """
        n_features = X.shape[1]
        X_aug = np.vstack([X, np.sqrt(self.alpha) * np.eye(n_features)])
        y_aug = np.concatenate([y, np.zeros(n_features)])
        coef: FloatArray = np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]
        return coef

    def _gradient_descent(self, X: FloatArray, y: FloatArray) -> tuple[FloatArray, int]:
        r"""Minimize the regularized mean squared loss by gradient descent.

        .. math::

            L(w) = \frac{1}{2n} \left( \lVert Xw - y \rVert_2^2
            + \alpha \lVert w \rVert_2^2 \right), \qquad
            \nabla L(w) = \frac{1}{n} \left( X^\top (Xw - y) + \alpha w \right).

        The gradient is evaluated ``max_iter + 1`` times so that the solution
        reached by the last update is also checked against ``tol``. Starting
        from ``w = 0`` keeps the result deterministic. Returns the weights and
        the number of updates performed.
        """
        n_samples = X.shape[0]
        coef = np.zeros(X.shape[1])

        # A learning rate that is too large overflows to inf/NaN; that is
        # reported below as an error, so NumPy's warnings would be noise.
        with np.errstate(over="ignore", invalid="ignore"):
            n_iter = 0
            while True:
                gradient = (X.T @ (X @ coef - y) + self.alpha * coef) / n_samples
                if np.linalg.norm(gradient) < self.tol:
                    return coef, n_iter
                if n_iter == self.max_iter:
                    break  # out of updates; the last solution was just checked
                coef = coef - self.learning_rate * gradient
                n_iter += 1
                if not np.all(np.isfinite(coef)):
                    raise ValueError(
                        "Gradient descent diverged: the weights are not finite. "
                        "Use a smaller learning_rate or scale the features."
                    )

        warnings.warn(
            f"Gradient descent did not converge within max_iter={self.max_iter} "
            f"iterations. Increase max_iter or learning_rate, or scale the "
            "features.",
            ConvergenceWarning,
            stacklevel=3,
        )
        return coef, self.max_iter

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the ridge model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : Ridge
            The fitted estimator.

        Raises
        ------
        ValueError
            If a hyperparameter or the data is invalid, or if the normal
            equation is singular (``solver="solve"`` with ``alpha=0`` and
            collinear features).
        """
        self._check_params()
        X_checked, y_checked = check_X_y(X, y)

        # Centering gives the (unpenalized) intercept for free and never
        # touches the caller's arrays: the subtractions create new ones.
        if self.fit_intercept:
            X_offset = X_checked.mean(axis=0)
            y_offset = float(y_checked.mean())
        else:
            X_offset = np.zeros(X_checked.shape[1])
            y_offset = 0.0
        X_centered = X_checked - X_offset
        y_centered = y_checked - y_offset

        if self.solver == "gradient_descent":
            coef, self.n_iter_ = self._gradient_descent(X_centered, y_centered)
        else:
            if self.solver == "solve":
                coef = self._solve(X_centered, y_centered)
            else:
                coef = self._lstsq(X_centered, y_centered)
            # A previous gradient descent fit must not leave a stale count.
            vars(self).pop("n_iter_", None)

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
