"""Binary logistic regression (L2-regularized, fitted by gradient descent)."""

from typing import Any, Self

import numpy as np

from mlfs._typing import ArrayLike, FloatArray, NDArray
from mlfs.base import BaseClassifier, check_is_fitted
from mlfs.utils._optimize import check_gradient_descent_params, gradient_descent
from mlfs.utils.validation import check_array, check_n_features, check_X_y

__all__ = ["LogisticRegression"]


def _sigmoid(z: FloatArray) -> FloatArray:
    r"""Compute the logistic function :math:`\sigma(z) = 1 / (1 + e^{-z})`.

    Must not overflow or emit warnings for very large ``|z|`` (for example
    ``z = -1000``, where ``np.exp(1000)`` is ``inf``).
    """
    e = np.exp(-np.abs(z))
    return np.where(z < 0, e / (1 + e), 1 / (1 + e))


def _gradient(
    params: FloatArray,
    X: FloatArray,
    y: FloatArray,
    *,
    alpha: float,
    fit_intercept: bool,
) -> FloatArray:
    r"""Gradient of the penalized average log-loss with respect to ``params``.

    Parameters
    ----------
    params : ndarray of shape (n_features + 1,) or (n_features,)
        The weights :math:`w`, followed by the intercept :math:`b` when
        ``fit_intercept`` is True.
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,)
        Encoded labels, ``0.0`` or ``1.0``.
    alpha : float
        L2 penalty weight (applied to the weights only).
    fit_intercept : bool
        Whether ``params`` ends with an intercept.

    Returns
    -------
    gradient : ndarray with the same shape as ``params``
    """
    # No penalty on the intercept, so it only enters through the residuals.
    n = X.shape[0]
    if fit_intercept:
        w = params[:-1]
        b = params[-1]
    else:
        w = params
        b = 0.0

    z = X @ w + b
    p = _sigmoid(z)
    res = p - y

    grad_w: FloatArray = (1 / n) * (X.T @ res + alpha * w)
    if fit_intercept:
        grad_b = (1 / n) * np.sum(res)
        gradient: FloatArray = np.append(grad_w, grad_b)
        return gradient

    return grad_w


def _encode_labels(y: NDArray[Any]) -> tuple[NDArray[Any], FloatArray]:
    """Map two arbitrary class labels to 0.0 and 1.0.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Labels of any sortable dtype, for example integers or strings.

    Returns
    -------
    classes : ndarray of shape (2,)
        The sorted unique labels. ``classes[1]`` is the positive class.
    y_encoded : ndarray of shape (n_samples,)
        ``0.0`` where ``y == classes[0]`` and ``1.0`` where ``y == classes[1]``.

    Raises
    ------
    ValueError
        If ``y`` has NaN or infinite values, or does not have exactly two
        distinct labels.
    """
    # np.unique would silently treat NaN as one more class label.
    if y.dtype.kind == "f" and not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")

    # return_inverse gives, for each sample, the index of its label in the
    # sorted classes: exactly the 0/1 encoding when there are two classes.
    classes, inverse = np.unique(y, return_inverse=True)
    if len(classes) != 2:
        raise ValueError(
            f"LogisticRegression supports binary classification only: y must "
            f"contain exactly 2 distinct labels, got {len(classes)}: "
            f"{classes.tolist()}."
        )
    return classes, inverse.astype(np.float64)


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

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated weights.
    intercept_ : float
        Estimated intercept; 0.0 if ``fit_intercept=False``.
    classes_ : ndarray of shape (2,)
        The sorted class labels. ``classes_[1]`` is the positive class, the
        one whose probability the model estimates.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_iter_ : int
        Number of gradient descent steps run.

    Examples
    --------
    >>> from mlfs.linear_model import LogisticRegression
    >>> X = [[-2.0], [-1.0], [1.0], [2.0]]
    >>> model = LogisticRegression().fit(X, [0, 0, 1, 1])
    >>> model.classes_
    array([0, 1])
    >>> model.predict([[-3.0], [3.0]])
    array([0, 1])
    >>> model.predict_proba([[0.0]]).round(3)
    array([[0.5, 0.5]])
    """

    coef_: FloatArray
    intercept_: float
    classes_: NDArray[Any]
    n_features_in_: int
    n_iter_: int

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
        """Fit the logistic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Scale the features first: gradient descent
            converges slowly (or diverges) on features of very different
            magnitudes.
        y : array-like of shape (n_samples,)
            Labels. Exactly two distinct values are required; they may be of
            any sortable type, for example integers or strings.

        Returns
        -------
        self : LogisticRegression
            The fitted estimator.

        Raises
        ------
        ValueError
            If a hyperparameter or the data is invalid, if ``y`` does not
            contain exactly two classes, or if gradient descent diverges.

        Warns
        -----
        ConvergenceWarning
            If ``max_iter`` steps were not enough to reach ``tol``.
        """
        self._check_params()
        X_checked, y_checked = check_X_y(X, y, y_numeric=False)
        classes, y_encoded = _encode_labels(y_checked)

        n_features = X_checked.shape[1]
        # The intercept is the last parameter, so that (w, b) is optimized as
        # one vector: centering cannot remove it as in linear regression.
        n_params = n_features + 1 if self.fit_intercept else n_features

        # Called directly from fit so the ConvergenceWarning points at the
        # user's fit(...) line.
        params, self.n_iter_ = gradient_descent(
            gradient=lambda theta: _gradient(
                params=theta,
                X=X_checked,
                y=y_encoded,
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
            ),
            initial=np.zeros(n_params),
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        self.coef_ = params[:n_features]
        self.intercept_ = float(params[n_features]) if self.fit_intercept else 0.0
        self.classes_ = classes
        self.n_features_in_ = n_features
        return self

    def decision_function(self, X: ArrayLike) -> FloatArray:
        """Return the linear score :math:`z = Xw + b` of each sample.

        Positive scores mean the positive class (``classes_[1]``) is more
        likely than the negative one.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        z : ndarray of shape (n_samples,)
            The scores, also known as log-odds.

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
        scores: FloatArray = X_checked @ self.coef_ + self.intercept_
        return scores

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Estimate the probability of each class for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Column 0 is the probability of ``classes_[0]`` and column 1 that
            of ``classes_[1]``. Each row sums to 1.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        ValueError
            If ``X`` is invalid or has a different number of features than
            the data seen during ``fit``.
        """
        z = self.decision_function(X)
        # sigmoid(-z) is 1 - sigmoid(z) computed without cancellation, which
        # keeps tiny probabilities accurate when |z| is large.
        proba: FloatArray = np.column_stack([_sigmoid(-z), _sigmoid(z)])
        return proba

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict class labels for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted labels, taken from ``classes_``. A sample whose
            probability is exactly 0.5 (score 0) gets ``classes_[0]``.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        ValueError
            If ``X`` is invalid or has a different number of features than
            the data seen during ``fit``.
        """
        # Thresholding the score at 0 is the same as thresholding the
        # probability at 0.5, without computing the sigmoid.
        positive = self.decision_function(X) > 0
        labels: NDArray[Any] = self.classes_[positive.astype(np.intp)]
        return labels
