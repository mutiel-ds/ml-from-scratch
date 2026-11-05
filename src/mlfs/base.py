"""Base classes that define the estimator contract shared by every algorithm.

Conventions (the same as scikit-learn's, so the API feels familiar):

* ``__init__`` only stores hyperparameters, unchanged, as attributes with the
  same names. No validation and no computation happen there.
* ``fit`` validates the hyperparameters and the data, learns from the data and
  returns ``self`` so calls can be chained: ``Model().fit(X, y).predict(X)``.
* Everything learned in ``fit`` is stored in attributes whose names end with a
  trailing underscore (``coef_``, ``n_features_in_``). Their presence is what
  marks an estimator as fitted.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Self

from mlfs._typing import ArrayLike, FloatArray, NDArray
from mlfs.exceptions import NotFittedError
from mlfs.metrics import accuracy_score, r2_score

__all__ = [
    "BaseClassifier",
    "BaseEstimator",
    "BaseRegressor",
    "BaseTransformer",
    "check_is_fitted",
]


class BaseEstimator:
    """Base class for all estimators.

    Provides hyperparameter introspection (``get_params``/``set_params``) and a
    readable ``repr`` based on the signature of the subclass's ``__init__``.
    """

    @classmethod
    def _get_param_names(cls) -> list[str]:
        if cls.__init__ is object.__init__:
            return []  # no explicit __init__, so no hyperparameters
        signature = inspect.signature(cls.__init__)
        names = []
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise TypeError(
                    f"{cls.__name__}.__init__ must not use *args or **kwargs; "
                    "every hyperparameter has to be an explicit argument."
                )
            names.append(name)
        return sorted(names)

    def get_params(self) -> dict[str, Any]:
        """Return the estimator's hyperparameters.

        Returns
        -------
        params : dict
            Mapping from hyperparameter name to its current value.
        """
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params: Any) -> Self:
        """Set hyperparameters of the estimator.

        Parameters
        ----------
        **params : dict
            Hyperparameter names mapped to their new values.

        Returns
        -------
        self : object
            The estimator, to allow chaining.
        """
        valid = set(self._get_param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter {name!r} for {type(self).__name__}. "
                    f"Valid parameters are: {sorted(valid)}."
                )
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        """Return ``ClassName(param=value, ...)``."""
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class BaseRegressor(BaseEstimator, ABC):
    """Base class for regressors: continuous targets, scored with R^2."""

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Learn from training data ``X`` and targets ``y``; return ``self``."""

    @abstractmethod
    def predict(self, X: ArrayLike) -> FloatArray:
        """Return predictions of shape ``(n_samples,)``."""

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination R^2 on ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            R^2 of ``self.predict(X)`` with respect to ``y``.
        """
        return r2_score(y, self.predict(X))


class BaseClassifier(BaseEstimator, ABC):
    """Base class for classifiers: discrete labels, scored with accuracy."""

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Learn from training data ``X`` and labels ``y``; return ``self``."""

    @abstractmethod
    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return predicted labels of shape ``(n_samples,)``."""

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the mean accuracy on ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Accuracy of ``self.predict(X)`` with respect to ``y``.
        """
        return accuracy_score(y, self.predict(X))


class BaseTransformer(BaseEstimator, ABC):
    """Base class for transformers: learn a mapping of ``X`` in ``fit``."""

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Learn the transformation from ``X``; ``y`` is ignored."""

    @abstractmethod
    def transform(self, X: ArrayLike) -> FloatArray:
        """Return the transformed data, one row per sample."""

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> FloatArray:
        """Fit to ``X`` and return the transformed ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored; present for API consistency.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X, y).transform(X)


def check_is_fitted(estimator: BaseEstimator) -> None:
    """Raise ``NotFittedError`` if ``estimator`` has not been fitted.

    An estimator counts as fitted when it has at least one public attribute
    ending in a trailing underscore, such as ``coef_``.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to check.

    Raises
    ------
    NotFittedError
        If no fitted attribute is found.
    """
    fitted = [
        name
        for name in vars(estimator)
        if name.endswith("_") and not name.startswith("_")
    ]
    if not fitted:
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this estimator."
        )
