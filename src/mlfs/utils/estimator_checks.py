"""Generic checks that every mlfs estimator must pass.

These checks encode the estimator contract described in :mod:`mlfs.base`, so
each algorithm's test suite only needs one line to verify it::

    from mlfs.utils.estimator_checks import check_estimator

    def test_estimator_contract() -> None:
        check_estimator(LinearRegression())

The checks only depend on NumPy (not on pytest), so they also work for custom
estimators written in other projects that build on mlfs.
"""

import copy
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np

from mlfs._typing import FloatArray, NDArray
from mlfs.base import (
    BaseClassifier,
    BaseEstimator,
    BaseRegressor,
    BaseTransformer,
)
from mlfs.exceptions import NotFittedError

__all__ = ["check_estimator"]

_N_SAMPLES = 30
_N_FEATURES = 3

Supported: TypeAlias = BaseRegressor | BaseClassifier | BaseTransformer
"""Estimator kinds that ``check_estimator`` knows how to exercise."""

Check = Callable[[Supported], None]


def check_estimator(estimator: BaseEstimator) -> None:
    """Run every applicable contract check on an unfitted estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        An unfitted instance of a regressor, classifier or transformer.

    Raises
    ------
    AssertionError
        If a check fails. The message names the failing check.
    TypeError
        If the estimator is not a regressor, classifier or transformer.
    """
    if not isinstance(estimator, BaseRegressor | BaseClassifier | BaseTransformer):
        raise TypeError(
            f"{type(estimator).__name__} must subclass BaseRegressor, "
            "BaseClassifier or BaseTransformer."
        )
    for check in _CHECKS:
        try:
            check(_fresh(estimator))
        except AssertionError as exc:
            raise AssertionError(
                f"{type(estimator).__name__} failed {check.__name__}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh(estimator: Supported) -> Supported:
    """Return an unfitted copy with a fixed seed, if it has one."""
    clone = type(estimator)(**copy.deepcopy(estimator.get_params()))
    if "random_state" in clone.get_params():
        clone.set_params(random_state=0)
    return clone


def _make_data(estimator: Supported) -> tuple[FloatArray, NDArray[Any]]:
    """Return a small, well-conditioned dataset suited to the estimator."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(_N_SAMPLES, _N_FEATURES))
    if isinstance(estimator, BaseClassifier):
        y: NDArray[Any] = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
    else:
        y = X @ np.array([1.5, -2.0, 0.5]) + 0.1 * rng.normal(size=_N_SAMPLES)
    return X, y


def _fit(estimator: Supported, X: Any, y: Any) -> Any:
    return estimator.fit(X, y)


def _output(estimator: Supported, X: Any) -> NDArray[Any]:
    if isinstance(estimator, BaseTransformer):
        return estimator.transform(X)
    return estimator.predict(X)


def _raises(exc_type: type[Exception], func: Callable[[], object]) -> bool:
    try:
        func()
    except exc_type:
        return True
    return False


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_get_params_roundtrip(estimator: Supported) -> None:
    """``__init__`` must store every hyperparameter unchanged."""
    params = estimator.get_params()
    rebuilt = type(estimator)(**params).get_params()
    for name, value in params.items():
        assert rebuilt[name] is value or rebuilt[name] == value, (
            f"hyperparameter {name!r} is modified in __init__"
        )


def check_not_fitted_raises(estimator: Supported) -> None:
    """Predicting or transforming before ``fit`` must raise NotFittedError."""
    X, _ = _make_data(estimator)
    assert _raises(NotFittedError, lambda: _output(estimator, X)), (
        "using the estimator before fit did not raise NotFittedError"
    )


def check_fit_returns_self(estimator: Supported) -> None:
    """``fit`` must return the estimator itself to allow chaining."""
    X, y = _make_data(estimator)
    assert _fit(estimator, X, y) is estimator, "fit did not return self"


def check_fit_does_not_mutate_inputs(estimator: Supported) -> None:
    """``fit`` must not modify the caller's arrays in place."""
    X, y = _make_data(estimator)
    X_before, y_before = X.copy(), y.copy()
    _fit(estimator, X, y)
    assert np.array_equal(X, X_before), "fit modified X in place"
    assert np.array_equal(y, y_before), "fit modified y in place"


def check_fitted_attributes(estimator: Supported) -> None:
    """``fit`` must set ``n_features_in_`` and not change hyperparameters."""
    X, y = _make_data(estimator)
    params_before = copy.deepcopy(estimator.get_params())
    _fit(estimator, X, y)
    assert getattr(estimator, "n_features_in_", None) == _N_FEATURES, (
        "fit must set n_features_in_ to the number of columns of X"
    )
    for name, value in params_before.items():
        assert estimator.get_params()[name] == value, (
            f"fit modified the hyperparameter {name!r}; store learned values "
            "in attributes with a trailing underscore instead"
        )


def check_output_shape(estimator: Supported) -> None:
    """Check that outputs have one row per input sample."""
    X, y = _make_data(estimator)
    out = _output(_fit(estimator, X, y), X[:7])
    if isinstance(estimator, BaseTransformer):
        assert out.ndim == 2, f"transform must return a 2D array, got {out.ndim}D"
        assert out.shape[0] == 7, "transform must return one row per sample"
    else:
        assert out.shape == (7,), f"predict must return shape (7,), got {out.shape}"


def check_n_features_mismatch_raises(estimator: Supported) -> None:
    """Check that data with a different number of features is rejected."""
    X, y = _make_data(estimator)
    _fit(estimator, X, y)
    X_wrong = np.ones((5, _N_FEATURES + 1))
    assert _raises(ValueError, lambda: _output(estimator, X_wrong)), (
        "data with the wrong number of features did not raise ValueError"
    )


def check_nan_raises(estimator: Supported) -> None:
    """NaN in the training data must raise ValueError."""
    X, y = _make_data(estimator)
    X[0, 0] = np.nan
    assert _raises(ValueError, lambda: _fit(estimator, X, y)), (
        "fitting on data containing NaN did not raise ValueError"
    )


def check_deterministic(estimator: Supported) -> None:
    """Two fits with the same data and seed must give the same outputs."""
    X, y = _make_data(estimator)
    first = _output(_fit(estimator, X, y), X)
    second = _output(_fit(_fresh(estimator), X, y), X)
    assert np.allclose(first, second), "two identical fits gave different results"


def check_accepts_lists(estimator: Supported) -> None:
    """Plain Python lists must work and give the same result as arrays."""
    X, y = _make_data(estimator)
    expected = _output(_fit(estimator, X, y), X)
    other = _fresh(estimator)
    got = _output(_fit(other, X.tolist(), y.tolist()), X.tolist())
    assert np.allclose(got, expected), "lists and arrays gave different results"


# Order matters: determinism is checked before list support, because the list
# check compares two fits and would otherwise also fail for random estimators.
_CHECKS: tuple[Check, ...] = (
    check_get_params_roundtrip,
    check_not_fitted_raises,
    check_fit_returns_self,
    check_fit_does_not_mutate_inputs,
    check_fitted_attributes,
    check_output_shape,
    check_n_features_mismatch_raises,
    check_nan_raises,
    check_deterministic,
    check_accepts_lists,
)
