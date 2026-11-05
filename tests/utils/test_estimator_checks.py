"""Tests for mlfs.utils.estimator_checks.

Each check is proven twice: correct estimators pass, and an estimator that
breaks exactly one rule fails with that check's name and message.
"""

import re
from typing import Any, Self

import numpy as np
import pytest

from mlfs._typing import ArrayLike, FloatArray, NDArray
from mlfs.base import BaseClassifier, BaseEstimator, check_is_fitted
from mlfs.utils.estimator_checks import check_estimator
from mlfs.utils.validation import check_array, check_n_features, check_X_y
from toy_estimators import Centerer, MajorityClassifier, MeanRegressor, NoisyRegressor

# --- Estimators that follow the contract --------------------------------------


class StoresEqualCopy(MeanRegressor):
    """Stores an equal but not identical value (int 0 becomes float 0.0)."""

    def __init__(self, offset: float = 0) -> None:
        self.offset = float(offset)


class CopiesListParam(MeanRegressor):
    """Stores a copy of a list hyperparameter: equal, but not the same object."""

    def __init__(self, offset: float = 0.0, weights: list[float] | None = None) -> None:
        self.offset = offset
        self.weights = list(weights) if weights is not None else None


class BincountClassifier(BaseClassifier):
    """Like many real classifiers, only accepts integer class labels."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        X_arr, y_arr = check_X_y(X, y, y_numeric=False)
        self.n_features_in_ = X_arr.shape[1]
        self.label_ = int(np.argmax(np.bincount(y_arr)))
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        check_is_fitted(self)
        X_arr = check_array(X)
        check_n_features(X_arr, self.n_features_in_)
        return np.full(X_arr.shape[0], self.label_)


# --- Estimators that each break exactly one rule ------------------------------


class ModifiesParamInInit(MeanRegressor):
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset + 1.0


class SkipsFittedCheck(MeanRegressor):
    def predict(self, X: ArrayLike) -> FloatArray:
        return np.zeros(check_array(X).shape[0])


class FitReturnsNone(MeanRegressor):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        super().fit(X, y)
        return None  # type: ignore[return-value]


class MutatesX(MeanRegressor):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        np.asarray(X)[0, 0] = 99.0
        return super().fit(X, y)


class MutatesY(MeanRegressor):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        np.asarray(y)[0] = 99.0
        return super().fit(X, y)


class MissingNFeaturesIn(MeanRegressor):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        super().fit(X, y)
        del self.n_features_in_
        return self


class ModifiesParamInFit(MeanRegressor):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        self.offset = self.offset + 1.0
        return super().fit(X, y)


class MutatesListParamInFit(MeanRegressor):
    def __init__(self, offset: float = 0.0, weights: list[float] | None = None) -> None:
        self.offset = offset
        self.weights = weights

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        if self.weights is not None:
            self.weights.append(1.0)
        return super().fit(X, y)


class PredictsColumn(MeanRegressor):
    def predict(self, X: ArrayLike) -> FloatArray:
        return super().predict(X).reshape(-1, 1)


class TransformsTo1D(Centerer):
    def transform(self, X: ArrayLike) -> FloatArray:
        return super().transform(X).ravel()


class TransformsToOneRow(Centerer):
    def transform(self, X: ArrayLike) -> FloatArray:
        return super().transform(X)[:1]


class SkipsFeatureCheck(MeanRegressor):
    def predict(self, X: ArrayLike) -> FloatArray:
        check_is_fitted(self)
        return np.full(check_array(X).shape[0], self.mean_)


class AcceptsNaN(MeanRegressor):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        X_arr = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X_arr.shape[1]
        self.mean_ = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self


class IgnoresRandomState(NoisyRegressor):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        super().fit(X, y)
        self.value_ = np.random.default_rng().normal()  # unseeded on purpose
        return self


class ListSensitive(MeanRegressor):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        super().fit(X, y)
        self.mean_ += 1.0 if isinstance(X, list) else 0.0
        return self


# --- Tests --------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(MeanRegressor(offset=0.5), id="regressor"),
        pytest.param(NoisyRegressor(), id="seeded-random-regressor"),
        pytest.param(MajorityClassifier(), id="classifier"),
        pytest.param(Centerer(), id="transformer"),
        pytest.param(StoresEqualCopy(), id="equal-copy-of-hyperparameter"),
        pytest.param(CopiesListParam(weights=[1.0]), id="copied-list-hyperparameter"),
        pytest.param(BincountClassifier(), id="integer-label-classifier"),
    ],
)
def test_check_estimator_with_valid_estimator_passes(estimator: BaseEstimator) -> None:
    check_estimator(estimator)


@pytest.mark.parametrize(
    ("estimator", "expected_message"),
    [
        pytest.param(
            ModifiesParamInInit(),
            "check_get_params_roundtrip: hyperparameter 'offset' is modified in "
            "__init__",
            id="modifies-param-in-init",
        ),
        pytest.param(
            SkipsFittedCheck(),
            "check_not_fitted_raises: using the estimator before fit did not raise "
            "NotFittedError",
            id="skips-fitted-check",
        ),
        pytest.param(
            FitReturnsNone(),
            "check_fit_returns_self: fit did not return self",
            id="fit-returns-none",
        ),
        pytest.param(
            MutatesX(),
            "check_fit_does_not_mutate_inputs: fit modified X in place",
            id="mutates-x",
        ),
        pytest.param(
            MutatesY(),
            "check_fit_does_not_mutate_inputs: fit modified y in place",
            id="mutates-y",
        ),
        pytest.param(
            MissingNFeaturesIn(),
            "check_fitted_attributes: fit must set n_features_in_ to the number of "
            "columns of X",
            id="missing-n-features-in",
        ),
        pytest.param(
            ModifiesParamInFit(),
            "check_fitted_attributes: fit modified the hyperparameter 'offset'; "
            "store learned values in attributes with a trailing underscore instead",
            id="modifies-param-in-fit",
        ),
        pytest.param(
            MutatesListParamInFit(weights=[]),
            "check_fitted_attributes: fit modified the hyperparameter 'weights'; "
            "store learned values in attributes with a trailing underscore instead",
            id="mutates-list-param-in-fit",
        ),
        pytest.param(
            PredictsColumn(),
            "check_output_shape: predict must return shape (7,), got (7, 1)",
            id="predicts-column",
        ),
        pytest.param(
            TransformsTo1D(),
            "check_output_shape: transform must return a 2D array, got 1D",
            id="transforms-to-1d",
        ),
        pytest.param(
            TransformsToOneRow(),
            "check_output_shape: transform must return one row per sample",
            id="transforms-to-one-row",
        ),
        pytest.param(
            SkipsFeatureCheck(),
            "check_n_features_mismatch_raises: data with the wrong number of "
            "features did not raise ValueError",
            id="skips-feature-check",
        ),
        pytest.param(
            AcceptsNaN(),
            "check_nan_raises: fitting on data containing NaN did not raise ValueError",
            id="accepts-nan",
        ),
        pytest.param(
            IgnoresRandomState(),
            "check_deterministic: two identical fits gave different results",
            id="ignores-random-state",
        ),
        pytest.param(
            ListSensitive(),
            "check_accepts_lists: lists and arrays gave different results",
            id="list-sensitive",
        ),
    ],
)
def test_check_estimator_with_broken_estimator_names_the_failed_check(
    estimator: BaseEstimator, expected_message: str
) -> None:
    full_message = f"{type(estimator).__name__} failed {expected_message}"

    with pytest.raises(AssertionError, match=f"^{re.escape(full_message)}$"):
        check_estimator(estimator)


def test_check_estimator_with_unsupported_estimator_type_raises_type_error() -> None:
    estimator = BaseEstimator()

    with pytest.raises(
        TypeError,
        match=r"^BaseEstimator must subclass BaseRegressor, BaseClassifier or "
        r"BaseTransformer\.$",
    ):
        check_estimator(estimator)


def test_check_estimator_does_not_fit_the_given_instance() -> None:
    estimator = MeanRegressor()

    check_estimator(estimator)

    assert not hasattr(estimator, "mean_")


def test_check_estimator_keeps_the_original_random_state() -> None:
    estimator = NoisyRegressor(random_state=None)

    check_estimator(estimator)

    assert estimator.random_state is None


def test_check_estimator_does_not_touch_the_given_instance_hyperparameters() -> None:
    weights: list[float] = []
    estimator = MutatesListParamInFit(weights=weights)

    with pytest.raises(AssertionError, match="check_fitted_attributes"):
        check_estimator(estimator)

    assert weights == []
