"""Minimal estimators that follow the mlfs contract, shared by several tests.

They are deliberately trivial, so their correct outputs are obvious by
inspection. Test modules subclass them to build estimators that break the
contract in one specific way.
"""

from typing import Any, Self

import numpy as np

from mlfs._typing import ArrayLike, FloatArray, NDArray
from mlfs.base import BaseClassifier, BaseRegressor, BaseTransformer, check_is_fitted
from mlfs.utils.validation import (
    check_array,
    check_n_features,
    check_random_state,
    check_X_y,
)


class MeanRegressor(BaseRegressor):
    """Always predicts the training mean of y, plus ``offset``."""

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        X_arr, y_arr = check_X_y(X, y)
        self.n_features_in_ = X_arr.shape[1]
        self.mean_ = float(y_arr.mean()) + self.offset
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        check_is_fitted(self)
        X_arr = check_array(X)
        check_n_features(X_arr, self.n_features_in_)
        return np.full(X_arr.shape[0], self.mean_)


class NoisyRegressor(BaseRegressor):
    """Predicts a random constant drawn from ``random_state`` at fit time."""

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        X_arr, _ = check_X_y(X, y)
        self.n_features_in_ = X_arr.shape[1]
        self.value_ = check_random_state(self.random_state).normal()
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        check_is_fitted(self)
        X_arr = check_array(X)
        check_n_features(X_arr, self.n_features_in_)
        return np.full(X_arr.shape[0], self.value_)


class MajorityClassifier(BaseClassifier):
    """Always predicts the most frequent training label."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        X_arr, y_arr = check_X_y(X, y, y_numeric=False)
        self.n_features_in_ = X_arr.shape[1]
        labels, counts = np.unique(y_arr, return_counts=True)
        self.label_ = labels[np.argmax(counts)]
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        check_is_fitted(self)
        X_arr = check_array(X)
        check_n_features(X_arr, self.n_features_in_)
        return np.full(X_arr.shape[0], self.label_)


class Centerer(BaseTransformer):
    """Subtracts the per-feature training mean."""

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        X_arr = check_array(X)
        self.n_features_in_ = X_arr.shape[1]
        self.mean_: FloatArray = X_arr.mean(axis=0)
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        check_is_fitted(self)
        X_arr = check_array(X)
        check_n_features(X_arr, self.n_features_in_)
        return X_arr - self.mean_
