"""Tests for mlfs.base: hyperparameter handling, scoring and fit checks.

The contract checks in mlfs.utils.estimator_checks have their own module.
"""

from typing import Any, Self

import numpy as np
import pytest

from mlfs._typing import ArrayLike
from mlfs.base import BaseEstimator, BaseRegressor, check_is_fitted
from mlfs.exceptions import NotFittedError
from toy_estimators import Centerer, MajorityClassifier, MeanRegressor

# --- Estimators used only in this module --------------------------------------


class TwoParams(BaseEstimator):
    """Hyperparameters deliberately declared in non-alphabetical order."""

    def __init__(self, beta: int = 2, alpha: str = "a") -> None:
        self.beta = beta
        self.alpha = alpha


class HasPrivateCache(MeanRegressor):
    """Unfitted, but has a private attribute that ends in an underscore."""

    def __init__(self, offset: float = 0.0) -> None:
        super().__init__(offset)
        self._cache_ = 1


class RecordsTarget(Centerer):
    """A transformer that uses y, as supervised transformers do."""

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        self.y_seen_ = y
        return super().fit(X, y)


class UsesArgs(BaseEstimator):
    def __init__(self, *args: Any) -> None:
        pass


class UsesKwargs(BaseEstimator):
    def __init__(self, **kwargs: Any) -> None:
        pass


# --- get_params ---------------------------------------------------------------


def test_get_params_returns_current_hyperparameter_values() -> None:
    estimator = TwoParams(beta=5, alpha="z")

    result = estimator.get_params()

    assert result == {"alpha": "z", "beta": 5}


def test_get_params_lists_names_in_alphabetical_order() -> None:
    estimator = TwoParams()

    result = estimator.get_params()

    assert list(result) == ["alpha", "beta"]


def test_get_params_without_init_returns_empty_dict() -> None:
    estimator = Centerer()

    result = estimator.get_params()

    assert result == {}


@pytest.mark.parametrize(
    "estimator_class",
    [pytest.param(UsesArgs, id="args"), pytest.param(UsesKwargs, id="kwargs")],
)
def test_get_params_with_variadic_init_raises_type_error(
    estimator_class: type[BaseEstimator],
) -> None:
    estimator = estimator_class()

    with pytest.raises(
        TypeError,
        match=rf"^{estimator_class.__name__}\.__init__ must not use \*args or "
        r"\*\*kwargs; every hyperparameter has to be an explicit argument\.$",
    ):
        estimator.get_params()


# --- set_params ---------------------------------------------------------------


def test_set_params_updates_given_hyperparameters_only() -> None:
    estimator = TwoParams(beta=1, alpha="a")

    estimator.set_params(beta=9)

    assert estimator.get_params() == {"alpha": "a", "beta": 9}


def test_set_params_updates_several_hyperparameters_at_once() -> None:
    estimator = TwoParams()

    estimator.set_params(beta=3, alpha="c")

    assert estimator.get_params() == {"alpha": "c", "beta": 3}


def test_set_params_returns_the_estimator_itself() -> None:
    estimator = TwoParams()

    result = estimator.set_params(beta=3)

    assert result is estimator


def test_set_params_with_unknown_name_raises_value_error_listing_valid_names() -> None:
    estimator = TwoParams()

    with pytest.raises(
        ValueError,
        match=r"^Invalid parameter 'gamma' for TwoParams\. "
        r"Valid parameters are: \['alpha', 'beta'\]\.$",
    ):
        estimator.set_params(gamma=1)


def test_set_params_with_unknown_name_leaves_estimator_unchanged() -> None:
    estimator = TwoParams(beta=1)

    with pytest.raises(ValueError, match="Invalid parameter 'gamma'"):
        estimator.set_params(gamma=1)

    assert not hasattr(estimator, "gamma")


# --- __repr__ -----------------------------------------------------------------


def test_repr_shows_class_name_and_hyperparameters() -> None:
    estimator = TwoParams(beta=5, alpha="z")

    result = repr(estimator)

    assert result == "TwoParams(alpha='z', beta=5)"


def test_repr_without_hyperparameters_shows_empty_parentheses() -> None:
    result = repr(Centerer())

    assert result == "Centerer()"


# --- BaseRegressor ------------------------------------------------------------


def test_regressor_score_returns_r2_of_predictions() -> None:
    X = [[0.0], [0.0], [0.0]]
    y = [1.0, 2.0, 3.0]
    model = MeanRegressor(offset=1.0).fit(X, y)  # always predicts 2 + 1 = 3

    result = model.score(X, y)

    # SS_res = (1-3)^2 + (2-3)^2 + 0 = 5, SS_tot = 2, R^2 = 1 - 5/2 = -1.5
    assert result == pytest.approx(-1.5)


def test_regressor_without_fit_and_predict_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseRegressor()  # type: ignore[abstract]


# --- BaseClassifier -----------------------------------------------------------


def test_classifier_score_returns_accuracy_of_predictions() -> None:
    X = [[0.0], [0.0], [0.0], [0.0]]
    y = ["a", "a", "a", "b"]
    model = MajorityClassifier().fit(X, y)  # always predicts "a"

    result = model.score(X, y)

    # 3 of 4 labels are "a"
    assert result == pytest.approx(0.75)


# --- BaseTransformer ----------------------------------------------------------


def test_fit_transform_returns_transformed_training_data() -> None:
    X = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]

    result = Centerer().fit_transform(X)

    # column means are [2, 3]
    np.testing.assert_array_equal(result, [[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])


def test_fit_transform_fits_the_estimator() -> None:
    X = [[0.0, 1.0], [2.0, 3.0]]
    transformer = Centerer()

    transformer.fit_transform(X)

    np.testing.assert_array_equal(transformer.mean_, [1.0, 2.0])


def test_fit_transform_passes_y_to_fit() -> None:
    transformer = RecordsTarget()

    transformer.fit_transform([[0.0], [1.0]], [5.0, 6.0])

    assert transformer.y_seen_ == [5.0, 6.0]


# --- check_is_fitted ----------------------------------------------------------


def test_check_is_fitted_before_fit_raises_not_fitted_error() -> None:
    estimator = MeanRegressor()

    with pytest.raises(
        NotFittedError,
        match=r"^This MeanRegressor instance is not fitted yet\. Call 'fit' with "
        r"appropriate arguments before using this estimator\.$",
    ):
        check_is_fitted(estimator)


def test_check_is_fitted_after_fit_does_not_raise() -> None:
    estimator = MeanRegressor().fit([[1.0], [2.0]], [1.0, 3.0])

    check_is_fitted(estimator)


def test_check_is_fitted_ignores_private_attributes_ending_in_underscore() -> None:
    estimator = HasPrivateCache()

    with pytest.raises(NotFittedError, match="not fitted yet"):
        check_is_fitted(estimator)


def test_not_fitted_error_is_both_value_error_and_attribute_error() -> None:
    error = NotFittedError("message")

    assert isinstance(error, ValueError)
    assert isinstance(error, AttributeError)
