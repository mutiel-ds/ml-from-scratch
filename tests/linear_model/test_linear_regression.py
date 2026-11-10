"""Tests for mlfs.linear_model.LinearRegression."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from mlfs._typing import FloatArray
from mlfs.exceptions import NotFittedError
from mlfs.linear_model import LinearRegression
from mlfs.utils.estimator_checks import check_estimator

# --- Contract ---------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        LinearRegression(fit_intercept=False),
    ],
    ids=["defaults", "no-intercept"],
)
def test_linear_regression_satisfies_estimator_contract(
    estimator: LinearRegression,
) -> None:
    check_estimator(estimator)


# --- Known answers ----------------------------------------------------------


def test_fit_with_exact_line_recovers_slope_and_intercept() -> None:
    X = [[0], [1], [2], [3], [4]]
    y = [1, 3, 5, 7, 9]  # y = 2x + 1

    model = LinearRegression().fit(X, y)

    # means: x̄ = 2, ȳ = 5; centered slope 2; intercept = 5 - 2 * 2 = 1
    np.testing.assert_allclose(model.coef_, [2.0])
    assert model.intercept_ == pytest.approx(1.0)


def test_fit_with_identical_rows_returns_zero_weight_and_mean_of_y() -> None:
    X = [[1], [1], [1]]
    y = [1, 2, 6]

    model = LinearRegression().fit(X, y)

    # centered X = 0, so w = 0; b = mean(y) = (1+2+6)/3 = 3
    np.testing.assert_allclose(model.coef_, [0.0], atol=1e-12)
    assert model.intercept_ == pytest.approx(3.0)


def test_fit_with_fit_intercept_false_returns_slope_and_zero_intercept() -> None:
    X = [[0], [1], [2], [3], [4]]
    y = [1, 3, 5, 7, 9]  # y = 2x + 1

    model = LinearRegression(fit_intercept=False).fit(X, y)

    # w = Σxy / Σx² = (0+3+10+21+36) / (0+1+4+9+16) = 70/30 = 7/3
    np.testing.assert_allclose(model.coef_, [7.0 / 3.0])
    assert model.intercept_ == pytest.approx(0.0)


def test_fit_with_exact_plane_recovers_both_weights_and_intercept() -> None:
    X = [[0, 0], [1, 0], [0, 1], [1, 1]]
    y = [1, 3, -2, 0]  # y = 1 + 2*x1 - 3*x2

    model = LinearRegression().fit(X, y)

    np.testing.assert_allclose(model.coef_, [2.0, -3.0], atol=1e-12)
    assert model.intercept_ == pytest.approx(1.0)


def test_fit_with_collinear_features_returns_minimum_norm_weights() -> None:
    X = [[1, 2], [2, 4], [3, 6]]  # second column is twice the first
    y = [1, 2, 3]

    model = LinearRegression(fit_intercept=False).fit(X, y)

    # Every w with w1 + 2*w2 = 1 fits exactly; the one with the smallest norm
    # is the projection of the origin onto that line: (1, 2) / (1 + 4).
    np.testing.assert_allclose(model.coef_, [0.2, 0.4], atol=1e-12)


def test_fit_with_single_sample_and_intercept_returns_zero_weight() -> None:
    model = LinearRegression().fit([[5]], [3])

    # Centering makes X all zeros, so the weight is 0 and b = mean(y) = 3.
    np.testing.assert_allclose(model.coef_, [0.0], atol=1e-12)
    assert model.intercept_ == pytest.approx(3.0)


def test_predict_after_fitting_exact_line_returns_line_values() -> None:
    model = LinearRegression().fit([[0], [1], [2], [3], [4]], [1, 3, 5, 7, 9])

    y_pred = model.predict([[3], [10]])

    np.testing.assert_allclose(y_pred, [7.0, 21.0])  # 2x + 1


# --- Reference: scikit-learn ------------------------------------------------


@pytest.mark.parametrize(
    "fit_intercept", [True, False], ids=["intercept", "no-intercept"]
)
def test_fit_matches_sklearn_on_noisy_data(fit_intercept: bool) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ [1.5, -2.0, 0.5] + 3.0 + rng.normal(scale=0.5, size=50)

    ours = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    reference = SklearnLinearRegression(fit_intercept=fit_intercept).fit(X, y)

    np.testing.assert_allclose(ours.coef_, reference.coef_, rtol=1e-7)
    np.testing.assert_allclose(ours.intercept_, reference.intercept_, rtol=1e-7)
    np.testing.assert_allclose(ours.predict(X), reference.predict(X), rtol=1e-7)


# --- Properties -------------------------------------------------------------

# A fixed, well-conditioned X: Hypothesis varies the targets, not the design
# matrix, so failures reflect the properties and not ill-conditioning.
_X_FIXED = np.random.default_rng(1).normal(size=(20, 3))
_bounded = st.floats(min_value=-10, max_value=10, allow_nan=False)
_targets = hnp.arrays(np.float64, 20, elements=_bounded)
_weights = hnp.arrays(np.float64, 3, elements=_bounded)

# Round-off is about 1e-15 times the data magnitude (<= ~100) times the
# conditioning of X, hence an absolute tolerance of 1e-8.
_ATOL = 1e-8


@given(y=_targets, c=_bounded)
def test_fit_scaling_targets_scales_coef_and_intercept(y: FloatArray, c: float) -> None:
    base = LinearRegression().fit(_X_FIXED, y)

    scaled = LinearRegression().fit(_X_FIXED, c * y)

    np.testing.assert_allclose(scaled.coef_, c * base.coef_, atol=_ATOL)
    assert scaled.intercept_ == pytest.approx(c * base.intercept_, abs=_ATOL)


@given(y=_targets, d=_bounded)
def test_fit_shifting_targets_shifts_only_the_intercept(
    y: FloatArray, d: float
) -> None:
    base = LinearRegression().fit(_X_FIXED, y)

    shifted = LinearRegression().fit(_X_FIXED, y + d)

    np.testing.assert_allclose(shifted.coef_, base.coef_, atol=_ATOL)
    assert shifted.intercept_ == pytest.approx(base.intercept_ + d, abs=_ATOL)


@given(w=_weights, b=_bounded)
def test_fit_on_exact_linear_data_recovers_generating_parameters(
    w: FloatArray, b: float
) -> None:
    y = _X_FIXED @ w + b

    model = LinearRegression().fit(_X_FIXED, y)

    np.testing.assert_allclose(model.coef_, w, atol=_ATOL)
    assert model.intercept_ == pytest.approx(b, abs=_ATOL)


# --- Errors and edge cases --------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"solver": "svd"}, "solver must be one of"),
        ({"learning_rate": 0.0}, "learning_rate must be > 0"),
        ({"max_iter": 0}, "max_iter must be >= 1"),
        ({"tol": -1.0}, "tol must be >= 0"),
    ],
    ids=["solver", "learning-rate", "max-iter", "tol"],
)
def test_fit_with_invalid_hyperparameter_raises_value_error(
    params: dict[str, object], message: str
) -> None:
    model = LinearRegression(**params)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        model.fit([[1.0], [2.0]], [1.0, 2.0])


def test_predict_before_fit_raises_not_fitted_error() -> None:
    with pytest.raises(NotFittedError, match="not fitted"):
        LinearRegression().predict([[1.0]])


def test_predict_with_wrong_number_of_features_raises_value_error() -> None:
    model = LinearRegression().fit([[0, 0], [1, 0], [0, 1]], [0, 1, 2])

    with pytest.raises(ValueError, match="X has 1 features"):
        model.predict([[1.0]])
