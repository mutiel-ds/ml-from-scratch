"""Tests for mlfs.linear_model.LinearRegression."""

import re
import sys

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from mlfs._typing import FloatArray
from mlfs.exceptions import ConvergenceWarning, NotFittedError
from mlfs.linear_model import LinearRegression
from mlfs.utils.estimator_checks import check_estimator

# --- Contract ---------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        LinearRegression(fit_intercept=False),
        LinearRegression(
            solver="gradient_descent", learning_rate=0.1, max_iter=5000, tol=1e-8
        ),
    ],
    ids=["defaults", "no-intercept", "gradient-descent"],
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


# --- Gradient descent -------------------------------------------------------


def test_gradient_descent_with_one_iteration_warns_and_returns_partial_solution() -> (
    None
):
    X = [[1], [2]]
    y = [1, 2]

    with pytest.warns(ConvergenceWarning, match="did not converge"):
        model = LinearRegression(
            solver="gradient_descent",
            learning_rate=0.1,
            max_iter=1,
            fit_intercept=False,
        ).fit(X, y)

    # gradient at w = 0: X^T (X*0 - y) / n = -(1*1 + 2*2) / 2 = -2.5
    # one update: w = 0 - 0.1 * (-2.5) = 0.25
    np.testing.assert_allclose(model.coef_, [0.25])
    assert model.n_iter_ == 1


def test_gradient_descent_with_two_iterations_returns_approximate_solution() -> None:
    X = [[1], [2]]
    y = [1, 2]

    with pytest.warns(ConvergenceWarning, match="did not converge"):
        model = LinearRegression(
            solver="gradient_descent",
            learning_rate=0.1,
            max_iter=2,
            fit_intercept=False,
        ).fit(X, y)

    # after the first update w = 0.25 (see the one-iteration test)
    # gradient at w = 0.25: residuals = [0.25 - 1, 0.5 - 2] = [-0.75, -1.5]
    #   -> (1*(-0.75) + 2*(-1.5)) / 2 = -1.875
    # second update: w = 0.25 - 0.1 * (-1.875) = 0.4375
    np.testing.assert_allclose(model.coef_, [0.4375])
    assert model.n_iter_ == 2


def test_gradient_descent_with_zero_gradient_returns_zero_weight_and_no_warning() -> (
    None
):
    X = [[1], [2]]
    y = [0, 0]

    # No pytest.warns here: pyproject.toml sets filterwarnings = ["error"], so
    # a ConvergenceWarning would turn into an exception and fail this test.
    model = LinearRegression(
        solver="gradient_descent", learning_rate=0.1, fit_intercept=False
    ).fit(X, y)

    # gradient at w = 0: -(1*0 + 2*0) / 2 = 0, which is below tol, so it stops
    # before the first update (atol because the expected weight is 0)
    np.testing.assert_allclose(model.coef_, [0.0], atol=1e-12)
    assert model.n_iter_ == 0


def test_gradient_descent_with_gradient_norm_equal_to_tol_keeps_going() -> None:
    X = [[1], [2]]
    y = [1, 2]

    model = LinearRegression(
        solver="gradient_descent",
        learning_rate=0.1,
        max_iter=1,
        tol=2.5,
        fit_intercept=False,
    ).fit(X, y)

    # gradient at w = 0: -(1*1 + 2*2)/2 = -2.5, norm 2.5 == tol, so a strict "<"
    # does not stop; one update gives w = 0.25 and the new gradient is -1.875 < 2.5.
    np.testing.assert_allclose(model.coef_, [0.25])
    assert model.n_iter_ == 1


def test_gradient_descent_on_exact_line_converges_to_slope_and_intercept() -> None:
    X = [[0], [1], [2], [3], [4]]
    y = [1, 3, 5, 7, 9]  # y = 2x + 1

    model = LinearRegression(
        solver="gradient_descent", learning_rate=0.1, max_iter=10_000, tol=1e-10
    ).fit(X, y)

    # Stopping at ||gradient|| < 1e-10 leaves a weight error of about
    # tol / lambda_min (lambda_min = 2 here), far below the 1e-6 tolerance.
    np.testing.assert_allclose(model.coef_, [2.0], atol=1e-6)
    assert model.intercept_ == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "fit_intercept", [True, False], ids=["intercept", "no-intercept"]
)
def test_gradient_descent_matches_lstsq_on_noisy_data(fit_intercept: bool) -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 3))
    y = X @ [1.5, -2.0, 0.5] + 3.0 + rng.normal(scale=0.5, size=50)
    gd = LinearRegression(
        fit_intercept=fit_intercept,
        solver="gradient_descent",
        learning_rate=0.1,
        max_iter=50_000,
        tol=1e-12,
    )

    gd.fit(X, y)
    exact = LinearRegression(fit_intercept=fit_intercept).fit(X, y)

    # Iterative solver: it stops when the gradient is below 1e-12, so the
    # weights agree with the closed form to roughly 1e-12 / lambda_min.
    np.testing.assert_allclose(gd.coef_, exact.coef_, atol=1e-8)
    assert gd.intercept_ == pytest.approx(exact.intercept_, abs=1e-8)


@given(y=_targets)
def test_gradient_descent_matches_lstsq_for_any_targets(y: FloatArray) -> None:
    gd = LinearRegression(
        solver="gradient_descent", learning_rate=0.1, max_iter=50_000, tol=1e-12
    ).fit(_X_FIXED, y)

    exact = LinearRegression().fit(_X_FIXED, y)

    # Same reasoning as above: the stopping rule bounds the weight error.
    np.testing.assert_allclose(gd.coef_, exact.coef_, atol=1e-7)
    assert gd.intercept_ == pytest.approx(exact.intercept_, abs=1e-7)


def test_gradient_descent_with_too_large_learning_rate_raises_value_error() -> None:
    model = LinearRegression(solver="gradient_descent", learning_rate=10.0)

    message = (
        "Gradient descent diverged: the weights are not finite. "
        "Use a smaller learning_rate or scale the features."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        model.fit([[0], [1], [2], [3], [4]], [1, 3, 5, 7, 9])


def test_gradient_descent_without_enough_iterations_warns() -> None:
    model = LinearRegression(solver="gradient_descent", max_iter=3)

    message = (
        "Gradient descent did not converge within max_iter=3 iterations. "
        "Increase max_iter or learning_rate, or scale the features."
    )
    with pytest.warns(ConvergenceWarning, match=re.escape(message)):
        model.fit([[0], [1], [2], [3], [4]], [1, 3, 5, 7, 9])


# mutmut wraps every function in a trampoline, which adds a stack frame and so
# shifts where stacklevel points. The test is only meaningful without it.
@pytest.mark.skipif(
    "mutmut" in sys.modules, reason="mutmut's trampoline changes the call stack"
)
def test_gradient_descent_warning_points_at_the_callers_fit_line() -> None:
    model = LinearRegression(solver="gradient_descent", max_iter=3)

    with pytest.warns(ConvergenceWarning) as record:
        model.fit([[0], [1], [2], [3], [4]], [1, 3, 5, 7, 9])

    # stacklevel=3 blames the line that called fit (this file), not the
    # library's internals.
    assert record[0].filename == __file__


def test_refit_with_lstsq_after_gradient_descent_clears_n_iter() -> None:
    X = [[1], [2]]
    y = [0, 0]  # zero gradient at w = 0, so gradient descent stops with n_iter_ = 0
    model = LinearRegression(solver="gradient_descent").fit(X, y)
    assert model.n_iter_ == 0

    model.set_params(solver="lstsq").fit(X, y)

    assert not hasattr(model, "n_iter_")


def test_fit_with_lstsq_solver_does_not_set_n_iter() -> None:
    model = LinearRegression(solver="lstsq").fit([[0], [1]], [0, 1])

    assert not hasattr(model, "n_iter_")


# --- Hyperparameters --------------------------------------------------------


def test_init_without_arguments_uses_documented_defaults() -> None:
    params = LinearRegression().get_params()

    assert params == {
        "fit_intercept": True,
        "solver": "lstsq",
        "learning_rate": 0.01,
        "max_iter": 1000,
        "tol": 1e-6,
    }


def test_fit_with_zero_tol_is_accepted() -> None:
    # tol=0 is a valid (if demanding) stopping rule, so the boundary is allowed.
    model = LinearRegression(tol=0.0).fit([[0], [1]], [0, 1])

    assert model.n_features_in_ == 1


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
