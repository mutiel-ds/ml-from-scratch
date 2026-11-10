"""Tests for mlfs.linear_model.Ridge."""

import re
import sys

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn.linear_model import Ridge as SklearnRidge

from mlfs._typing import FloatArray
from mlfs.exceptions import ConvergenceWarning, NotFittedError
from mlfs.linear_model import LinearRegression, Ridge
from mlfs.utils.estimator_checks import check_estimator

_CLOSED_FORM = ["solve", "lstsq"]

# --- Contract ---------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator",
    [
        Ridge(),
        Ridge(fit_intercept=False),
        Ridge(alpha=0.0),
        Ridge(solver="lstsq"),
        Ridge(solver="gradient_descent", learning_rate=0.1, max_iter=5000, tol=1e-8),
    ],
    ids=["defaults", "no-intercept", "alpha-zero", "lstsq", "gradient-descent"],
)
def test_ridge_satisfies_estimator_contract(estimator: Ridge) -> None:
    check_estimator(estimator)


# --- Known answers ----------------------------------------------------------


@pytest.mark.parametrize("solver", _CLOSED_FORM)
def test_fit_with_one_feature_shrinks_slope_towards_zero(solver: str) -> None:
    X = [[1], [2], [3]]
    y = [2, 4, 6]

    model = Ridge(alpha=2, solver=solver).fit(X, y)  # type: ignore[arg-type]

    # Centered: x_c = [-1, 0, 1], y_c = [-2, 0, 2], so
    # w = sum(x_c * y_c) / (sum(x_c**2) + alpha) = 4 / (2 + 2) = 1 (OLS gives 2).
    # Intercept: b = y_mean - x_mean * w = 4 - 2 * 1 = 2.
    np.testing.assert_allclose(model.coef_, [1])
    assert model.intercept_ == pytest.approx(2)


@pytest.mark.parametrize("solver", _CLOSED_FORM)
def test_fit_with_alpha_zero_on_exact_line_recovers_slope_and_intercept(
    solver: str,
) -> None:
    X = [[1], [2], [3]]
    y = [2, 5, 8]  # y = 3x - 1

    model = Ridge(alpha=0, solver=solver).fit(X, y)  # type: ignore[arg-type]

    # alpha = 0 is plain OLS: x_c = [-1, 0, 1], y_c = [-3, 0, 3], so
    # w = 6 / 2 = 3 and b = y_mean - x_mean * w = 5 - 2 * 3 = -1.
    np.testing.assert_allclose(model.coef_, [3])
    assert model.intercept_ == pytest.approx(-1)


@pytest.mark.parametrize("solver", _CLOSED_FORM)
def test_fit_with_fit_intercept_false_applies_penalty_without_centering(
    solver: str,
) -> None:
    X = [[1], [2]]
    y = [1, 2]

    model = Ridge(alpha=5, fit_intercept=False, solver=solver).fit(X, y)  # type: ignore[arg-type]

    # No centering: w = sum(x * y) / (sum(x**2) + alpha) = (1 + 4) / (1 + 4 + 5)
    # = 5 / 10 = 0.5, and the intercept stays 0.
    np.testing.assert_allclose(model.coef_, [0.5])
    assert model.intercept_ == pytest.approx(0)


@pytest.mark.parametrize("solver", _CLOSED_FORM)
def test_fit_with_collinear_features_and_alpha_returns_symmetric_weights(
    solver: str,
) -> None:
    X = [[1, 2], [2, 4], [3, 6]]  # second column is twice the first
    y = [1, 2, 3]

    model = Ridge(alpha=30.0, fit_intercept=False, solver=solver).fit(X, y)  # type: ignore[arg-type]

    # X^T X = [[14, 28], [28, 56]], X^T y = [14, 28]. The solution has the form
    # w = t * (1, 2), and the first row of (X^T X + alpha I) w = X^T y gives
    # (14 + 56) t + alpha t = 14, so t = 14 / (70 + 30) = 0.14.
    np.testing.assert_allclose(model.coef_, [0.14, 0.28])


def test_fit_with_lstsq_alpha_zero_and_collinear_features_returns_min_norm() -> None:
    X = [[1, 2], [2, 4], [3, 6]]
    y = [1, 2, 3]

    model = Ridge(alpha=0.0, fit_intercept=False, solver="lstsq").fit(X, y)

    # Same system as above with alpha = 0: t = 14 / 70 = 0.2, so w = (0.2, 0.4),
    # which is also the minimum-norm solution of w1 + 2*w2 = 1.
    np.testing.assert_allclose(model.coef_, [0.2, 0.4], atol=1e-12)


def test_fit_with_identical_rows_returns_zero_weight_and_mean_of_y() -> None:
    model = Ridge(alpha=1.0).fit([[1], [1], [1]], [1, 2, 6])

    # centered X = 0, so w = 0 whatever alpha is; b = mean(y) = 3
    np.testing.assert_allclose(model.coef_, [0.0], atol=1e-12)
    assert model.intercept_ == pytest.approx(3.0)


def test_fit_with_single_sample_and_intercept_returns_zero_weight() -> None:
    model = Ridge().fit([[5]], [3])

    # Centering makes X all zeros, so the weight is 0 and b = mean(y) = 3.
    np.testing.assert_allclose(model.coef_, [0.0], atol=1e-12)
    assert model.intercept_ == pytest.approx(3.0)


def test_predict_after_fit_returns_weights_times_x_plus_intercept() -> None:
    # Same data as the TODO case 1: w = 1, b = 2, so y_pred = x + 2.
    model = Ridge(alpha=2.0).fit([[1], [2], [3]], [2, 4, 6])

    y_pred = model.predict([[0], [10]])

    np.testing.assert_allclose(y_pred, [2.0, 12.0])


# --- Reference: scikit-learn ------------------------------------------------


@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no-int"])
@pytest.mark.parametrize("solver", _CLOSED_FORM)
@pytest.mark.parametrize("alpha", [0.1, 1.0, 50.0])
def test_fit_matches_sklearn_on_noisy_data(
    alpha: float, solver: str, fit_intercept: bool
) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ [1.5, -2.0, 0.5] + 3.0 + rng.normal(scale=0.5, size=50)

    ours = Ridge(alpha=alpha, solver=solver, fit_intercept=fit_intercept).fit(X, y)  # type: ignore[arg-type]
    reference = SklearnRidge(alpha=alpha, fit_intercept=fit_intercept).fit(X, y)

    np.testing.assert_allclose(ours.coef_, reference.coef_, rtol=1e-7)
    np.testing.assert_allclose(ours.intercept_, reference.intercept_, rtol=1e-7)
    np.testing.assert_allclose(ours.predict(X), reference.predict(X), rtol=1e-7)


@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no-int"])
def test_gradient_descent_matches_sklearn_on_noisy_data(fit_intercept: bool) -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 3))
    y = X @ [1.5, -2.0, 0.5] + 3.0 + rng.normal(scale=0.5, size=50)
    gd = Ridge(
        alpha=5.0,
        fit_intercept=fit_intercept,
        solver="gradient_descent",
        learning_rate=0.1,
        max_iter=50_000,
        tol=1e-12,
    ).fit(X, y)

    reference = SklearnRidge(alpha=5.0, fit_intercept=fit_intercept).fit(X, y)

    # Iterative solver: it stops when the gradient is below 1e-12, so the
    # weights agree with the closed form to roughly 1e-12 / lambda_min.
    np.testing.assert_allclose(gd.coef_, reference.coef_, atol=1e-8)
    assert gd.intercept_ == pytest.approx(reference.intercept_, abs=1e-8)


# --- Properties -------------------------------------------------------------

# A fixed, well-conditioned X: Hypothesis varies the targets, not the design
# matrix, so failures reflect the properties and not ill-conditioning.
_X_FIXED = np.random.default_rng(1).normal(size=(20, 3))
_bounded = st.floats(min_value=-10, max_value=10, allow_nan=False)
_targets = hnp.arrays(np.float64, 20, elements=_bounded)
_alphas = st.floats(min_value=0.0, max_value=100.0)

# Round-off is about 1e-15 times the data magnitude (<= ~100) times the
# conditioning of X, hence an absolute tolerance of 1e-8.
_ATOL = 1e-8


@given(y=_targets, alpha=_alphas, extra=st.floats(min_value=0.0, max_value=100.0))
def test_fit_with_larger_alpha_never_increases_coef_norm(
    y: FloatArray, alpha: float, extra: float
) -> None:
    smaller = Ridge(alpha=alpha).fit(_X_FIXED, y)

    larger = Ridge(alpha=alpha + extra).fit(_X_FIXED, y)

    # The penalized solution norm is monotonically non-increasing in alpha;
    # the slack only absorbs round-off.
    assert np.linalg.norm(larger.coef_) <= np.linalg.norm(smaller.coef_) + _ATOL


@given(y=_targets)
@pytest.mark.parametrize("solver", _CLOSED_FORM)
def test_fit_with_alpha_zero_matches_linear_regression(
    solver: str, y: FloatArray
) -> None:
    ridge = Ridge(alpha=0.0, solver=solver).fit(_X_FIXED, y)  # type: ignore[arg-type]

    ols = LinearRegression().fit(_X_FIXED, y)

    np.testing.assert_allclose(ridge.coef_, ols.coef_, atol=_ATOL)
    assert ridge.intercept_ == pytest.approx(ols.intercept_, abs=_ATOL)


@given(y=_targets, d=_bounded, alpha=_alphas)
def test_fit_shifting_targets_shifts_only_the_intercept(
    y: FloatArray, d: float, alpha: float
) -> None:
    base = Ridge(alpha=alpha).fit(_X_FIXED, y)

    shifted = Ridge(alpha=alpha).fit(_X_FIXED, y + d)

    # The intercept is not penalized, so a constant shift of y moves only it.
    np.testing.assert_allclose(shifted.coef_, base.coef_, atol=_ATOL)
    assert shifted.intercept_ == pytest.approx(base.intercept_ + d, abs=_ATOL)


@given(y=_targets, alpha=_alphas)
def test_fit_solve_and_lstsq_agree(y: FloatArray, alpha: float) -> None:
    via_solve = Ridge(alpha=alpha, solver="solve").fit(_X_FIXED, y)

    via_lstsq = Ridge(alpha=alpha, solver="lstsq").fit(_X_FIXED, y)

    np.testing.assert_allclose(via_solve.coef_, via_lstsq.coef_, atol=_ATOL)


def test_fit_does_not_modify_the_callers_arrays() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)
    X_before, y_before = X.copy(), y.copy()

    Ridge().fit(X, y)

    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


# --- Gradient descent -------------------------------------------------------
# Hand-checkable setup used by several tests: X = [[1], [2]], y = [1, 2],
# fit_intercept=False, alpha = 1, learning_rate = 0.1, n = 2.
# gradient(w) = (X^T (X w - y) + alpha * w) / n


def test_gradient_descent_with_one_iteration_warns_and_returns_partial_solution() -> (
    None
):
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        model = Ridge(
            alpha=1.0,
            solver="gradient_descent",
            learning_rate=0.1,
            max_iter=1,
            fit_intercept=False,
        ).fit([[1], [2]], [1, 2])

    # gradient at w = 0: (-(1*1 + 2*2) + 1*0) / 2 = -2.5
    # one update: w = 0 - 0.1 * (-2.5) = 0.25
    np.testing.assert_allclose(model.coef_, [0.25])
    assert model.n_iter_ == 1


def test_gradient_descent_with_two_iterations_includes_the_penalty_term() -> None:
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        model = Ridge(
            alpha=1.0,
            solver="gradient_descent",
            learning_rate=0.1,
            max_iter=2,
            fit_intercept=False,
        ).fit([[1], [2]], [1, 2])

    # after the first update w = 0.25
    # residuals at w = 0.25: [0.25 - 1, 0.5 - 2] = [-0.75, -1.5]
    #   X^T residuals = 1*(-0.75) + 2*(-1.5) = -3.75
    #   plus alpha * w = 0.25 -> -3.5; divided by n = 2 -> -1.75
    # second update: w = 0.25 - 0.1 * (-1.75) = 0.425
    # (without the penalty term it would be 0.4375, as in LinearRegression)
    np.testing.assert_allclose(model.coef_, [0.425])
    assert model.n_iter_ == 2


def test_gradient_descent_with_zero_gradient_returns_zero_weight_and_no_warning() -> (
    None
):
    # No pytest.warns here: pyproject.toml sets filterwarnings = ["error"], so
    # a ConvergenceWarning would turn into an exception and fail this test.
    model = Ridge(
        solver="gradient_descent", learning_rate=0.1, fit_intercept=False
    ).fit([[1], [2]], [0, 0])

    # gradient at w = 0 is 0, below tol, so it stops before the first update
    np.testing.assert_allclose(model.coef_, [0.0], atol=1e-12)
    assert model.n_iter_ == 0


def test_gradient_descent_with_gradient_norm_equal_to_tol_keeps_going() -> None:
    model = Ridge(
        alpha=1.0,
        solver="gradient_descent",
        learning_rate=0.1,
        max_iter=1,
        tol=2.5,
        fit_intercept=False,
    ).fit([[1], [2]], [1, 2])

    # gradient at w = 0 is -2.5, norm 2.5 == tol, so a strict "<" does not stop;
    # one update gives w = 0.25 and the new gradient has norm 1.75 < 2.5.
    np.testing.assert_allclose(model.coef_, [0.25])
    assert model.n_iter_ == 1


def test_gradient_descent_with_too_large_learning_rate_raises_value_error() -> None:
    model = Ridge(solver="gradient_descent", learning_rate=10.0)

    message = (
        "Gradient descent diverged: the weights are not finite. "
        "Use a smaller learning_rate or scale the features."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        model.fit([[0], [1], [2], [3], [4]], [1, 3, 5, 7, 9])


def test_gradient_descent_without_enough_iterations_warns() -> None:
    model = Ridge(solver="gradient_descent", max_iter=3)

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
    model = Ridge(solver="gradient_descent", max_iter=3)

    with pytest.warns(ConvergenceWarning) as record:
        model.fit([[0], [1], [2], [3], [4]], [1, 3, 5, 7, 9])

    # stacklevel=3 blames the line that called fit (this file), not the
    # library's internals.
    assert record[0].filename == __file__


@pytest.mark.parametrize("solver", _CLOSED_FORM)
def test_refit_with_closed_form_after_gradient_descent_clears_n_iter(
    solver: str,
) -> None:
    X = [[1], [2]]
    y = [0, 0]  # zero gradient at w = 0, so gradient descent stops with n_iter_ = 0
    model = Ridge(solver="gradient_descent").fit(X, y)
    assert model.n_iter_ == 0

    model.set_params(solver=solver).fit(X, y)

    assert not hasattr(model, "n_iter_")


@pytest.mark.parametrize("solver", _CLOSED_FORM)
def test_fit_with_closed_form_solver_does_not_set_n_iter(solver: str) -> None:
    model = Ridge(solver=solver).fit([[0], [1]], [0, 1])  # type: ignore[arg-type]

    assert not hasattr(model, "n_iter_")


# --- Hyperparameters --------------------------------------------------------


def test_init_without_arguments_uses_documented_defaults() -> None:
    params = Ridge().get_params()

    assert params == {
        "alpha": 1.0,
        "fit_intercept": True,
        "solver": "solve",
        "learning_rate": 0.01,
        "max_iter": 1000,
        "tol": 1e-6,
    }


def test_fit_with_zero_alpha_is_accepted() -> None:
    # alpha=0 is plain least squares, so the boundary must be allowed.
    model = Ridge(alpha=0.0).fit([[0], [1]], [0, 1])

    assert model.n_features_in_ == 1


def test_fit_with_zero_tol_is_accepted() -> None:
    model = Ridge(tol=0.0).fit([[0], [1]], [0, 1])

    assert model.n_features_in_ == 1


# --- Errors and edge cases --------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"alpha": -1.0}, "alpha must be finite and >= 0"),
        ({"alpha": float("nan")}, "alpha must be finite and >= 0"),
        ({"alpha": float("inf")}, "alpha must be finite and >= 0"),
        ({"solver": "svd"}, "solver must be one of"),
        ({"learning_rate": 0.0}, "learning_rate must be > 0"),
        ({"max_iter": 0}, "max_iter must be >= 1"),
        ({"tol": -1.0}, "tol must be >= 0"),
    ],
    ids=["alpha-negative", "alpha-nan", "alpha-inf", "solver", "lr", "max-iter", "tol"],
)
def test_fit_with_invalid_hyperparameter_raises_value_error(
    params: dict[str, object], message: str
) -> None:
    model = Ridge(**params)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        model.fit([[1.0], [2.0]], [1.0, 2.0])


def test_fit_with_solve_alpha_zero_and_collinear_features_raises_value_error() -> None:
    model = Ridge(alpha=0.0, solver="solve", fit_intercept=False)

    message = (
        "The regularized normal equation is singular. Use alpha > 0 or solver='lstsq'."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        model.fit([[1, 2], [2, 4], [3, 6]], [1, 2, 3])


def test_predict_before_fit_raises_not_fitted_error() -> None:
    with pytest.raises(NotFittedError, match="not fitted"):
        Ridge().predict([[1.0]])


def test_predict_with_wrong_number_of_features_raises_value_error() -> None:
    model = Ridge().fit([[0, 0], [1, 0], [0, 1]], [0, 1, 2])

    with pytest.raises(ValueError, match="X has 1 features"):
        model.predict([[1.0]])
