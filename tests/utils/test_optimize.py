"""Tests for mlfs.utils._optimize."""

import re
import sys

import numpy as np
import pytest

from mlfs._typing import FloatArray
from mlfs.exceptions import ConvergenceWarning
from mlfs.utils._optimize import check_gradient_descent_params, gradient_descent

# Hand-checkable loss used throughout: L(theta) = 1/2 * (theta - 3)^2, so the
# gradient is theta - 3 and the minimum is theta = 3.


def _gradient_towards_three(theta: FloatArray) -> FloatArray:
    return theta - 3.0


# --- gradient_descent: known answers ----------------------------------------


def test_gradient_descent_with_unit_step_reaches_minimum_in_one_update() -> None:
    params, n_iter = gradient_descent(
        _gradient_towards_three, np.zeros(1), learning_rate=1.0, max_iter=10, tol=1e-6
    )

    # gradient at 0 is -3; theta = 0 - 1 * (-3) = 3, where the gradient is 0
    np.testing.assert_allclose(params, [3.0])
    assert n_iter == 1


def test_gradient_descent_with_half_step_halves_the_distance_each_update() -> None:
    with pytest.warns(ConvergenceWarning):
        params, n_iter = gradient_descent(
            _gradient_towards_three,
            np.zeros(1),
            learning_rate=0.5,
            max_iter=2,
            tol=1e-6,
        )

    # theta_1 = 0 - 0.5 * (0 - 3) = 1.5; theta_2 = 1.5 - 0.5 * (1.5 - 3) = 2.25
    np.testing.assert_allclose(params, [2.25])
    assert n_iter == 2


def test_gradient_descent_updates_every_parameter_independently() -> None:
    params, _ = gradient_descent(
        _gradient_towards_three,
        np.array([1.0, 5.0]),
        learning_rate=1.0,
        max_iter=10,
        tol=1e-6,
    )

    # each component jumps to 3 in one unit step, from below and from above
    np.testing.assert_allclose(params, [3.0, 3.0])


def test_gradient_descent_starting_at_the_minimum_makes_no_update() -> None:
    params, n_iter = gradient_descent(
        _gradient_towards_three,
        np.array([3.0]),
        learning_rate=0.1,
        max_iter=10,
        tol=1e-6,
    )

    np.testing.assert_allclose(params, [3.0])
    assert n_iter == 0


def test_gradient_descent_with_gradient_norm_equal_to_tol_keeps_going() -> None:
    params, n_iter = gradient_descent(
        _gradient_towards_three, np.zeros(1), learning_rate=1.0, max_iter=10, tol=3.0
    )

    # gradient norm at 0 is exactly 3 == tol: a strict "<" must not stop there
    np.testing.assert_allclose(params, [3.0])
    assert n_iter == 1


def test_gradient_descent_checks_the_parameters_after_the_last_update() -> None:
    # Converges exactly on the only allowed update, so there must be no
    # warning (filterwarnings = ["error"] would turn one into a failure).
    params, n_iter = gradient_descent(
        _gradient_towards_three, np.zeros(1), learning_rate=1.0, max_iter=1, tol=1e-6
    )

    np.testing.assert_allclose(params, [3.0])
    assert n_iter == 1


def test_gradient_descent_does_not_modify_initial() -> None:
    initial = np.array([3.0])

    params, _ = gradient_descent(
        _gradient_towards_three, initial, learning_rate=1.0, max_iter=10, tol=1e-6
    )
    params[0] = 99.0

    # returned at once (no update), yet it must still be a separate array
    np.testing.assert_array_equal(initial, [3.0])


# --- gradient_descent: failure modes ----------------------------------------


def test_gradient_descent_without_enough_iterations_warns() -> None:
    message = (
        "Gradient descent did not converge within max_iter=2 iterations. "
        "Increase max_iter or learning_rate, or scale the features."
    )
    with pytest.warns(ConvergenceWarning, match=re.escape(message)):
        gradient_descent(
            _gradient_towards_three,
            np.zeros(1),
            learning_rate=0.5,
            max_iter=2,
            tol=1e-6,
        )


def test_gradient_descent_with_too_large_learning_rate_raises_value_error() -> None:
    # With learning_rate = 3 the distance to the minimum doubles in size and
    # flips sign each update (theta - 3 -> -2 * (theta - 3)), so it overflows.
    message = (
        "Gradient descent diverged: the weights are not finite. "
        "Use a smaller learning_rate or scale the features."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        gradient_descent(
            _gradient_towards_three,
            np.zeros(1),
            learning_rate=3.0,
            max_iter=10_000,
            tol=1e-6,
        )


def _fit_like_an_estimator() -> None:
    """Stand-in for an estimator's ``fit``, which calls the helper directly."""
    gradient_descent(
        _gradient_towards_three, np.zeros(1), learning_rate=0.5, max_iter=1, tol=1e-6
    )


# mutmut wraps every function in a trampoline, which adds a stack frame and so
# shifts where stacklevel points. The test is only meaningful without it.
@pytest.mark.skipif(
    "mutmut" in sys.modules, reason="mutmut's trampoline changes the call stack"
)
def test_gradient_descent_warning_points_at_the_line_that_called_fit() -> None:
    call_line = sys._getframe().f_lineno + 2  # the _fit_like_an_estimator() line
    with pytest.warns(ConvergenceWarning) as record:
        _fit_like_an_estimator()

    # stacklevel=3 skips gradient_descent and "fit", landing on the user's line
    assert record[0].filename == __file__
    assert record[0].lineno == call_line


# --- check_gradient_descent_params ------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"learning_rate": 0.0, "max_iter": 1, "tol": 0.0},
            "learning_rate must be > 0, got 0.0.",
        ),
        (
            {"learning_rate": 0.1, "max_iter": 0, "tol": 0.0},
            "max_iter must be >= 1, got 0.",
        ),
        (
            {"learning_rate": 0.1, "max_iter": 1, "tol": -1.0},
            "tol must be >= 0, got -1.0.",
        ),
    ],
    ids=["learning-rate", "max-iter", "tol"],
)
def test_check_gradient_descent_params_with_invalid_value_raises_value_error(
    params: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        check_gradient_descent_params(**params)  # type: ignore[arg-type]


def test_check_gradient_descent_params_accepts_the_boundaries() -> None:
    # smallest valid max_iter and tol; any positive learning rate
    check_gradient_descent_params(learning_rate=1e-12, max_iter=1, tol=0.0)
