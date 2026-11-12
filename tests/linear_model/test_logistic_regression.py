"""Tests for mlfs.linear_model.LogisticRegression."""

import numpy as np
import pytest

from mlfs.linear_model import LogisticRegression
from mlfs.linear_model._logistic_regression import _sigmoid

# --- Sigmoid ----------------------------------------------------------------


def test_sigmoid_at_zero_is_one_half() -> None:
    z = np.array([0.0])

    sigma = _sigmoid(z)

    # e^0 = 1, so sigma(0) = 1 / (1 + 1) = 0.5.
    np.testing.assert_allclose(sigma, [0.5])


def test_sigmoid_at_log_three_is_three_quarters() -> None:
    z = np.array([np.log(3.0)])

    sigma = _sigmoid(z)

    # e^{-ln 3} = 1/3, so sigma = 1 / (1 + 1/3) = 3/4.
    np.testing.assert_allclose(sigma, [0.75])


@pytest.mark.parametrize("z", [0.5, 1.0, 4.0])
def test_sigmoid_is_symmetric(z: float) -> None:
    sigma_positive = _sigmoid(np.array([z]))
    sigma_negative = _sigmoid(np.array([-z]))

    # Property: sigma(-z) = 1 - sigma(z).
    np.testing.assert_allclose(sigma_negative, 1 - sigma_positive)


def test_sigmoid_at_extreme_inputs_saturates_without_warnings() -> None:
    z = np.array([-1000.0, 1000.0])

    # pytest is configured with filterwarnings = ["error"], so any overflow
    # RuntimeWarning raised here would already fail the test.
    sigma = _sigmoid(z)

    # atol is needed because assert_allclose's default is relative: it would
    # reject any tiny result when comparing with an exact 0.
    np.testing.assert_allclose(sigma, [0.0, 1.0], atol=1e-300)


# --- Errors and edge cases --------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"alpha": -1.0}, "alpha must be finite and >= 0"),
        ({"alpha": float("nan")}, "alpha must be finite and >= 0"),
        ({"alpha": float("inf")}, "alpha must be finite and >= 0"),
        ({"learning_rate": 0.0}, "learning_rate must be > 0"),
        ({"max_iter": 0}, "max_iter must be >= 1"),
        ({"tol": -1.0}, "tol must be >= 0"),
    ],
    ids=["alpha-negative", "alpha-nan", "alpha-inf", "lr", "max-iter", "tol"],
)
def test_fit_with_invalid_hyperparameter_raises_value_error(
    params: dict[str, object], message: str
) -> None:
    model = LogisticRegression(**params)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        model.fit([[1.0], [2.0]], [0, 1])


def test_init_stores_hyperparameters_unchanged() -> None:
    model = LogisticRegression(
        alpha=0.5, fit_intercept=False, learning_rate=0.2, max_iter=7, tol=1e-3
    )

    assert model.get_params() == {
        "alpha": 0.5,
        "fit_intercept": False,
        "learning_rate": 0.2,
        "max_iter": 7,
        "tol": 1e-3,
    }
