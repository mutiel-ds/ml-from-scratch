"""Tests for mlfs.linear_model.LogisticRegression."""

import numpy as np
import pytest

from mlfs.linear_model import LogisticRegression
from mlfs.linear_model._logistic_regression import _encode_labels, _gradient, _sigmoid

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


# --- Gradient ---------------------------------------------------------------


def test_gradient_at_zero_params_without_penalty() -> None:
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, 0.0])
    params = np.array([0.0, 0.0])

    grad = _gradient(params, X, y, alpha=0.0, fit_intercept=True)

    # p = 0.5 everywhere, so p - y = [-0.5, 0.5];
    # d/dw = (1*(-0.5) + (-1)*0.5) / 2 = -0.5;  d/db = (-0.5 + 0.5) / 2 = 0.
    np.testing.assert_allclose(grad, [-0.5, 0.0], atol=1e-12)


def test_gradient_penalizes_the_weight_but_not_the_intercept() -> None:
    X = np.array([[2.0]])
    y = np.array([1.0])
    params = np.array([1.0, -2.0])

    grad = _gradient(params, X, y, alpha=3.0, fit_intercept=True)

    # p = 0.5, so p - y = [-0.5];
    # d/dw = 2*(-0.5) + 3*1= -1 + 3 = 2;  d/db = -0.5 = -0.5.
    np.testing.assert_allclose(grad, [2.0, -0.5], atol=1e-12)


def test_gradient_without_intercept_returns_only_weight_gradient() -> None:
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, 0.0])
    params = np.array([0.0])

    grad = _gradient(params, X, y, alpha=0.0, fit_intercept=False)

    # Same data as the first test: p - y = [-0.5, 0.5] and
    # d/dw = (1*(-0.5) + (-1)*0.5) / 2 = -0.5. There is no intercept entry,
    # so the result has shape (1,), not (2,).
    np.testing.assert_allclose(grad, [-0.5])


# --- Label encoding ---------------------------------------------------------


@pytest.mark.parametrize(
    ("y", "expected_classes", "expected_encoded"),
    [
        (["no", "yes", "yes", "no"], ["no", "yes"], [0, 1, 1, 0]),
        (["b", "a", "b", "b"], ["a", "b"], [1, 0, 1, 1]),
        ([-1, 1, 1, -1], [-1, 1], [0, 1, 1, 0]),
        ([5, 2, 2, 5], [2, 5], [1, 0, 0, 1]),
        ([False, True, True, False], [False, True], [0, 1, 1, 0]),
    ],
    ids=["strings", "unsorted-strings", "minus-one-one", "unsorted-ints", "bools"],
)
def test_encode_labels_maps_second_sorted_label_to_one(
    y: list[object], expected_classes: list[object], expected_encoded: list[int]
) -> None:
    classes, y_encoded = _encode_labels(np.asarray(y))

    # classes are the sorted unique labels; the larger label is the positive
    # class (1) and the smaller one is the negative class (0).
    np.testing.assert_array_equal(classes, expected_classes)
    np.testing.assert_array_equal(y_encoded, expected_encoded)
    assert y_encoded.dtype == np.float64


@pytest.mark.parametrize(
    ("y", "message"),
    [
        ([1, 1, 1], r"got 1: \[1\]"),
        ([0, 1, 2], r"got 3: \[0, 1, 2\]"),
        (["a", "b", "c", "a"], "got 3"),
    ],
    ids=["one-class", "three-classes", "three-string-classes"],
)
def test_encode_labels_without_exactly_two_classes_raises_value_error(
    y: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _encode_labels(np.asarray(y))


@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
def test_encode_labels_with_non_finite_float_label_raises_value_error(
    bad: float,
) -> None:
    y = np.array([0.0, 1.0, bad])

    with pytest.raises(ValueError, match="NaN or infinite"):
        _encode_labels(y)


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
