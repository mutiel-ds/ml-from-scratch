"""Tests for mlfs.metrics.

Expected values are derived by hand (arithmetic in comments), from
mathematical properties (Hypothesis), or from scikit-learn as an independent
reference.
"""

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn import metrics as sk_metrics

from mlfs._typing import ArrayLike, FloatArray, NDArray
from mlfs.metrics import accuracy_score, mean_squared_error, r2_score

# Bounded so that squaring and summing never overflows.
finite_floats = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
)
target_pairs = st.integers(min_value=1, max_value=30).flatmap(
    lambda n: st.tuples(
        hnp.arrays(np.float64, n, elements=finite_floats),
        hnp.arrays(np.float64, n, elements=finite_floats),
    )
)
targets = hnp.arrays(
    np.float64, st.integers(min_value=1, max_value=30), elements=finite_floats
)
label_pairs = st.integers(min_value=1, max_value=30).flatmap(
    lambda n: st.tuples(
        hnp.arrays(np.int64, n, elements=st.integers(0, 3)),
        hnp.arrays(np.int64, n, elements=st.integers(0, 3)),
    )
)


@pytest.fixture
def noisy_regression_targets() -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.3, size=50)
    return y_true, y_pred


# --- mean_squared_error: known answers ----------------------------------------


def test_mean_squared_error_with_one_wrong_prediction_returns_mean_of_squares() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 5.0]

    result = mean_squared_error(y_true, y_pred)

    # (0^2 + 0^2 + 2^2) / 3 = 4 / 3
    assert result == pytest.approx(4 / 3)


def test_mean_squared_error_with_mixed_errors_returns_mean_of_squares() -> None:
    y_true = [0.0, 0.0, 0.0, 0.0]
    y_pred = [1.0, -1.0, 3.0, 0.0]

    result = mean_squared_error(y_true, y_pred)

    # (1 + 1 + 9 + 0) / 4 = 11 / 4: signs must not cancel out
    assert result == pytest.approx(2.75)


def test_mean_squared_error_with_perfect_prediction_returns_zero() -> None:
    y_true = [1.5, -2.0, 3.0]

    result = mean_squared_error(y_true, y_true)

    assert result == 0.0


def test_mean_squared_error_accepts_integer_lists() -> None:
    y_true = [1, 2]
    y_pred = [3, 2]

    result = mean_squared_error(y_true, y_pred)

    # (2^2 + 0^2) / 2 = 2
    assert result == pytest.approx(2.0)


# --- mean_squared_error: reference and properties -----------------------------


def test_mean_squared_error_matches_sklearn(
    noisy_regression_targets: tuple[FloatArray, FloatArray],
) -> None:
    y_true, y_pred = noisy_regression_targets

    result = mean_squared_error(y_true, y_pred)

    assert result == pytest.approx(sk_metrics.mean_squared_error(y_true, y_pred))


@given(target_pairs)
def test_mean_squared_error_is_never_negative(
    pair: tuple[FloatArray, FloatArray],
) -> None:
    y_true, y_pred = pair

    result = mean_squared_error(y_true, y_pred)

    assert result >= 0.0


@given(target_pairs)
def test_mean_squared_error_is_symmetric(pair: tuple[FloatArray, FloatArray]) -> None:
    y_true, y_pred = pair

    forward = mean_squared_error(y_true, y_pred)
    backward = mean_squared_error(y_pred, y_true)

    assert forward == backward


@given(targets)
def test_mean_squared_error_of_target_with_itself_is_zero(y: FloatArray) -> None:
    result = mean_squared_error(y, y)

    assert result == 0.0


# --- r2_score: known answers --------------------------------------------------


def test_r2_score_with_perfect_prediction_returns_one() -> None:
    y_true = [1.0, 2.0, 3.0]

    result = r2_score(y_true, y_true)

    assert result == 1.0


def test_r2_score_with_mean_prediction_returns_zero() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 2.0]  # always the mean of y_true

    result = r2_score(y_true, y_pred)

    # SS_res = 1 + 0 + 1 = 2, SS_tot = 1 + 0 + 1 = 2, R^2 = 1 - 2/2 = 0
    assert result == pytest.approx(0.0)


def test_r2_score_with_partial_fit_returns_fraction_explained() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]

    result = r2_score(y_true, y_pred)

    # SS_res = 0 + 0 + 1 = 1, SS_tot = 2, R^2 = 1 - 1/2 = 0.5
    assert result == pytest.approx(0.5)


def test_r2_score_worse_than_mean_returns_negative() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [3.0, 2.0, 1.0]

    result = r2_score(y_true, y_pred)

    # SS_res = 4 + 0 + 4 = 8, SS_tot = 2, R^2 = 1 - 8/2 = -3
    assert result == pytest.approx(-3.0)


def test_r2_score_with_constant_target_and_perfect_prediction_returns_one() -> None:
    y_true = [2.0, 2.0]

    result = r2_score(y_true, y_true)

    # SS_tot = 0, so the formula is undefined; the convention is 1.0
    assert result == 1.0


def test_r2_score_with_constant_target_and_wrong_prediction_returns_zero() -> None:
    y_true = [2.0, 2.0]
    y_pred = [1.0, 3.0]

    result = r2_score(y_true, y_pred)

    # SS_tot = 0, so the formula is undefined; the convention is 0.0
    assert result == 0.0


# --- r2_score: reference and properties ---------------------------------------


def test_r2_score_matches_sklearn(
    noisy_regression_targets: tuple[FloatArray, FloatArray],
) -> None:
    y_true, y_pred = noisy_regression_targets

    result = r2_score(y_true, y_pred)

    assert result == pytest.approx(sk_metrics.r2_score(y_true, y_pred))


@pytest.mark.parametrize(
    "y_pred",
    [pytest.param([2.0, 2.0], id="perfect"), pytest.param([1.0, 3.0], id="wrong")],
)
def test_r2_score_with_constant_target_matches_sklearn(y_pred: list[float]) -> None:
    y_true = [2.0, 2.0]

    result = r2_score(y_true, y_pred)

    assert result == sk_metrics.r2_score(y_true, y_pred)


@given(target_pairs)
def test_r2_score_is_never_above_one(pair: tuple[FloatArray, FloatArray]) -> None:
    y_true, y_pred = pair

    result = r2_score(y_true, y_pred)

    assert result <= 1.0


@given(targets)
def test_r2_score_of_target_with_itself_is_one(y: FloatArray) -> None:
    result = r2_score(y, y)

    assert result == 1.0


# --- regression metrics: invalid input ----------------------------------------

RegressionMetric = Callable[[ArrayLike, ArrayLike], float]

regression_metrics = pytest.mark.parametrize(
    "metric",
    [
        pytest.param(mean_squared_error, id="mse"),
        pytest.param(r2_score, id="r2"),
    ],
)


@regression_metrics
def test_regression_metric_with_different_lengths_raises_value_error(
    metric: RegressionMetric,
) -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0]

    with pytest.raises(
        ValueError,
        match=r"^y_true and y_pred have different shapes: \(3,\) != \(2,\)\.$",
    ):
        metric(y_true, y_pred)


@regression_metrics
def test_regression_metric_with_nan_raises_value_error(
    metric: RegressionMetric,
) -> None:
    y_true = [1.0, np.nan]
    y_pred = [1.0, 2.0]

    with pytest.raises(ValueError, match="NaN or infinite"):
        metric(y_true, y_pred)


@regression_metrics
def test_regression_metric_with_2d_input_raises_value_error(
    metric: RegressionMetric,
) -> None:
    y_true = [[1.0], [2.0]]
    y_pred = [[1.0], [2.0]]

    with pytest.raises(ValueError, match="Expected a 1D array"):
        metric(y_true, y_pred)


@regression_metrics
def test_regression_metric_with_empty_input_raises_value_error(
    metric: RegressionMetric,
) -> None:
    with pytest.raises(ValueError, match="0 samples"):
        metric([], [])


# --- accuracy_score: known answers --------------------------------------------


def test_accuracy_score_with_string_labels_returns_fraction_correct() -> None:
    y_true = ["a", "b", "b", "c", "a"]
    y_pred = ["a", "b", "c", "c", "b"]

    result = accuracy_score(y_true, y_pred)

    # matches at positions 0, 1, 3 -> 3 / 5
    assert result == pytest.approx(0.6)


def test_accuracy_score_with_all_correct_returns_one() -> None:
    y_true = [0, 1, 1]

    result = accuracy_score(y_true, y_true)

    assert result == 1.0


def test_accuracy_score_with_all_wrong_returns_zero() -> None:
    y_true = [0, 1, 1]
    y_pred = [1, 0, 0]

    result = accuracy_score(y_true, y_pred)

    assert result == 0.0


def test_accuracy_score_with_single_sample_is_accepted() -> None:
    result = accuracy_score([1], [1])

    assert result == 1.0


# --- accuracy_score: reference and properties ---------------------------------


def test_accuracy_score_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=50)
    y_pred = rng.integers(0, 3, size=50)

    result = accuracy_score(y_true, y_pred)

    assert result == pytest.approx(sk_metrics.accuracy_score(y_true, y_pred))


@given(label_pairs)
def test_accuracy_score_is_between_zero_and_one(
    pair: tuple[NDArray[np.int64], NDArray[np.int64]],
) -> None:
    y_true, y_pred = pair

    result = accuracy_score(y_true, y_pred)

    assert 0.0 <= result <= 1.0


@given(label_pairs)
def test_accuracy_score_is_symmetric(
    pair: tuple[NDArray[np.int64], NDArray[np.int64]],
) -> None:
    y_true, y_pred = pair

    forward = accuracy_score(y_true, y_pred)
    backward = accuracy_score(y_pred, y_true)

    assert forward == backward


# --- accuracy_score: invalid input --------------------------------------------


def test_accuracy_score_with_different_lengths_raises_value_error() -> None:
    y_true = [1, 2]
    y_pred = [1]

    with pytest.raises(
        ValueError,
        match=r"^y_true and y_pred must be 1D with the same shape, "
        r"got \(2,\) and \(1,\)\.$",
    ):
        accuracy_score(y_true, y_pred)


def test_accuracy_score_with_2d_labels_raises_value_error() -> None:
    y_true = [[1], [2]]
    y_pred = [[1], [2]]

    with pytest.raises(
        ValueError,
        match=r"^y_true and y_pred must be 1D with the same shape, "
        r"got \(2, 1\) and \(2, 1\)\.$",
    ):
        accuracy_score(y_true, y_pred)


def test_accuracy_score_with_empty_input_raises_value_error() -> None:
    with pytest.raises(
        ValueError, match=r"^y_true contains 0 samples; at least 1 is required\.$"
    ):
        accuracy_score([], [])
