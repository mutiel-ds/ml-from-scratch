"""Tests for mlfs.utils.validation."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from mlfs._typing import FloatArray
from mlfs.utils.validation import (
    check_array,
    check_n_features,
    check_random_state,
    check_X_y,
)

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
finite_matrices = hnp.arrays(
    dtype=np.float64,
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=10),
    elements=finite_floats,
)


# --- check_array: valid input -----------------------------------------------


def test_check_array_with_int_lists_returns_same_values_as_float64() -> None:
    X = [[1, 2], [3, 4]]

    result = check_array(X)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])


def test_check_array_with_single_sample_and_feature_is_accepted() -> None:
    X = [[5.0]]

    result = check_array(X)

    assert result.shape == (1, 1)


def test_check_array_with_ensure_2d_false_accepts_1d_input() -> None:
    y = [1.0, 2.0, 3.0]

    result = check_array(y, ensure_2d=False)

    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_check_array_with_ensure_2d_false_accepts_single_value() -> None:
    y = [7.0]

    result = check_array(y, ensure_2d=False)

    assert result.shape == (1,)


def test_check_array_does_not_modify_input() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_before = X.copy()

    check_array(X)

    np.testing.assert_array_equal(X, X_before)


@given(finite_matrices)
def test_check_array_with_any_finite_matrix_returns_equal_values(
    X: FloatArray,
) -> None:
    result = check_array(X)

    np.testing.assert_array_equal(result, X)


# --- check_array: invalid input ---------------------------------------------


def test_check_array_with_1d_input_raises_value_error() -> None:
    X = [1.0, 2.0]

    with pytest.raises(ValueError, match=r"Expected a 2D array, got a 1D array"):
        check_array(X)


def test_check_array_with_3d_input_raises_value_error() -> None:
    X = np.ones((2, 2, 2))

    with pytest.raises(ValueError, match=r"Expected a 2D array, got a 3D array"):
        check_array(X)


def test_check_array_with_ensure_2d_false_and_2d_input_raises_value_error() -> None:
    y = [[1.0], [2.0]]

    with pytest.raises(ValueError, match=r"Expected a 1D array, got a 2D array"):
        check_array(y, ensure_2d=False)


def test_check_array_with_zero_samples_raises_value_error() -> None:
    X = np.empty((0, 3))

    with pytest.raises(
        ValueError, match=r"^Input contains 0 samples; at least 1 is required\.$"
    ):
        check_array(X)


def test_check_array_with_zero_features_raises_value_error() -> None:
    X = np.empty((3, 0))

    with pytest.raises(
        ValueError, match=r"^Input contains 0 features; at least 1 is required\.$"
    ):
        check_array(X)


def test_check_array_with_empty_1d_input_raises_value_error() -> None:
    y: list[float] = []

    with pytest.raises(ValueError, match="0 samples"):
        check_array(y, ensure_2d=False)


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="positive-inf"),
        pytest.param(-np.inf, id="negative-inf"),
    ],
)
def test_check_array_with_non_finite_value_raises_value_error(
    bad_value: float,
) -> None:
    X = [[1.0, 2.0], [3.0, bad_value]]

    with pytest.raises(ValueError, match=r"^Input contains NaN or infinite values\.$"):
        check_array(X)


@given(
    finite_matrices,
    st.integers(min_value=0),
    st.sampled_from([np.nan, np.inf, -np.inf]),
)
def test_check_array_with_any_non_finite_entry_raises_value_error(
    X: FloatArray, position: int, bad_value: float
) -> None:
    X = X.copy()
    X.flat[position % X.size] = bad_value

    with pytest.raises(ValueError, match="NaN or infinite"):
        check_array(X)


def test_check_array_with_non_numeric_strings_raises_value_error() -> None:
    X = [["a", "b"]]

    with pytest.raises(ValueError, match="could not be converted to float64"):
        check_array(X)


def test_check_array_with_ragged_rows_raises_value_error() -> None:
    X = [[1.0, 2.0], [3.0]]

    with pytest.raises(ValueError, match="could not be converted to float64"):
        check_array(X)


# --- check_X_y ----------------------------------------------------------------


def test_check_X_y_with_numeric_target_returns_float64_target() -> None:
    X = [[1.0], [2.0]]
    y = [3, 4]

    _, y_checked = check_X_y(X, y)

    assert y_checked.dtype == np.float64
    np.testing.assert_array_equal(y_checked, [3.0, 4.0])


def test_check_X_y_returns_validated_features() -> None:
    X = [[1, 2], [3, 4]]
    y = [0.0, 1.0]

    X_checked, _ = check_X_y(X, y)

    assert X_checked.dtype == np.float64
    np.testing.assert_array_equal(X_checked, [[1.0, 2.0], [3.0, 4.0]])


def test_check_X_y_with_y_numeric_false_keeps_string_labels() -> None:
    X = [[1.0], [2.0]]
    y = ["cat", "dog"]

    _, y_checked = check_X_y(X, y, y_numeric=False)

    assert y_checked.tolist() == ["cat", "dog"]


def test_check_X_y_with_nan_in_numeric_target_raises_value_error() -> None:
    X = [[1.0], [2.0]]
    y = [1.0, np.nan]

    with pytest.raises(ValueError, match="NaN or infinite"):
        check_X_y(X, y)


def test_check_X_y_with_2d_numeric_target_raises_value_error() -> None:
    X = [[1.0], [2.0]]
    y = [[1.0], [2.0]]

    with pytest.raises(ValueError, match="Expected a 1D array"):
        check_X_y(X, y)


def test_check_X_y_with_2d_labels_raises_value_error() -> None:
    X = [[1.0], [2.0]]
    y = [["a"], ["b"]]

    with pytest.raises(ValueError, match=r"Expected y to be a 1D array"):
        check_X_y(X, y, y_numeric=False)


@pytest.mark.parametrize(
    "y_numeric", [pytest.param(True, id="numeric"), pytest.param(False, id="labels")]
)
def test_check_X_y_with_fewer_targets_than_samples_raises_value_error(
    y_numeric: bool,
) -> None:
    X = [[1.0], [2.0], [3.0]]
    y = [1, 2]

    with pytest.raises(ValueError, match=r"inconsistent numbers of samples: 3 != 2"):
        check_X_y(X, y, y_numeric=y_numeric)


def test_check_X_y_with_more_targets_than_samples_raises_value_error() -> None:
    X = [[1.0], [2.0]]
    y = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match=r"inconsistent numbers of samples: 2 != 3"):
        check_X_y(X, y)


def test_check_X_y_with_invalid_features_raises_value_error() -> None:
    X = [1.0, 2.0]
    y = [1.0, 2.0]

    with pytest.raises(ValueError, match="Expected a 2D array"):
        check_X_y(X, y)


# --- check_n_features ---------------------------------------------------------


def test_check_n_features_with_matching_count_does_not_raise() -> None:
    X = np.ones((4, 3))

    check_n_features(X, 3)


@pytest.mark.parametrize(
    ("n_columns", "n_features_in"),
    [pytest.param(2, 3, id="fewer"), pytest.param(4, 3, id="more")],
)
def test_check_n_features_with_mismatched_count_raises_value_error(
    n_columns: int, n_features_in: int
) -> None:
    X = np.ones((4, n_columns))

    with pytest.raises(
        ValueError,
        match=rf"X has {n_columns} features, .* fitted with {n_features_in} features",
    ):
        check_n_features(X, n_features_in)


# --- check_random_state -------------------------------------------------------


def test_check_random_state_with_same_int_gives_same_numbers() -> None:
    first = check_random_state(42).normal(size=5)
    second = check_random_state(42).normal(size=5)

    np.testing.assert_array_equal(first, second)


def test_check_random_state_with_int_matches_default_rng() -> None:
    result = check_random_state(7).normal(size=5)

    np.testing.assert_array_equal(result, np.random.default_rng(7).normal(size=5))


def test_check_random_state_with_numpy_integer_matches_default_rng() -> None:
    result = check_random_state(np.int64(7)).normal(size=5)  # type: ignore[arg-type]

    np.testing.assert_array_equal(result, np.random.default_rng(7).normal(size=5))


def test_check_random_state_with_generator_returns_same_object() -> None:
    rng = np.random.default_rng(0)

    result = check_random_state(rng)

    assert result is rng


def test_check_random_state_with_none_returns_new_generator() -> None:
    result = check_random_state(None)

    assert isinstance(result, np.random.Generator)


@pytest.mark.parametrize(
    ("seed", "type_name"),
    [
        pytest.param("42", "str", id="string"),
        pytest.param(1.5, "float", id="float"),
        pytest.param(np.random.RandomState(0), "RandomState", id="legacy"),
    ],
)
def test_check_random_state_with_invalid_type_raises_value_error_naming_it(
    seed: object, type_name: str
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"^random_state must be None, an int or a numpy\.random\.Generator, "
        rf"got {type_name}\.$",
    ):
        check_random_state(seed)  # type: ignore[arg-type]
