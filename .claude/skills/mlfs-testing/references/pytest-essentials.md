# pytest essentials for mlfs

## Test shape

```python
def test_mean_squared_error_with_one_wrong_prediction_returns_mean_of_squares() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 5.0]

    result = mean_squared_error(y_true, y_pred)

    # (0^2 + 0^2 + 2^2) / 3 = 4 / 3
    assert result == pytest.approx(4 / 3)
```

- Arrange, Act, Assert, separated by blank lines.
- Type hints on test functions (`-> None`): mypy runs in strict mode on tests.
- No docstrings needed on tests (ruff's `D` rules are off for `tests/`); the
  name is the documentation.

## Naming

`test_<unit>_<condition>_<expected>`:

| Good | Bad |
|---|---|
| `test_check_array_with_nan_raises_value_error` | `test_nan` |
| `test_r2_score_with_constant_target_and_perfect_prediction_returns_one` | `test_r2_edge` |
| `test_fit_with_one_sample_raises_value_error` | `test_fit_2` |

Long names are fine; they show up in failure output.

## Exceptions

Always match part of the message, so the test fails if a *different* error
with the same type is raised:

```python
with pytest.raises(ValueError, match="NaN or infinite"):
    check_array([[np.nan]])
```

Escape regex metacharacters (`re.escape`) or use a raw string.

## Parametrize instead of loops

Use `pytest.param(..., id="...")` so each case has a readable name in the
output:

```python
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
    X = [[1.0, bad_value]]

    with pytest.raises(ValueError, match="NaN or infinite"):
        check_array(X)
```

Parametrize only cases that share **the same behavior and assertion**. If the
expected outcome differs between cases, write separate tests.

## Fixtures

- Function scope (the default) unless setup is expensive; data generation here
  is cheap, so module or session scope is rarely justified.
- A fixture returns a fresh object per test, so tests cannot affect each other.
- Name fixtures after what they are: `linear_data`, not `data`.
- Fixtures shared by several files go in `tests/conftest.py`; otherwise keep
  them in the test file.

```python
@pytest.fixture
def linear_data() -> tuple[FloatArray, FloatArray]:
    """y = 2 * x0 - 3 * x1 + 1, with no noise."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = 2 * X[:, 0] - 3 * X[:, 1] + 1
    return X, y
```

## Test classes

Not needed. Group with section comments (`# --- check_array ---`) in a flat
module; it keeps names greppable and fixtures simple.
