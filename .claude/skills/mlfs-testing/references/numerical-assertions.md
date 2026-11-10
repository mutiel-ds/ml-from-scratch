# Numerical assertions

## Choosing the assertion

| Comparing | Use |
|---|---|
| Two arrays of floats | `np.testing.assert_allclose(actual, expected, rtol=..., atol=...)` |
| Two arrays that must be bit-identical (copies, labels, ints) | `np.testing.assert_array_equal` |
| A single float | `assert actual == pytest.approx(expected)` |
| A single exact value (int, str, bool, shape, dtype) | `assert actual == expected` |

Never compare floats with `==` unless the value is exactly representable and
produced without arithmetic (for example a copy).

## Tolerances

- `assert_allclose` defaults to `rtol=1e-7, atol=0`. Keep that for closed-form
  computations on well-conditioned data.
- Use `atol` when the expected value is 0 (`rtol` is meaningless there):
  `assert_allclose(residuals, 0.0, atol=1e-12)`. This also applies when only
  **some** elements are 0, such as a zero intercept or one zero weight: the
  allowed error there is `rtol * 0 = 0`, so a harmless `1e-16` rounding error
  fails the test. Pass both, for example `rtol=1e-7, atol=1e-12`.
- Iterative methods (gradient descent) converge approximately: set the
  tolerance from the algorithm's stopping criterion and **write the reason in
  a comment**. Loosening a tolerance until a test passes is not a reason.

## Known answers (the strongest evidence)

Pick inputs small enough to solve on paper, and put the derivation next to
the assertion:

```python
def test_r2_score_with_mean_prediction_returns_zero() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 2.0]  # always the mean of y_true

    result = r2_score(y_true, y_pred)

    # SS_res = 1 + 0 + 1 = 2, SS_tot = 1 + 0 + 1 = 2, R^2 = 1 - 2/2 = 0
    assert result == pytest.approx(0.0)
```

Good sources of known answers:

- data lying exactly on a line, plane or cluster centers;
- symmetric data, where the answer follows from symmetry;
- degenerate cases: one sample, constant target, identical features;
- textbook worked examples (cite them in a comment).

## scikit-learn as a reference

scikit-learn is an independent implementation, so agreement with it is real
evidence. Rules:

- Import it **only in tests** (it is a dev dependency).
- Use a seeded random dataset large enough to be non-trivial (for example
  50 samples, 3 features).
- Compare the public outputs (`predict`, `transform`) and the fitted
  attributes that both share (`coef_`, `intercept_`).
- Match hyperparameters explicitly; do not rely on both libraries having the
  same defaults.
- Known answers come first. The scikit-learn test is a second check, not a
  substitute.

```python
def test_linear_regression_matches_sklearn(linear_data_with_noise) -> None:
    X, y = linear_data_with_noise

    ours = LinearRegression().fit(X, y)
    reference = sklearn.linear_model.LinearRegression().fit(X, y)

    np.testing.assert_allclose(ours.coef_, reference.coef_, rtol=1e-7)
    np.testing.assert_allclose(ours.intercept_, reference.intercept_, rtol=1e-7)
```

## Anti-patterns

```python
# Same formula as the implementation: proves nothing.
expected = np.mean((y_true - y_pred) ** 2)
assert mean_squared_error(y_true, y_pred) == pytest.approx(expected)

# Output pasted in as the expected value: freezes current bugs.
assert model.coef_[0] == pytest.approx(1.8371920263)

# Tolerance so loose almost anything passes.
np.testing.assert_allclose(ours, reference, rtol=0.5)
```
