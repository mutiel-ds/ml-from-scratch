# Property-based testing with Hypothesis

Example-based tests check the cases you thought of. Hypothesis generates many
inputs and checks a **property that must hold for all of them**, then shrinks
any failure to the smallest input that breaks it.

Use properties *in addition to* known answers, never instead of them.

## Configuration

Profiles are registered in `tests/conftest.py`:

- `default` (local runs): random examples, so new edge cases keep being found.
- `ci` (selected with `HYPOTHESIS_PROFILE=ci`): `derandomize=True`, so CI is
  reproducible and never flaky.

Failing examples are stored in `.hypothesis/` (git-ignored) and replayed first
on the next run.

## Generating NumPy data

```python
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# Finite, moderately sized floats: avoids overflow noise that is not a bug.
finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)

matrices = hnp.arrays(
    dtype=np.float64,
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=20),
    elements=finite_floats,
)


@given(matrices)
def test_check_array_with_finite_matrix_returns_equal_values(X: FloatArray) -> None:
    result = check_array(X)

    np.testing.assert_array_equal(result, X)
```

Bound the element range deliberately. Unbounded floats mostly test overflow,
which is NumPy's behavior, not ours. If a bound hides a real bug, write a
dedicated test for it.

## Properties that fit ML code

| Property | Example |
|---|---|
| Bounds | `0 <= accuracy <= 1`, `mse >= 0` |
| Identity / perfect case | `mse(y, y) == 0`, `r2(y, y) == 1` for non-constant `y` |
| Symmetry | `mse(a, b) == mse(b, a)` |
| Invariance | k-means labels unchanged by translating all points; tree predictions unchanged by permuting training rows |
| Equivariance | scaling `y` by `c` scales linear regression coefficients by `c` |
| Round trip | PCA `inverse_transform(transform(X)) ≈ X` with all components |
| Agreement | our result `≈` scikit-learn's for any generated well-conditioned input |
| Validation | any input containing NaN raises `ValueError` |

## Keep properties honest

- A property must not restate the implementation (`mse == mean(sq diff)` is
  the formula, not a property).
- Use `hypothesis.assume(...)` sparingly; if many examples are rejected,
  narrow the strategy instead.
- For floating point, compare with tolerances (`assert_allclose`), and pick
  element bounds that keep the math well-conditioned.
- Slow properties (fitting models) can lower `max_examples` with
  `@settings(max_examples=25)` and a comment explaining why.
