# ml-from-scratch: working rules

Classic machine learning algorithms implemented from scratch in NumPy, built
like a real, installable library. This is a public portfolio repository, read
by recruiters and interviewers: code, docs, comments and commit messages must
be in **English**, clear and professional.

The distribution is `ml-from-scratch`; the import package is `mlfs`
("Machine Learning From Scratch").

## How work is organized

- One **coordinator** session owns the repository plan (`docs/ROADMAP.md`),
  shared code and docs.
- Each algorithm is built by its own **algorithm session**, in its own git
  worktree and branch (the app names it `claude/<algorithm>-<suffix>`), from a
  brief written by the coordinator.
- **Never commit, push, or create GitHub repositories, branches or PRs.** The
  owner (Mario) reviews and commits all work himself. Leave changes in the
  working tree.

### Files an algorithm session may touch

| Allowed | Example |
|---|---|
| Its implementation module | `src/mlfs/linear_model/_linear_regression.py` |
| Its test module | `tests/linear_model/test_linear_regression.py` |
| Its subpackage `__init__.py`, **only** to add its own import and `__all__` entry (create the file if the subpackage is new) | `src/mlfs/linear_model/__init__.py` |
| Its example script, if the brief asks for one | `examples/linear_regression.py` |

Everything else is owned by the coordinator: `pyproject.toml`, `uv.lock`,
`CLAUDE.md`, `README.md`, `docs/`, `.github/`, `src/mlfs/base.py`,
`src/mlfs/_typing.py`, `src/mlfs/exceptions.py`, `src/mlfs/utils/`,
`src/mlfs/metrics/`. If you need a change there (a new metric, a validation
helper, a dependency), **do not make it**: describe it in your final report.

## Commands

```bash
uv run pytest                  # tests
uv run pytest --cov            # tests with coverage
uv run ruff format             # format
uv run ruff check --fix        # lint
uv run mypy                    # strict type checking
```

All four must pass before work counts as done.

## The estimator contract

Read `src/mlfs/base.py` and `src/mlfs/utils/validation.py` before writing an
estimator. The rules:

1. Subclass `BaseRegressor`, `BaseClassifier` or `BaseTransformer` from
   `mlfs.base`.
2. `__init__` takes every hyperparameter as an explicit keyword argument with a
   default (no `*args`/`**kwargs`) and stores each one **unchanged** under the
   same name. No validation or computation in `__init__`.
3. `fit(X, y)`:
   - validates hyperparameters (raise `ValueError` with a helpful message);
     check floats with `np.isfinite` first, because `nan < 0` is False and
     would slip through a plain range check;
   - turns predictable numerical failures (for example `np.linalg.LinAlgError`
     on a singular system) into a `ValueError` that says how to fix them;
   - validates data with `check_X_y` (or `check_array` for transformers);
   - never modifies the caller's arrays in place;
   - stores everything it learns in attributes with a trailing underscore
     (`coef_`, `intercept_`), always including `n_features_in_`;
   - returns `self`.
4. `predict` / `transform` call `check_is_fitted(self)`, then `check_array`,
   then `check_n_features(X, self.n_features_in_)`.
5. Randomness only through a `random_state` hyperparameter converted with
   `check_random_state`. Never use the global `np.random` state.
6. Gradient-based solvers use `gradient_descent` and
   `check_gradient_descent_params` from `mlfs.utils._optimize`: pass the
   gradient of your loss and call it directly from `fit`, so its
   `ConvergenceWarning` points at the user's line. Do not write a new loop.
7. NumPy is the only runtime dependency. Do not import scikit-learn, SciPy or
   anything else from `src/`.
8. Prefer vectorized NumPy over Python loops over samples. Prefer numerically
   stable formulations (for example `np.linalg.lstsq` or `solve` over
   explicitly inverting a matrix).

## Code style

- Full type hints; `mypy --strict` must pass. Use the aliases in
  `mlfs._typing` (`ArrayLike` for inputs, `FloatArray` for float outputs).
- NumPy-style docstrings on every public class and function, with
  `Parameters`, `Returns`/`Attributes`, `Raises` where relevant, and
  `Examples` on estimators. The class docstring explains the **math** (LaTeX
  in `.. math::` blocks) and links to a reference if useful.
- `Examples` sections are run as doctests by `uv run pytest`
  (`--doctest-modules`), and the printed output must match exactly. Use
  small, hand-checkable numbers so the expected output is short and stable.
- Implementation modules are private (`_name.py`); users import from the
  subpackage (`from mlfs.linear_model import LinearRegression`).
- Comments explain *why*, not *what*.
- Scripts in `examples/` show how a user would use the library, so they import
  only the public API (never private modules such as `mlfs._typing`; use
  `numpy.typing` for type hints there).

## Testing standard

Use the project skill `.claude/skills/mlfs-testing` (load it before
writing any test). It defines the trust rules, float assertions, Hypothesis and
mutation testing. In short, each algorithm's test module must include:

1. **Contract:** `check_estimator(Estimator())` from
   `mlfs.utils.estimator_checks`, plus one call per important hyperparameter
   combination.
2. **Correctness against scikit-learn:** compare fitted attributes and
   predictions with the scikit-learn equivalent using `np.testing.assert_allclose`.
   scikit-learn is a dev-only dependency: it may be imported in tests, and in
   `examples/` only as an optional import (`try`/`except ImportError`), never
   from `src/`.
3. **Known answers:** at least one tiny hand-checkable case (for example data
   lying exactly on a line).
4. **Edge cases and errors:** invalid hyperparameters, a single sample or
   feature, and any algorithm-specific failure mode.
5. Determinism: seed every random generator in tests.

Reach 100% line and branch coverage of the new module, then run mutation
testing on it (`uv run mutmut run "mlfs.<module>*"`) and report survivors.

## Definition of done and final report

When finished, report back to the coordinator with:

- the files created or changed;
- the public API (class names, hyperparameters, fitted attributes);
- results of the four commands above;
- any design decision worth knowing, and any change you need in
  coordinator-owned files;
- a suggested commit message (Conventional Commits, for example
  `feat(linear_model): add LinearRegression`).
