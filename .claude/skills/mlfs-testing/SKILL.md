---
name: mlfs-testing
description: How to write and verify tests in ml-from-scratch (mlfs), a NumPy machine learning library. Use whenever writing, rewriting or reviewing tests in this repository, including the tests for a new algorithm. Covers the trust rules, test file layout, hand-checkable known answers, float assertions, scikit-learn as a reference, property-based testing with Hypothesis, and mutation testing with mutmut.
---

# Testing patterns for mlfs

Adapted from the MIT-licensed `python-testing-patterns` skill in
[NickCrew/Claude-Cortex](https://github.com/NickCrew/Claude-Cortex/tree/main/skills/python-testing-patterns),
narrowed to this project: a pure NumPy library with no I/O, network, database
or async code, so mocking, monkeypatching and integration-test patterns are
deliberately left out.

## Why this skill exists

The owner does not trust AI-generated tests by default, for good reason: they
often pass without proving anything. Every test written here must be
**auditable by a human in under a minute** and must be **shown to fail when
the code is wrong**. Coverage only proves a line ran; it is not evidence.

## The trust rules

1. **Expected values come from outside the code under test.** Use, in order of
   preference:
   - a hand-checkable known answer, with the arithmetic in a comment;
   - a mathematical property that must hold (see Hypothesis below);
   - scikit-learn as an independent reference implementation.
   Never compute the expected value with the same formula as the
   implementation, and never paste the function's own output in as the
   expected value.
2. **One behavior per test**, named `test_<unit>_<condition>_<expected>`, for
   example `test_check_array_with_nan_raises_value_error`.
3. **Arrange / Act / Assert**, separated by blank lines (add `# Arrange`
   comments only when a section is long enough to need them).
4. **No logic in tests.** No loops, no `if`, no `try`. Use
   `@pytest.mark.parametrize` with readable `id`s instead of loops.
5. **Assert specifically.** `pytest.raises(ValueError, match=...)` with a
   message fragment, exact values or `assert_allclose` with a stated
   tolerance, never `assert result is not None` or a bare `except`.
6. **Tolerances are justified.** Default `rtol=1e-7` for closed-form math;
   anything looser needs a comment saying why (iterative solver, etc.).
7. **Deterministic.** Seed every generator with `np.random.default_rng(<int>)`.
   Never use the global `np.random` state.
8. **Do not mock your own code.** There is nothing external to mock here.
9. **Every new test must be able to fail.** Prove it with mutation testing
   (below): a mutant that survives means a missing or weak test.

## Layout

```
tests/
├── conftest.py              # Hypothesis profiles, shared fixtures
├── toy_estimators.py        # tiny contract-following estimators (import directly)
├── test_base.py             # mirrors src/mlfs/base.py
├── metrics/test_metrics.py
├── utils/test_validation.py
└── linear_model/test_linear_regression.py
```

Tests mirror `src/mlfs`. No `unit/`/`integration/` split: everything is a fast
unit test. Shared fixtures live in `tests/conftest.py`; a fixture used by only
one file stays in that file. Shared helper classes live in plain modules such as
`tests/toy_estimators.py` (`tests/` is on the import path, so
`from toy_estimators import MeanRegressor` works); a helper used by only one
file stays in that file.

## Anatomy of an algorithm test module

Group tests in this order with section comments:

1. **Contract:** `check_estimator(Estimator())` (see `CLAUDE.md`).
2. **Known answers:** tiny inputs with hand-derived outputs.
3. **Reference:** agreement with scikit-learn on a seeded random dataset.
4. **Properties:** Hypothesis tests for invariants.
5. **Errors and edge cases:** invalid hyperparameters, one sample, one
   feature, algorithm-specific failure modes.

## Workflow

1. Read the source file under test and list its behaviors and branches.
2. Write the tests, following the rules above.
3. Run `uv run pytest --cov` and get 100% line and branch coverage.
4. Run mutation testing on the module and kill or justify every survivor.
5. Run `uv run ruff format`, `uv run ruff check`, `uv run mypy`.

## Commands

```bash
uv run pytest                              # all tests
uv run pytest tests/utils -v               # one folder, verbose
uv run pytest -k nan                       # tests whose name contains "nan"
uv run pytest --lf -x                      # rerun last failures, stop at first
uv run pytest --cov                        # coverage (configured in pyproject)
uv run mutmut run                          # mutation testing
uv run mutmut results                      # list surviving mutants
uv run mutmut show <mutant-name>           # see one mutant's diff
```

## References

Load only what the task needs:

| Topic | File |
|---|---|
| pytest structure, naming, parametrize, fixtures, conftest | `references/pytest-essentials.md` |
| Float comparisons, known answers, scikit-learn as reference | `references/numerical-assertions.md` |
| Hypothesis and `hypothesis.extra.numpy`, ML invariants | `references/property-based-testing.md` |
| Mutation testing with mutmut and handling survivors | `references/mutation-testing.md` |
| Coverage rules and when `pragma: no cover` is allowed | `references/coverage.md` |

## Review checklist

Before reporting tests as done, confirm each item:

- [ ] Every expected value is hand-derived, a property, or from scikit-learn.
- [ ] Each test checks one behavior and its name says which.
- [ ] No loops, conditionals or try/except inside tests.
- [ ] Every `pytest.raises` has a `match=`.
- [ ] Every non-default tolerance has a comment.
- [ ] All randomness is seeded.
- [ ] 100% line and branch coverage of the module.
- [ ] Mutation testing run; survivors killed or listed with a reason.
