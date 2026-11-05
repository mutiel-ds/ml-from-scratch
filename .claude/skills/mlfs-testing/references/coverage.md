# Coverage

Configured in `pyproject.toml` (`[tool.coverage.*]`): branch coverage is on,
and only `mlfs` is measured.

```bash
uv run pytest --cov                          # terminal report, missing lines
uv run pytest --cov --cov-report=html        # browsable report in htmlcov/
```

## Target

**100% line and branch coverage** for every module in `src/mlfs`. This is a
small library of pure functions, so anything less means untested behavior.

Coverage is a floor, not proof: a test without meaningful assertions still
counts as covering a line. Mutation testing (`mutation-testing.md`) is what
shows the covered lines are actually checked.

## `pragma: no cover`

Allowed only for lines that cannot run in a correct program, and always with a
reason on the same line:

```python
except PackageNotFoundError:  # pragma: no cover - only when run from a raw checkout
```

Not allowed for error branches that are merely inconvenient to trigger;
write the test instead.
