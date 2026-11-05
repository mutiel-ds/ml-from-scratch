# Mutation testing with mutmut

## What it proves

mutmut makes small changes ("mutants") to the source code, such as `>` to
`>=`, `+` to `-`, `and` to `or`, a number `0` to `1`, or a string to
`"XX...XX"`, and runs the tests against each one.

- **Killed:** some test failed. Good: the tests notice that bug.
- **Survived:** every test still passed. The tests would not notice that bug.

Coverage says a line *ran*; mutation testing says the tests *check* it. It is
the main evidence that the tests are trustworthy.

## Running it

Configuration lives in `[tool.mutmut]` in `pyproject.toml`. Mutants are
written to `mutants/` (git-ignored).

```bash
uv run mutmut run                       # mutate all of src/mlfs
uv run mutmut run "mlfs.utils.validation*"   # only one module
uv run mutmut results                   # list survivors
uv run mutmut show <mutant-name>        # show the diff of one mutant
```

Run it on the module you changed, not the whole package, while iterating.

## Handling survivors

For each surviving mutant, decide which case it is:

1. **Missing test.** The mutant changes behavior and no test checks it. Write
   the test that kills it (often a boundary: exactly 1 sample, exactly equal
   shapes).
2. **Weak assertion.** A test runs the code but asserts too little (no
   `match=`, a loose tolerance). Tighten it.
3. **Equivalent mutant.** The change cannot alter behavior, for example
   mutating a default that is always overridden, or `x * 1` style no-ops.
   These cannot be killed. List them in your report with a one-line reason.

Do not change the source code only to make a mutant easier to kill, unless the
change also makes the code clearer.

## Reporting

Report the final counts (killed / survived / total) per module, and every
remaining survivor with its classification.
