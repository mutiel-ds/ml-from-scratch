# ml-from-scratch

[![CI](https://github.com/mutiel-ds/ml-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/mutiel-ds/ml-from-scratch/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Typed](https://img.shields.io/badge/typing-mypy%20strict-informational)

Classic machine learning algorithms implemented **from scratch in NumPy**,
built like a real library: a consistent scikit-learn-style API, strict typing,
tests that check every algorithm against scikit-learn, and CI on every push.

The goal is to show how these algorithms actually work, in code that is clean
enough to reuse.

## Algorithms

| Algorithm | Import | Status |
|---|---|---|
| Linear regression | `mlfs.linear_model.LinearRegression` | Done |
| Ridge regression | `mlfs.linear_model.Ridge` | Planned |
| Logistic regression | `mlfs.linear_model.LogisticRegression` | Planned |
| k-nearest neighbors | `mlfs.neighbors` | Planned |
| k-means | `mlfs.cluster.KMeans` | Planned |
| Decision tree (CART) | `mlfs.tree` | Planned |
| Gaussian Naive Bayes | `mlfs.naive_bayes.GaussianNB` | Planned |
| PCA | `mlfs.decomposition.PCA` | Planned |

Also available: metrics (`mean_squared_error`, `r2_score`, `accuracy_score`),
input validation, and the estimator base classes. See
[docs/ROADMAP.md](docs/ROADMAP.md) for progress.

## Installation

Install straight from GitHub (pin a tag for reproducible projects):

```bash
uv add "ml-from-scratch @ git+https://github.com/mutiel-ds/ml-from-scratch"
```

or with pip:

```bash
pip install "git+https://github.com/mutiel-ds/ml-from-scratch"
```

The package is called `ml-from-scratch`, and you import it as `mlfs`
(Machine Learning From Scratch):

```python
from mlfs.linear_model import LinearRegression
from mlfs.metrics import r2_score

X = [[0.0], [1.0], [2.0], [3.0]]
y = [1.0, 3.0, 5.0, 7.0]  # y = 2x + 1

model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_)  # [2.] 1.0
print(r2_score(y, model.predict(X)))  # 1.0
```

[`examples/linear_regression.py`](examples/linear_regression.py) compares the
closed-form solver, gradient descent and a by-hand implementation, and times
them against scikit-learn.

## Design

Every estimator follows the same contract, defined in
[`mlfs/base.py`](src/mlfs/base.py):

- hyperparameters go in `__init__` and are stored unchanged;
- `fit(X, y)` validates its inputs, learns, and returns `self`;
- learned values live in attributes with a trailing underscore (`coef_`);
- using an estimator before `fit` raises `NotFittedError`.

The contract is enforced by tests, not just documented:
[`check_estimator`](src/mlfs/utils/estimator_checks.py) runs the same checks
against every algorithm, and it also works for your own estimators built on
`mlfs.base`.

NumPy is the only runtime dependency. scikit-learn is used only in the test
suite, as a reference to check results against, and optionally in the
examples for timing comparisons.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                        # create the environment
uv run pre-commit install      # run checks on every commit
uv run pytest --cov            # tests with coverage
uv run ruff check              # lint
uv run mypy                    # strict type checking
```

## Project layout

```
src/mlfs/
├── base.py                # estimator base classes and check_is_fitted
├── exceptions.py          # NotFittedError, ConvergenceWarning
├── linear_model/          # LinearRegression
├── metrics/               # regression and classification metrics
└── utils/
    ├── validation.py      # input validation shared by all estimators
    └── estimator_checks.py
tests/                     # mirrors src/mlfs
examples/                  # runnable scripts using the public API
```

## License

[MIT](LICENSE)
