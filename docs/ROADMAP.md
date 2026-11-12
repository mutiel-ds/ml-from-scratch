# Roadmap

Maintained by the coordinator session. Each algorithm is built in its own
session, worktree and branch, following the rules in `CLAUDE.md`. The branch
column gives the short name; the app creates the branch as
`claude/<name>-<suffix>`, which is merged into `main` with `--no-ff`.

## Phase 0: foundation

| Item | Status |
|---|---|
| Installable package (`src/mlfs`, hatchling, `py.typed`, Python 3.11+) | Done |
| Estimator base classes and `NotFittedError` (`mlfs.base`) | Done |
| Input validation helpers (`mlfs.utils.validation`) | Done |
| Shared contract checks (`mlfs.utils.estimator_checks`) | Done |
| Metrics: `mean_squared_error`, `r2_score`, `accuracy_score` | Done |
| Tooling: ruff, mypy (strict), pytest + coverage, pre-commit, GitHub Actions | Done |
| Runtime and dev dependencies installed with uv | Done |

## Phase 1: reference implementation

| Algorithm | Module | Branch | Status |
|---|---|---|---|
| Linear regression (least squares + gradient descent) | `mlfs.linear_model` | `linear-regression` | Done |

What the first session taught us, now reflected in `CLAUDE.md` and the
`mlfs-testing` skill:

- `ConvergenceWarning` (in `mlfs.exceptions`) is the shared warning for
  iterative solvers that hit `max_iter`.
- Examples use only the public API and may import scikit-learn optionally.
- `assert_allclose` needs `atol` whenever an expected value is 0, and tests
  that inspect the call stack are skipped under mutmut and checked by hand.

## Phase 2: parallel algorithm sessions

| Algorithm | Module | Branch | Status |
|---|---|---|---|
| Ridge regression (`solve`, augmented `lstsq`, gradient descent) | `mlfs.linear_model` | `ridge` | Done |
| Logistic regression (binary, L2, gradient descent) | `mlfs.linear_model` | `logistic-regression` | Done |
| Multiclass logistic regression (softmax; `coef_` becomes `(n_classes, n_features)`) | `mlfs.linear_model` | `logistic-multiclass` | Next |
| k-nearest neighbors | `mlfs.neighbors` | `knn` | Planned |
| k-means (k-means++ init) | `mlfs.cluster` | `kmeans` | Planned |
| Decision tree (CART) | `mlfs.tree` | `decision-tree` | Planned |
| Gaussian Naive Bayes | `mlfs.naive_bayes` | `naive-bayes` | Planned |
| PCA | `mlfs.decomposition` | `pca` | Planned |
| Train/test split | `mlfs.model_selection` | `train-test-split` | Planned |

Notes:

- k-means needs a `BaseClusterer` in `mlfs.base`; the coordinator adds it
  before that session starts.
- Iterative solvers use the shared `gradient_descent` and
  `check_gradient_descent_params` in `mlfs.utils._optimize` (extracted from
  `LinearRegression` and `Ridge`); each estimator supplies only its gradient.
- Sessions sharing a subpackage (`linear_model`) only conflict on its
  `__init__.py` exports; the coordinator resolves those when merging.

## Phase 3: optimizers and extensions

Shared infrastructure owned by the coordinator, then small changes in each
estimator.

| Item | Module | Status |
|---|---|---|
| `log_loss` metric (the loss value that second-order and quasi-Newton solvers need) | `mlfs.metrics` | Planned |
| Newton's method (IRLS) and/or L-BFGS as shared solvers, offered through a `solver` hyperparameter | `mlfs.utils._optimize` | Planned |
| Mini-batch stochastic gradient descent with `random_state` | `mlfs.utils._optimize` | Planned |
| Standardization transformer (`StandardScaler`) | `mlfs.preprocessing` | Planned |
| One-vs-rest wrapper for any binary classifier | `mlfs.multiclass` | Planned |

Why later: plain gradient descent is 25 to 250 times slower than
scikit-learn's L-BFGS in `examples/logistic_regression.py`, but a faster
solver serves every iterative model, so it is built once, after the Phase 2
algorithms give the library more breadth.
