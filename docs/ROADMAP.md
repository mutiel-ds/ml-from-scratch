# Roadmap

Maintained by the coordinator session. Each algorithm is built in its own
session and branch (`feat/<algorithm>`), following the rules in `CLAUDE.md`.

## Phase 0: foundation

| Item | Status |
|---|---|
| Installable package (`src/mlfs`, hatchling, `py.typed`, Python 3.11+) | Done |
| Estimator base classes and `NotFittedError` (`mlfs.base`) | Done |
| Input validation helpers (`mlfs.utils.validation`) | Done |
| Shared contract checks (`mlfs.utils.estimator_checks`) | Done |
| Metrics: `mean_squared_error`, `r2_score`, `accuracy_score` | Done |
| Tooling: ruff, mypy (strict), pytest + coverage, pre-commit, GitHub Actions | Done |
| Runtime and dev dependencies installed with uv | Waiting for owner |

## Phase 1: reference implementation

| Algorithm | Module | Branch | Status |
|---|---|---|---|
| Linear regression (normal equation + gradient descent) | `mlfs.linear_model` | `feat/linear-regression` | Planned |

Once linear regression is merged, `CLAUDE.md` is reviewed and adjusted based on
what the first session learned, before starting Phase 2.

## Phase 2: parallel algorithm sessions

| Algorithm | Module | Branch | Status |
|---|---|---|---|
| Ridge regression | `mlfs.linear_model` | `feat/ridge` | Planned |
| Logistic regression | `mlfs.linear_model` | `feat/logistic-regression` | Planned |
| k-nearest neighbors | `mlfs.neighbors` | `feat/knn` | Planned |
| k-means (k-means++ init) | `mlfs.cluster` | `feat/kmeans` | Planned |
| Decision tree (CART) | `mlfs.tree` | `feat/decision-tree` | Planned |
| Gaussian Naive Bayes | `mlfs.naive_bayes` | `feat/naive-bayes` | Planned |
| PCA | `mlfs.decomposition` | `feat/pca` | Planned |
| Train/test split | `mlfs.model_selection` | `feat/train-test-split` | Planned |

Notes:

- k-means needs a `BaseClusterer` in `mlfs.base`; the coordinator adds it
  before that session starts.
- Sessions sharing a subpackage (`linear_model`) only conflict on its
  `__init__.py` exports; the coordinator resolves those when merging.
