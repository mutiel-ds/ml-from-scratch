"""Ridge regression: shrinkage, stability with collinear features, solvers.

Run with ``uv run python examples/ridge.py``.

1. How the weights shrink towards zero as ``alpha`` grows.
2. Why ridge exists: with two almost identical features, least squares gives
   huge, unstable weights of opposite sign, while ridge stays calm.
3. All three solvers (``"solve"``, ``"lstsq"``, ``"gradient_descent"``) reach
   the same solution, and agree with scikit-learn's ``Ridge`` when installed.
4. A benchmark times every solver, and scikit-learn's ``Ridge`` when it is
   installed, on problems of growing size.
"""

import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from mlfs.linear_model import LinearRegression, Ridge


def make_collinear_data(
    n_samples: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two nearly identical features; only their sum matters for ``y``."""
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n_samples)
    x2 = x1 + rng.normal(scale=1e-3, size=n_samples)  # almost a copy of x1
    X = np.column_stack([x1, x2])
    y = x1 + x2 + rng.normal(scale=0.5, size=n_samples)  # true weights: (1, 1)
    return X, y


def show_shrinkage() -> None:
    """Print how the weight norm falls as alpha grows."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 3.0 + rng.normal(scale=0.5, size=100)

    print("Shrinkage (true weights: [1.5, -2.0, 0.5])")
    for alpha in [0.0, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha).fit(X, y)
        norm = np.linalg.norm(model.coef_)
        print(f"  alpha={alpha:>7}: w = {np.round(model.coef_, 3)}  ||w|| = {norm:.3f}")


def show_collinearity() -> None:
    """Compare least squares and ridge on almost collinear features."""
    X, y = make_collinear_data()

    ols = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=1.0).fit(X, y)

    print("\nNearly collinear features (true weights: [1, 1])")
    print(f"  least squares: w = {np.round(ols.coef_, 2)}")
    print(f"  ridge alpha=1: w = {np.round(ridge.coef_, 2)}")


def compare_solvers() -> None:
    """Fit the same problem with every solver and compare the weights."""
    X, y = make_collinear_data()
    solvers = {
        "solve": Ridge(alpha=1.0, solver="solve"),
        "lstsq": Ridge(alpha=1.0, solver="lstsq"),
        "gradient_descent": Ridge(
            alpha=1.0,
            solver="gradient_descent",
            learning_rate=0.1,
            max_iter=100_000,
            tol=1e-12,
        ),
    }

    print("\nSolvers (alpha=1)")
    for name, model in solvers.items():
        model.fit(X, y)
        print(
            f"  {name:<17} w = {np.round(model.coef_, 6)}  b = {model.intercept_:.6f}"
        )

    try:
        # scikit-learn is only a dev dependency, so this part is optional.
        from sklearn.linear_model import Ridge as SklearnRidge

        reference = SklearnRidge(alpha=1.0).fit(X, y)
        print(f"  {'sklearn Ridge':<17} w = {np.round(reference.coef_, 6)}")
    except ImportError:
        print("  (scikit-learn not installed: skipping the reference)")


def best_time(func: Callable[[], object], repeats: int = 5) -> float:
    """Return the fastest of ``repeats`` runs of ``func``, in milliseconds.

    The minimum is the least noisy estimate: other processes can only make a
    run slower, never faster.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times) * 1e3


def benchmark(n_samples: int, n_features: int) -> None:
    """Time every solver on one random problem and print a small table."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n_samples, n_features))
    y = X @ rng.normal(size=n_features) + 1.0 + rng.normal(size=n_samples)

    methods: dict[str, Callable[[], object]] = {
        "mlfs solve": lambda: Ridge(solver="solve").fit(X, y),
        "mlfs lstsq": lambda: Ridge(solver="lstsq").fit(X, y),
        "mlfs gradient descent": lambda: Ridge(
            solver="gradient_descent", learning_rate=0.1, max_iter=10_000, tol=1e-8
        ).fit(X, y),
    }
    try:
        # scikit-learn is only a dev dependency, so this part is optional.
        from sklearn.linear_model import Ridge as SklearnRidge

        methods["sklearn Ridge"] = lambda: SklearnRidge().fit(X, y)
    except ImportError:
        print("(scikit-learn not installed: skipping the reference timing)")

    print(f"\nn_samples={n_samples}, n_features={n_features} (best of 5, ms)")
    for name, func in methods.items():
        print(f"  {name:<26}{best_time(func):>10.3f}")


def main() -> None:
    """Run the demonstrations and the benchmark."""
    show_shrinkage()
    show_collinearity()
    compare_solvers()

    for n_samples, n_features in [(100, 3), (10_000, 20), (100_000, 50)]:
        benchmark(n_samples, n_features)


if __name__ == "__main__":
    main()
