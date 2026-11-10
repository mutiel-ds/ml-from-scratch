"""Linear regression three ways: closed form, library gradient descent, by hand.

Run with ``uv run python examples/linear_regression.py``.

1. ``LinearRegression(solver="lstsq")``: closed-form least squares.
2. ``LinearRegression(solver="gradient_descent")``: the library solver, which
   centers the data and learns only the weights.
3. A standalone loop below that learns the weights *and* the bias with their
   own gradients. It is the textbook version, kept here for learning purposes.

All three should recover the same line. A final section then times them,
and compares them with scikit-learn's ``LinearRegression`` when it is installed.
"""

import time
from collections.abc import Callable

import numpy as np

from mlfs._typing import FloatArray
from mlfs.linear_model import LinearRegression


def gradient_descent_with_bias(
    X: FloatArray,
    y: FloatArray,
    learning_rate: float = 0.1,
    max_iter: int = 10_000,
    tol: float = 1e-12,
) -> tuple[FloatArray, float]:
    """Minimize ``(1 / 2n) * ||Xw + b - y||^2`` over both ``w`` and ``b``.

    With residuals ``r = Xw + b - y`` the gradients are

        dL/dw = X^T r / n        dL/db = mean(r)

    The bias gradient is just the average residual: ``b`` behaves like the
    weight of a feature that is always 1.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(max_iter):
        residuals = X @ w + b - y
        grad_w = X.T @ residuals / n_samples
        grad_b = residuals.mean()
        if np.sqrt(grad_w @ grad_w + grad_b**2) < tol:
            break
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

    return w, float(b)


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
    """Time every method on one random problem and print a small table."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n_samples, n_features))
    y = X @ rng.normal(size=n_features) + 1.0 + rng.normal(size=n_samples)

    gd_kwargs = {"learning_rate": 0.1, "max_iter": 10_000, "tol": 1e-8}
    methods: dict[str, Callable[[], object]] = {
        "mlfs lstsq": lambda: LinearRegression().fit(X, y),
        "mlfs gradient descent": lambda: LinearRegression(
            solver="gradient_descent", **gd_kwargs
        ).fit(X, y),
        "by-hand GD with bias": lambda: gradient_descent_with_bias(
            X, y, tol=gd_kwargs["tol"]
        ),
    }
    try:
        # scikit-learn is only a dev dependency, so this part is optional.
        from sklearn.linear_model import LinearRegression as SklearnLinearRegression

        methods["sklearn LinearRegression"] = lambda: SklearnLinearRegression().fit(
            X, y
        )
    except ImportError:
        print("(scikit-learn not installed: skipping the reference timing)")

    print(f"\nn_samples={n_samples}, n_features={n_features} (best of 5, ms)")
    for name, func in methods.items():
        print(f"  {name:<26}{best_time(func):>10.3f}")


def main() -> None:
    """Fit the same noisy data with all three methods and compare them."""
    rng = np.random.default_rng(0)
    true_w = np.array([1.5, -2.0, 0.5])
    true_b = 3.0
    X = rng.normal(size=(100, 3))
    y = X @ true_w + true_b + rng.normal(scale=0.5, size=100)

    closed_form = LinearRegression(solver="lstsq").fit(X, y)
    library_gd = LinearRegression(
        solver="gradient_descent", learning_rate=0.1, max_iter=10_000, tol=1e-12
    ).fit(X, y)
    w_by_hand, b_by_hand = gradient_descent_with_bias(X, y)

    rows = {
        "true": (true_w, true_b),
        "lstsq": (closed_form.coef_, closed_form.intercept_),
        "library GD": (library_gd.coef_, library_gd.intercept_),
        "GD with bias": (w_by_hand, b_by_hand),
    }
    for name, (w, b) in rows.items():
        print(f"{name:<13} w = {np.round(w, 4)}  b = {b:.4f}")

    # Tolerance: both gradient descents stop at a gradient norm below 1e-12.
    agree = np.allclose(library_gd.coef_, closed_form.coef_, atol=1e-8) and np.allclose(
        w_by_hand, closed_form.coef_, atol=1e-8
    )
    print(f"\nAll three agree on the weights: {agree}")
    print(f"R^2 of the closed-form fit: {closed_form.score(X, y):.4f}")

    for n_samples, n_features in [(100, 3), (10_000, 20), (100_000, 50)]:
        benchmark(n_samples, n_features)


if __name__ == "__main__":
    main()
