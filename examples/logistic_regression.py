"""Logistic regression: probabilities, regularization, feature scaling.

Run with ``uv run python examples/logistic_regression.py``.

1. Classify two noisy clusters, with string labels, and read probabilities.
2. How the penalty ``alpha`` shrinks the weights and softens the probabilities.
3. Why features must be scaled: gradient descent needs far more iterations
   (or fails to converge) on features of very different magnitudes.
4. Linearly separable data: without a penalty the weights grow without bound.
5. Agreement with scikit-learn's ``LogisticRegression`` when it is installed.
6. A benchmark times ``fit`` here, and scikit-learn's when it is installed, on
   problems of growing size.
"""

import time
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from mlfs.exceptions import ConvergenceWarning
from mlfs.linear_model import LogisticRegression


def make_clusters(
    n_samples: int = 200,
) -> tuple[NDArray[np.float64], NDArray[np.str_]]:
    """Two overlapping Gaussian clusters labelled "cat" and "dog"."""
    rng = np.random.default_rng(0)
    is_dog = rng.random(n_samples) < 0.5
    centers = np.where(is_dog[:, None], [1.0, 1.0], [-1.0, -1.0])
    X = centers + rng.normal(size=(n_samples, 2))
    y = np.where(is_dog, "dog", "cat")
    return X, y


def standardize(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Give every feature mean 0 and standard deviation 1."""
    result: NDArray[np.float64] = (X - X.mean(axis=0)) / X.std(axis=0)
    return result


def show_classification() -> None:
    """Fit on string labels and print accuracy and a few probabilities."""
    X, y = make_clusters()
    model = LogisticRegression(max_iter=10_000).fit(X, y)

    print("Classification (labels are strings)")
    print(f"  classes_ = {model.classes_}  (the second one is the positive class)")
    print(f"  accuracy = {model.score(X, y):.3f}")
    for point in ([-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]):
        proba = model.predict_proba([point])[0]
        label = model.predict([point])[0]
        print(
            f"  x = {point}: P(cat) = {proba[0]:.3f}, P(dog) = {proba[1]:.3f}"
            f" -> {label}"
        )


def show_regularization() -> None:
    """Print how the weight norm falls as alpha grows."""
    X, y = make_clusters()

    print("\nRegularization (the intercept is never penalized)")
    for alpha in [0.0, 1.0, 10.0, 100.0]:
        model = LogisticRegression(alpha=alpha, max_iter=20_000).fit(X, y)
        confident = np.mean(np.max(model.predict_proba(X), axis=1) > 0.9)
        print(
            f"  alpha={alpha:>6}: ||w|| = {np.linalg.norm(model.coef_):.3f}, "
            f"{confident:.0%} of the predictions have probability > 0.9"
        )


def fit_and_count(X: NDArray[np.float64], y: NDArray[np.str_]) -> str:
    """Fit with a fixed budget and describe how it went."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LogisticRegression(max_iter=20_000).fit(X, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        return f"did NOT converge in {model.n_iter_} iterations"
    return f"converged in {model.n_iter_} iterations"


def show_scaling() -> None:
    """Compare convergence with and without feature scaling."""
    X, y = make_clusters()
    X_unscaled = X * np.array([1.0, 100.0])  # second feature in other units

    print("\nFeature scaling (second feature multiplied by 100)")
    print(f"  raw features:    {fit_and_count(X_unscaled, y)}")
    print(f"  standardized:    {fit_and_count(standardize(X_unscaled), y)}")


def show_separable_data() -> None:
    """Without a penalty the loss has no minimum on separable data."""
    X = np.array([[1.0], [2.0], [-1.0], [-2.0]])
    y = np.array([1, 1, 0, 0])

    print("\nLinearly separable data")
    for alpha in [0.0, 1.0]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LogisticRegression(alpha=alpha, max_iter=5000).fit(X, y)
        status = "warned (not converged)" if caught else "converged"
        print(
            f"  alpha={alpha}: w = {model.coef_[0]:.3f} after "
            f"{model.n_iter_} iterations, {status}"
        )


def compare_with_sklearn() -> None:
    """Compare with scikit-learn, whose ``C`` is ``1 / alpha``."""
    X, y = make_clusters()
    ours = LogisticRegression(alpha=1.0, max_iter=100_000, tol=1e-10).fit(X, y)

    print("\nAgreement with scikit-learn (alpha=1, i.e. C=1)")
    print(f"  mlfs:    w = {np.round(ours.coef_, 6)}  b = {ours.intercept_:.6f}")
    try:
        # scikit-learn is only a dev dependency, so this part is optional.
        from sklearn.linear_model import LogisticRegression as SklearnLogReg

        reference = SklearnLogReg(C=1.0, tol=1e-10).fit(X, y)
        print(
            f"  sklearn: w = {np.round(reference.coef_[0], 6)}  "
            f"b = {reference.intercept_[0]:.6f}"
        )
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
    """Time ``fit`` on one random, standardized problem and print a table.

    Plain gradient descent needs thousands of cheap steps, while scikit-learn's
    L-BFGS uses curvature information to need only dozens, so expect ours to
    be much slower: the point of this library is clarity, not speed.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n_samples, n_features))
    probability = 1 / (1 + np.exp(-(X @ rng.normal(size=n_features))))
    y = (rng.random(n_samples) < probability).astype(int)

    def fit_ours() -> None:
        # Hitting max_iter is fine for timing; keep the output clean.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            LogisticRegression(alpha=1.0, max_iter=10_000, tol=1e-8).fit(X, y)

    methods: dict[str, Callable[[], object]] = {"mlfs gradient descent": fit_ours}
    try:
        # scikit-learn is only a dev dependency, so this part is optional.
        from sklearn.linear_model import LogisticRegression as SklearnLogReg

        methods["sklearn (lbfgs)"] = lambda: SklearnLogReg(tol=1e-8).fit(X, y)
    except ImportError:
        print("(scikit-learn not installed: skipping the reference timing)")

    print(f"\nn_samples={n_samples}, n_features={n_features} (best of 5, ms)")
    for name, func in methods.items():
        print(f"  {name:<26}{best_time(func):>10.3f}")


def main() -> None:
    """Run all the demonstrations and the benchmark."""
    show_classification()
    show_regularization()
    show_scaling()
    show_separable_data()
    compare_with_sklearn()

    for n_samples, n_features in [(100, 3), (2_000, 10), (20_000, 30)]:
        benchmark(n_samples, n_features)


if __name__ == "__main__":
    main()
