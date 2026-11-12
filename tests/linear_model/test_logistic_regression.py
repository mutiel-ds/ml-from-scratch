"""Tests for mlfs.linear_model.LogisticRegression."""

import pytest

from mlfs.linear_model import LogisticRegression

# --- Errors and edge cases --------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"alpha": -1.0}, "alpha must be finite and >= 0"),
        ({"alpha": float("nan")}, "alpha must be finite and >= 0"),
        ({"alpha": float("inf")}, "alpha must be finite and >= 0"),
        ({"learning_rate": 0.0}, "learning_rate must be > 0"),
        ({"max_iter": 0}, "max_iter must be >= 1"),
        ({"tol": -1.0}, "tol must be >= 0"),
    ],
    ids=["alpha-negative", "alpha-nan", "alpha-inf", "lr", "max-iter", "tol"],
)
def test_fit_with_invalid_hyperparameter_raises_value_error(
    params: dict[str, object], message: str
) -> None:
    model = LogisticRegression(**params)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        model.fit([[1.0], [2.0]], [0, 1])


def test_init_stores_hyperparameters_unchanged() -> None:
    model = LogisticRegression(
        alpha=0.5, fit_intercept=False, learning_rate=0.2, max_iter=7, tol=1e-3
    )

    assert model.get_params() == {
        "alpha": 0.5,
        "fit_intercept": False,
        "learning_rate": 0.2,
        "max_iter": 7,
        "tol": 1e-3,
    }
