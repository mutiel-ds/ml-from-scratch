"""Tests for mlfs.linear_model.LogisticRegression."""

import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from mlfs._typing import FloatArray
from mlfs.exceptions import ConvergenceWarning, NotFittedError
from mlfs.linear_model import LogisticRegression
from mlfs.linear_model._logistic_regression import _encode_labels, _gradient, _sigmoid
from mlfs.utils.estimator_checks import check_estimator

# --- Contract ---------------------------------------------------------------

# check_estimator's data is linearly separable, and pytest turns a
# ConvergenceWarning into an error. So every combination keeps alpha > 0 and
# gets enough iterations: with the default settings this data needs about
# 1400 steps, more than the default max_iter=1000.


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(max_iter=10_000),
        LogisticRegression(fit_intercept=False, max_iter=10_000),
        LogisticRegression(alpha=10.0),
        LogisticRegression(learning_rate=0.5),
    ],
    ids=["more-iterations", "no-intercept", "strong-penalty", "large-learning-rate"],
)
def test_logistic_regression_satisfies_estimator_contract(
    estimator: LogisticRegression,
) -> None:
    check_estimator(estimator)


# --- Sigmoid ----------------------------------------------------------------


def test_sigmoid_at_zero_is_one_half() -> None:
    z = np.array([0.0])

    sigma = _sigmoid(z)

    # e^0 = 1, so sigma(0) = 1 / (1 + 1) = 0.5.
    np.testing.assert_allclose(sigma, [0.5])


def test_sigmoid_at_log_three_is_three_quarters() -> None:
    z = np.array([np.log(3.0)])

    sigma = _sigmoid(z)

    # e^{-ln 3} = 1/3, so sigma = 1 / (1 + 1/3) = 3/4.
    np.testing.assert_allclose(sigma, [0.75])


@pytest.mark.parametrize("z", [0.5, 1.0, 4.0])
def test_sigmoid_is_symmetric(z: float) -> None:
    sigma_positive = _sigmoid(np.array([z]))
    sigma_negative = _sigmoid(np.array([-z]))

    # Property: sigma(-z) = 1 - sigma(z).
    np.testing.assert_allclose(sigma_negative, 1 - sigma_positive)


def test_sigmoid_at_extreme_inputs_saturates_without_warnings() -> None:
    z = np.array([-1000.0, 1000.0])

    # pytest is configured with filterwarnings = ["error"], so any overflow
    # RuntimeWarning raised here would already fail the test.
    sigma = _sigmoid(z)

    # atol is needed because assert_allclose's default is relative: it would
    # reject any tiny result when comparing with an exact 0.
    np.testing.assert_allclose(sigma, [0.0, 1.0], atol=1e-300)


# --- Gradient ---------------------------------------------------------------


def test_gradient_at_zero_params_without_penalty() -> None:
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, 0.0])
    params = np.array([0.0, 0.0])

    grad = _gradient(params, X, y, alpha=0.0, fit_intercept=True)

    # p = 0.5 everywhere, so p - y = [-0.5, 0.5];
    # d/dw = (1*(-0.5) + (-1)*0.5) / 2 = -0.5;  d/db = (-0.5 + 0.5) / 2 = 0.
    np.testing.assert_allclose(grad, [-0.5, 0.0], atol=1e-12)


def test_gradient_penalizes_the_weight_but_not_the_intercept() -> None:
    X = np.array([[2.0]])
    y = np.array([1.0])
    params = np.array([1.0, -2.0])

    grad = _gradient(params, X, y, alpha=3.0, fit_intercept=True)

    # p = 0.5, so p - y = [-0.5];
    # d/dw = 2*(-0.5) + 3*1= -1 + 3 = 2;  d/db = -0.5 = -0.5.
    np.testing.assert_allclose(grad, [2.0, -0.5], atol=1e-12)


def test_gradient_without_intercept_returns_only_weight_gradient() -> None:
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, 0.0])
    params = np.array([0.0])

    grad = _gradient(params, X, y, alpha=0.0, fit_intercept=False)

    # Same data as the first test: p - y = [-0.5, 0.5] and
    # d/dw = (1*(-0.5) + (-1)*0.5) / 2 = -0.5. There is no intercept entry,
    # so the result has shape (1,), not (2,).
    np.testing.assert_allclose(grad, [-0.5])


# --- Label encoding ---------------------------------------------------------


@pytest.mark.parametrize(
    ("y", "expected_classes", "expected_encoded"),
    [
        (["no", "yes", "yes", "no"], ["no", "yes"], [0, 1, 1, 0]),
        (["b", "a", "b", "b"], ["a", "b"], [1, 0, 1, 1]),
        ([-1, 1, 1, -1], [-1, 1], [0, 1, 1, 0]),
        ([5, 2, 2, 5], [2, 5], [1, 0, 0, 1]),
        ([False, True, True, False], [False, True], [0, 1, 1, 0]),
    ],
    ids=["strings", "unsorted-strings", "minus-one-one", "unsorted-ints", "bools"],
)
def test_encode_labels_maps_second_sorted_label_to_one(
    y: list[object], expected_classes: list[object], expected_encoded: list[int]
) -> None:
    classes, y_encoded = _encode_labels(np.asarray(y))

    # classes are the sorted unique labels; the larger label is the positive
    # class (1) and the smaller one is the negative class (0).
    np.testing.assert_array_equal(classes, expected_classes)
    np.testing.assert_array_equal(y_encoded, expected_encoded)
    assert y_encoded.dtype == np.float64


@pytest.mark.parametrize(
    ("y", "message"),
    [
        ([1, 1, 1], r"got 1: \[1\]"),
        ([0, 1, 2], r"got 3: \[0, 1, 2\]"),
        (["a", "b", "c", "a"], "got 3"),
    ],
    ids=["one-class", "three-classes", "three-string-classes"],
)
def test_encode_labels_without_exactly_two_classes_raises_value_error(
    y: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _encode_labels(np.asarray(y))


@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
def test_encode_labels_with_non_finite_float_label_raises_value_error(
    bad: float,
) -> None:
    y = np.array([0.0, 1.0, bad])

    with pytest.raises(ValueError, match="NaN or infinite"):
        _encode_labels(y)


# --- Known answers ----------------------------------------------------------


def test_fit_with_identical_samples_and_balanced_labels_keeps_zero_parameters() -> None:
    X = [[1.0], [1.0]]
    y = [0, 1]

    model = LogisticRegression(alpha=0.0).fit(X, y)

    # At w = b = 0 every p = 0.5, so p - y = [0.5, -0.5] and both gradients
    # are 0: the start is already optimal and no step is taken.
    np.testing.assert_allclose(model.coef_, [0.0], atol=1e-12)
    assert model.intercept_ == pytest.approx(0.0, abs=1e-12)
    assert model.n_iter_ == 0


def test_fit_one_step_moves_parameters_along_negative_gradient() -> None:
    X = [[1.0], [-1.0]]
    y = [1, 0]
    model = LogisticRegression(alpha=0.0, learning_rate=0.1, max_iter=1)

    with pytest.warns(ConvergenceWarning, match="did not converge"):
        model.fit(X, y)

    # Gradient at 0 is [-0.5, 0] (see the gradient tests), so one step of
    # size 0.1 gives [0 - 0.1*(-0.5), 0 - 0.1*0] = [0.05, 0].
    np.testing.assert_allclose(model.coef_, [0.05])
    assert model.intercept_ == pytest.approx(0.0, abs=1e-12)
    assert model.n_iter_ == 1


def test_fit_with_string_labels_stores_sorted_classes() -> None:
    model = LogisticRegression().fit([[1.0], [-1.0], [2.0]], ["yes", "no", "yes"])

    np.testing.assert_array_equal(model.classes_, ["no", "yes"])


def test_fit_without_intercept_sets_intercept_zero_and_one_weight_per_feature() -> None:
    X = [[1.0, 2.0], [-1.0, 0.5], [2.0, -1.0]]
    y = [1, 0, 1]

    model = LogisticRegression(fit_intercept=False).fit(X, y)

    assert model.coef_.shape == (2,)
    assert model.intercept_ == 0.0


def test_fit_sets_number_of_features() -> None:
    model = LogisticRegression().fit([[1.0, 0.0], [-1.0, 0.5], [0.5, -1.0]], [1, 0, 1])

    assert model.n_features_in_ == 2


# --- Prediction -------------------------------------------------------------


def _model_with_parameters(
    coef: float, intercept: float, classes: list[object]
) -> LogisticRegression:
    """Build a fitted model by hand, so prediction is tested without ``fit``."""
    model = LogisticRegression()
    model.coef_ = np.array([coef])
    model.intercept_ = intercept
    model.classes_ = np.array(classes)
    model.n_features_in_ = 1
    return model


def test_decision_function_returns_linear_score() -> None:
    model = _model_with_parameters(2.0, -1.0, ["a", "b"])

    z = model.decision_function([[0.0], [0.5], [1.0]])

    # z = 2x - 1.
    np.testing.assert_allclose(z, [-1.0, 0.0, 1.0], atol=1e-12)


def test_predict_proba_returns_sigmoid_of_score_for_both_classes() -> None:
    model = _model_with_parameters(2.0, -1.0, ["a", "b"])

    proba = model.predict_proba([[0.0], [0.5], [1.0]])

    # Scores are [-1, 0, 1]; sigmoid(1) = 1 / (1 + e^-1) = 0.7310585786...,
    # sigmoid(0) = 0.5 and sigmoid(-1) = 1 - sigmoid(1) = 0.2689414214...
    expected = [[0.7310585786, 0.2689414214], [0.5, 0.5], [0.2689414214, 0.7310585786]]
    np.testing.assert_allclose(proba, expected, rtol=1e-9)


def test_predict_proba_at_extreme_scores_saturates_without_warnings() -> None:
    model = _model_with_parameters(1000.0, 0.0, ["a", "b"])

    proba = model.predict_proba([[-1.0], [1.0]])

    np.testing.assert_allclose(proba, [[1.0, 0.0], [0.0, 1.0]], atol=1e-300)


def test_predict_returns_original_labels_and_breaks_ties_to_first_class() -> None:
    model = _model_with_parameters(2.0, -1.0, ["no", "yes"])

    y_pred = model.predict([[0.0], [0.5], [1.0]])

    # Scores are [-1, 0, 1]: negative -> "no", exactly 0 -> classes_[0] = "no",
    # positive -> "yes".
    np.testing.assert_array_equal(y_pred, ["no", "no", "yes"])


@pytest.mark.parametrize("method", ["decision_function", "predict_proba", "predict"])
def test_prediction_before_fit_raises_not_fitted_error(method: str) -> None:
    with pytest.raises(NotFittedError, match="not fitted"):
        getattr(LogisticRegression(), method)([[1.0]])


@pytest.mark.parametrize("method", ["decision_function", "predict_proba", "predict"])
def test_prediction_with_wrong_number_of_features_raises_value_error(
    method: str,
) -> None:
    model = _model_with_parameters(1.0, 0.0, [0, 1])

    with pytest.raises(ValueError, match="X has 2 features"):
        getattr(model, method)([[1.0, 2.0]])


# --- Reference: scikit-learn ------------------------------------------------


def _noisy_binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a seeded, non-separable dataset, so the unpenalized fit exists."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 3))
    probability = 1 / (1 + np.exp(-(X @ [1.5, -2.0, 0.5] + 0.5)))
    y = (rng.random(100) < probability).astype(int)
    return X, y


# sklearn's C is 1 / alpha (alpha = 0 is C = inf: no penalty).
@pytest.mark.parametrize(
    ("alpha", "C"),
    [(1.0, 1.0), (5.0, 0.2), (0.0, np.inf)],
    ids=["alpha-1", "alpha-5", "no-penalty"],
)
@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no-int"])
def test_fit_matches_sklearn_on_noisy_data(
    alpha: float, C: float, fit_intercept: bool
) -> None:
    X, y = _noisy_binary_data()
    ours = LogisticRegression(
        alpha=alpha,
        fit_intercept=fit_intercept,
        learning_rate=0.1,
        max_iter=200_000,
        tol=1e-12,
    ).fit(X, y)

    reference = SklearnLogisticRegression(
        C=C, fit_intercept=fit_intercept, tol=1e-12, max_iter=10_000
    ).fit(X, y)

    # Both solvers are iterative and stop on a gradient tolerance, so the
    # parameters agree only to the solvers' accuracy (observed: below 1e-7).
    np.testing.assert_allclose(ours.coef_, reference.coef_[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        ours.intercept_, reference.intercept_[0], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        ours.predict_proba(X), reference.predict_proba(X), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(ours.predict(X), reference.predict(X))


# --- Properties -------------------------------------------------------------

# A fixed, well-conditioned X: Hypothesis varies the labels and the
# hyperparameters, not the design matrix, so failures reflect the properties
# and not ill-conditioning.
_X_FIXED = np.random.default_rng(1).normal(size=(30, 3))
# The first two labels are fixed so that both classes are always present.
_labels = hnp.arrays(np.int64, 28, elements=st.integers(0, 1)).map(
    lambda rest: np.concatenate([[0, 1], rest])
)
_alphas = st.floats(min_value=1.0, max_value=50.0)

# Fitting is iterative, so these properties run fewer examples. max_iter is
# generous because pytest turns a ConvergenceWarning into an error.
_FEW_EXAMPLES = settings(max_examples=25)
_FIT_KWARGS = {"max_iter": 20_000, "tol": 1e-10}


@_FEW_EXAMPLES
@given(y=_labels, alpha=_alphas)
def test_predict_proba_gives_valid_probabilities_consistent_with_predict(
    y: FloatArray, alpha: float
) -> None:
    model = LogisticRegression(alpha=alpha, **_FIT_KWARGS).fit(_X_FIXED, y)  # type: ignore[arg-type]

    proba = model.predict_proba(_X_FIXED)

    assert np.all((proba >= 0.0) & (proba <= 1.0))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    # argmax picks the first column on a tie, matching the documented rule.
    np.testing.assert_array_equal(
        model.predict(_X_FIXED), model.classes_[proba.argmax(axis=1)]
    )


@_FEW_EXAMPLES
@given(y=_labels, alpha=_alphas)
def test_fit_with_swapped_labels_negates_the_parameters(
    y: FloatArray, alpha: float
) -> None:
    model = LogisticRegression(alpha=alpha, **_FIT_KWARGS)  # type: ignore[arg-type]

    original = model.fit(_X_FIXED, y).coef_.copy(), model.intercept_
    swapped = model.fit(_X_FIXED, 1 - y).coef_.copy(), model.intercept_

    # sigmoid(-z) = 1 - sigmoid(z): swapping the classes mirrors the whole
    # optimization, so (w, b) -> (-w, -b). Gradient descent follows the exact
    # mirrored path, so only round-off separates the two results.
    np.testing.assert_allclose(swapped[0], -original[0], atol=1e-9)
    assert swapped[1] == pytest.approx(-original[1], abs=1e-9)


@_FEW_EXAMPLES
@given(y=_labels, alpha=_alphas, order=st.permutations(range(30)))
def test_fit_is_invariant_to_the_order_of_the_training_samples(
    y: FloatArray, alpha: float, order: list[int]
) -> None:
    model = LogisticRegression(alpha=alpha, **_FIT_KWARGS)  # type: ignore[arg-type]

    original = model.fit(_X_FIXED, y).coef_.copy()
    shuffled = model.fit(_X_FIXED[order], y[order]).coef_.copy()

    # The loss is a sum over samples, so row order cannot matter; the sums
    # are only reordered, which changes round-off.
    np.testing.assert_allclose(shuffled, original, atol=1e-9)


@_FEW_EXAMPLES
@given(y=_labels, alpha=_alphas, extra=st.floats(min_value=0.0, max_value=50.0))
def test_fit_with_larger_alpha_never_increases_coef_norm(
    y: FloatArray, alpha: float, extra: float
) -> None:
    smaller = LogisticRegression(alpha=alpha, **_FIT_KWARGS).fit(_X_FIXED, y)  # type: ignore[arg-type]
    larger = LogisticRegression(alpha=alpha + extra, **_FIT_KWARGS).fit(_X_FIXED, y)  # type: ignore[arg-type]

    # The penalized minimum shrinks monotonically in alpha. The slack covers
    # the stopping tolerance: the gradient norm is below 1e-10 and the
    # curvature is at least alpha / n >= 1/30, so the error is about 3e-9.
    assert np.linalg.norm(larger.coef_) <= np.linalg.norm(smaller.coef_) + 1e-7


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


def test_fit_with_three_classes_raises_value_error() -> None:
    with pytest.raises(ValueError, match="exactly 2 distinct labels"):
        LogisticRegression().fit([[1.0], [2.0], [3.0]], [0, 1, 2])


def test_fit_with_a_single_sample_raises_value_error() -> None:
    with pytest.raises(ValueError, match="got 1"):
        LogisticRegression().fit([[1.0]], [0])


def test_fit_with_a_single_feature_works() -> None:
    model = LogisticRegression().fit([[-2.0], [-1.0], [1.0], [2.0]], [0, 0, 1, 1])

    assert model.coef_.shape == (1,)
    assert model.coef_[0] > 0  # larger x makes the positive class more likely


def test_fit_on_separable_data_without_penalty_warns_and_grows_the_weight() -> None:
    model = LogisticRegression(alpha=0.0, max_iter=100)

    with pytest.warns(ConvergenceWarning, match="did not converge"):
        model.fit([[1.0], [-1.0]], [1, 0])

    # No finite weight minimizes the loss: it keeps increasing, so the run
    # ends at max_iter. Each step adds lr * (1 - sigmoid(w)) / 2 > 0.
    assert model.n_iter_ == 100
    assert model.coef_[0] > 1.0


def test_fit_on_separable_data_with_penalty_converges() -> None:
    model = LogisticRegression(alpha=1.0).fit([[1.0], [-1.0]], [1, 0])

    # The penalty gives the loss a finite minimizer, so no warning is raised
    # (pytest would turn one into an error) and the weight stays moderate.
    assert 0.0 < model.coef_[0] < 5.0
    assert model.n_iter_ < 1000


# mutmut wraps every function in a trampoline, which adds a stack frame and so
# shifts where stacklevel points. The test is only meaningful without it.
@pytest.mark.skipif(
    "mutmut" in sys.modules, reason="mutmut's trampoline changes the call stack"
)
def test_convergence_warning_points_at_the_callers_fit_line() -> None:
    model = LogisticRegression(max_iter=1)

    with pytest.warns(ConvergenceWarning) as record:
        model.fit([[1.0], [-1.0]], [1, 0])

    # stacklevel=3 blames the line that called fit (this file), not the
    # library's internals.
    assert record[0].filename == __file__


def test_refit_replaces_everything_learned_from_the_previous_data() -> None:
    model = LogisticRegression()
    model.fit([[1.0, 0.0], [-1.0, 0.5], [0.5, -1.0]], [1, 0, 1])

    model.fit([[-2.0], [-1.0], [1.0], [2.0]], ["no", "no", "yes", "yes"])

    assert model.n_features_in_ == 1
    assert model.coef_.shape == (1,)
    np.testing.assert_array_equal(model.classes_, ["no", "yes"])
