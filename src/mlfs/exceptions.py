"""Custom exceptions raised by mlfs estimators."""

__all__ = ["ConvergenceWarning", "NotFittedError"]


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before ``fit`` has been called.

    It inherits from both ``ValueError`` and ``AttributeError`` so that callers
    catching either of the usual exceptions still handle it.
    """


class ConvergenceWarning(UserWarning):
    """Warn that an iterative algorithm stopped before reaching its tolerance.

    Raised through ``warnings.warn`` when ``max_iter`` iterations were not
    enough for the stopping criterion to be met. The fitted model is still
    usable, but it may be far from the optimum.
    """
