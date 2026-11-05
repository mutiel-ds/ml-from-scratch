"""Custom exceptions raised by mlfs estimators."""

__all__ = ["NotFittedError"]


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before ``fit`` has been called.

    It inherits from both ``ValueError`` and ``AttributeError`` so that callers
    catching either of the usual exceptions still handle it.
    """
