"""Evaluation metrics for regression and classification."""

from mlfs.metrics._classification import accuracy_score
from mlfs.metrics._regression import mean_squared_error, r2_score

__all__ = ["accuracy_score", "mean_squared_error", "r2_score"]
