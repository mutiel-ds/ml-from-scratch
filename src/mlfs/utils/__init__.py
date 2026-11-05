"""Utilities shared by every estimator: input validation and contract checks."""

from mlfs.utils.validation import (
    check_array,
    check_n_features,
    check_random_state,
    check_X_y,
)

__all__ = ["check_X_y", "check_array", "check_n_features", "check_random_state"]
