"""Shared type aliases used across the package."""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
"""A NumPy array of 64-bit floats: the internal representation of features."""

RandomState: TypeAlias = int | np.random.Generator | None
"""Accepted values for an estimator's ``random_state`` hyperparameter."""

__all__ = ["ArrayLike", "FloatArray", "NDArray", "RandomState"]
