"""mlfs: classic machine learning algorithms implemented from scratch in NumPy.

The name stands for "Machine Learning From Scratch". Estimators follow a
scikit-learn-style API: hyperparameters go in ``__init__``, learning happens in
``fit`` (which returns ``self``), and fitted attributes end with an underscore.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ml-from-scratch")
except PackageNotFoundError:  # pragma: no cover - only when run from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
