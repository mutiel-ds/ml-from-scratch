"""Shared test configuration."""

import os

from hypothesis import settings

# Local runs explore new random examples each time; CI is reproducible.
settings.register_profile("default", deadline=None)
settings.register_profile("ci", deadline=None, derandomize=True, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
