"""Athlete injury risk detection."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the version declared in pyproject.toml, which
    # release-please bumps. Everything else reads it from here.
    __version__ = version("injury-risk")
except PackageNotFoundError:  # not installed (e.g. running from a raw checkout)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
