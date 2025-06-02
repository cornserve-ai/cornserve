"""Utilities for handling optional package imports."""

from __future__ import annotations

from typing import Any


class PlaceholderModule:
    """A placeholder class that replaces optional packages when they are not installed."""

    def __init__(self, name: str, optional_dependency_name: str) -> None:
        """Instantiate a placeholder module with the package's name."""
        self.__name = name
        self.__optional_dependency_name = optional_dependency_name

    def __getattr__(self, name: str) -> Any:
        """Raise an error when any attribute is accessed."""
        if name == "__name":
            return self.__name

        if name == "__optional_dependency_name":
            return self.__optional_dependency_name

        raise RuntimeError(
            f"Optional package '{self.__name}' is not installed. Please install "
            f"optional dependencies with `pip install cornserve[{self.__optional_dependency_name}]`."
        )
