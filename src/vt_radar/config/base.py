"""
Base configuration classes for TI mmWave SDK configurations.
"""
from typing import Any, List


class Cfg(object):
    """Base class for all configuration classes."""

    def parse(self, args: List[str]) -> None:
        """Parse configuration parameters from command line arguments.

        Args:
            args: List of command line arguments
        """
        raise NotImplementedError("Subclasses must implement parse method")

    def describe(self) -> None:
        """Describe the configuration parameters."""
        raise NotImplementedError("Subclasses must implement describe method")
