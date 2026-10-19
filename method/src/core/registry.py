"""
Unified Strategy Registry System.

This module provides a generic registry pattern for pluggable strategies,
enabling clean Strategy Pattern implementations across the codebase.

Example Usage:
    from src.core.registry import StrategyRegistry
    from .base import BasePairingStrategy

    # Create registry
    pairing_registry = StrategyRegistry("pairing", BasePairingStrategy)

    # Register strategy with decorator
    @pairing_registry.register("FaceForensics++", "FF++", "FF")
    class FaceForensicsPairing(BasePairingStrategy):
        ...

    # Get strategy class
    cls = pairing_registry.get("FF++")

    # Create instance directly
    instance = pairing_registry.create("FaceForensics++")
"""

import logging
from collections.abc import Callable
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StrategyRegistry(Generic[T]):
    """
    Generic registry for pluggable strategy classes.

    Supports:
    - Multiple aliases per strategy (e.g., "FaceForensics++", "FF++", "FF")
    - Type checking against base class
    - Factory method for direct instantiation
    - Introspection for available strategies

    Attributes:
        name: Human-readable name for this registry (for error messages)
        base_class: ABC that all registered strategies must inherit from
    """

    def __init__(self, name: str, base_class: type[T]):
        """
        Initialize a strategy registry.

        Args:
            name: Registry identifier (e.g., "pairing", "extraction")
            base_class: Abstract base class that strategies must inherit
        """
        self.name = name
        self.base_class = base_class
        self._strategies: dict[str, type[T]] = {}
        self._aliases: dict[str, str] = {}

    def register(self, *names: str) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a strategy class with one or more names.

        The first name is considered the "canonical" name, and subsequent
        names are aliases that resolve to the canonical name.

        Args:
            *names: One or more names/aliases for this strategy

        Returns:
            Decorator function

        Raises:
            ValueError: If no names provided
            TypeError: If class doesn't inherit from base_class

        Example:
            @registry.register("FaceForensics++", "FF++", "FF")
            class FaceForensicsPairing(BasePairingStrategy):
                ...
        """
        if not names:
            raise ValueError("At least one name must be provided for registration")

        def decorator(cls: type[T]) -> type[T]:
            if not issubclass(cls, self.base_class):
                raise TypeError(f"Cannot register {cls.__name__}: must inherit from {self.base_class.__name__}")

            primary = names[0]
            self._strategies[primary] = cls
            logger.debug(f"Registered {self.name} strategy: {primary} -> {cls.__name__}")

            for alias in names[1:]:
                self._aliases[alias] = primary
                logger.debug(f"Registered {self.name} alias: {alias} -> {primary}")

            return cls

        return decorator

    def register_class(self, cls: type[T], *names: str) -> None:
        """
        Register a strategy class programmatically (non-decorator form).

        Args:
            cls: Strategy class to register
            *names: One or more names/aliases
        """
        self.register(*names)(cls)

    def get(self, name: str) -> type[T] | None:
        """
        Get strategy class by name or alias.

        Args:
            name: Strategy name or alias

        Returns:
            Strategy class if found, None otherwise
        """
        canonical = self._aliases.get(name, name)
        return self._strategies.get(canonical)

    def create(self, name: str, *args, **kwargs) -> T:
        """
        Create a strategy instance by name.

        Args:
            name: Strategy name or alias
            *args: Positional arguments for strategy constructor
            **kwargs: Keyword arguments for strategy constructor

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy name not found
        """
        cls = self.get(name)
        if cls is None:
            available = ", ".join(sorted(self.list_strategies()))
            raise ValueError(f"Unknown {self.name} strategy: '{name}'. Available: [{available}]")
        return cls(*args, **kwargs)

    def create_or_none(self, name: str, *args, **kwargs) -> T | None:
        """
        Create a strategy instance, returning None if not found.

        Useful for optional strategies where absence is valid.

        Args:
            name: Strategy name or alias
            *args: Positional arguments for strategy constructor
            **kwargs: Keyword arguments for strategy constructor

        Returns:
            Strategy instance if found, None otherwise
        """
        cls = self.get(name)
        if cls is None:
            return None
        return cls(*args, **kwargs)

    def list_strategies(self) -> list[str]:
        """
        List all registered strategy names (canonical names only).

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())

    def list_all_names(self) -> list[str]:
        """
        List all registered names including aliases.

        Returns:
            List of all names and aliases
        """
        return list(self._strategies.keys()) + list(self._aliases.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a strategy name or alias is registered."""
        return self.get(name) is not None

    def __len__(self) -> int:
        """Return number of registered strategies (not counting aliases)."""
        return len(self._strategies)

    def __repr__(self) -> str:
        return f"StrategyRegistry(name={self.name!r}, strategies={list(self._strategies.keys())})"
