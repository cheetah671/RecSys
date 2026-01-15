"""
Factory pattern for creating reranker instances.

Supports registration of custom rerankers and easy instantiation by name.
"""

import logging
from typing import Callable, Dict, Optional, Type

from .base import BaseReranker, RerankerConfig

logger = logging.getLogger(__name__)

# Global registry of reranker classes
_RERANKER_REGISTRY: Dict[str, Type[BaseReranker]] = {}


def register_reranker(name: str) -> Callable[[Type[BaseReranker]], Type[BaseReranker]]:
    """
    Decorator to register a reranker class.
    
    Usage:
        @register_reranker("my_reranker")
        class MyReranker(BaseReranker):
            ...
    """
    def decorator(cls: Type[BaseReranker]) -> Type[BaseReranker]:
        if name in _RERANKER_REGISTRY:
            logger.warning(f"Overwriting existing reranker: {name}")
        _RERANKER_REGISTRY[name] = cls
        logger.debug(f"Registered reranker: {name}")
        return cls
    return decorator


class RerankerFactory:
    """Factory for creating reranker instances."""
    
    @staticmethod
    def create(
        name: str,
        config: Optional[RerankerConfig] = None,
        **kwargs
    ) -> BaseReranker:
        """
        Create a reranker instance by name.
        
        Args:
            name: Name of the reranker (must be registered)
            config: Optional configuration object
            **kwargs: Additional parameters passed to config
            
        Returns:
            Instantiated reranker
            
        Raises:
            ValueError: If reranker name is not registered
        """
        if name not in _RERANKER_REGISTRY:
            available = list(_RERANKER_REGISTRY.keys())
            raise ValueError(
                f"Unknown reranker: '{name}'. Available: {available}"
            )
        
        # Build config if not provided
        if config is None:
            config = RerankerConfig(name=name, extra_params=kwargs)
        else:
            config.name = name
            config.extra_params.update(kwargs)
        
        cls = _RERANKER_REGISTRY[name]
        return cls(config=config)
    
    @staticmethod
    def list_available() -> list:
        """Return list of registered reranker names."""
        return list(_RERANKER_REGISTRY.keys())
    
    @staticmethod
    def is_registered(name: str) -> bool:
        """Check if a reranker is registered."""
        return name in _RERANKER_REGISTRY


def get_reranker_registry() -> Dict[str, Type[BaseReranker]]:
    """Get a copy of the reranker registry."""
    return _RERANKER_REGISTRY.copy()
