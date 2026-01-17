"""
A/B Testing framework using GrowthBook.

Provides integration with GrowthBook for running experiments
between different reranker strategies.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

from growthbook import GrowthBook, Feature

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    experiment_id: str
    feature_key: str
    variants: List[str]  # List of reranker names
    weights: Optional[List[float]] = None  # Traffic allocation weights
    enabled: bool = True
    description: str = ""
    
    def __post_init__(self):
        if self.weights is None:
            # Equal distribution by default
            self.weights = [1.0 / len(self.variants)] * len(self.variants)


class ABTestingManager:
    """
    Manages A/B testing for reranker experiments.
    
    Uses GrowthBook SDK for consistent user bucketing and
    feature flag management.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_host: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize A/B testing manager.
        
        Args:
            api_key: GrowthBook API key (or set GROWTHBOOK_API_KEY env var)
            api_host: GrowthBook API host (or set GROWTHBOOK_API_HOST env var)
            features: Static feature definitions (for offline/local mode)
        """
        self.api_key = api_key or os.getenv("GROWTHBOOK_API_KEY")
        self.api_host = api_host or os.getenv("GROWTHBOOK_API_HOST", "https://cdn.growthbook.io")
        
        # Store experiments
        self.experiments: Dict[str, ExperimentConfig] = {}
        
        # Static features for local mode (when not using GrowthBook API)
        self._static_features = features or {}
        
        # Track assignments
        self.assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {exp_id: variant}
    
    def register_experiment(self, config: ExperimentConfig) -> None:
        """Register an experiment configuration."""
        self.experiments[config.experiment_id] = config
        
        # Build feature definition for GrowthBook
        variations = []
        for i, variant in enumerate(config.variants):
            variations.append({
                "value": variant,
                "weight": config.weights[i] if config.weights else 1.0 / len(config.variants)
            })
        
        self._static_features[config.feature_key] = {
            "defaultValue": config.variants[0],
            "rules": [
                {
                    "variations": [v["value"] for v in variations],
                    "weights": [v["weight"] for v in variations],
                    "hashAttribute": "user_id",
                }
            ] if config.enabled else []
        }
        
        logger.info(f"Registered experiment: {config.experiment_id} with variants {config.variants}")
    
    def _create_growthbook(self, user_id: str) -> GrowthBook:
        """Create a GrowthBook instance for a user."""
        gb = GrowthBook(
            attributes={"user_id": user_id},
            features=self._static_features,
        )
        return gb
    
    def get_variant(
        self,
        user_id: str,
        experiment_id: str,
        default: Optional[str] = None
    ) -> str:
        """
        Get the variant assignment for a user in an experiment.
        
        Args:
            user_id: User identifier for bucketing
            experiment_id: Experiment identifier
            default: Default variant if experiment not found
            
        Returns:
            Assigned variant name (reranker name)
        """
        # Check if already assigned
        if user_id in self.assignments and experiment_id in self.assignments[user_id]:
            return self.assignments[user_id][experiment_id]
        
        # Get experiment config
        if experiment_id not in self.experiments:
            logger.warning(f"Unknown experiment: {experiment_id}")
            return default or "baseline"
        
        config = self.experiments[experiment_id]
        
        if not config.enabled:
            return config.variants[0]  # Control variant
        
        # Use GrowthBook for assignment
        gb = self._create_growthbook(user_id)
        variant = gb.get_feature_value(config.feature_key, config.variants[0])
        
        # Cache assignment
        if user_id not in self.assignments:
            self.assignments[user_id] = {}
        self.assignments[user_id][experiment_id] = variant
        
        logger.debug(f"User {user_id[:8]}... assigned to variant '{variant}' in {experiment_id}")
        
        return variant
    
    def get_reranker_for_user(
        self,
        user_id: str,
        experiment_id: str = "reranker_experiment"
    ) -> str:
        """
        Get the reranker name for a user based on experiment assignment.
        
        This is the main entry point for selecting which reranker to use.
        """
        return self.get_variant(user_id, experiment_id)
    
    def is_in_experiment(self, user_id: str, experiment_id: str) -> bool:
        """Check if a user is enrolled in an experiment."""
        return (
            user_id in self.assignments and 
            experiment_id in self.assignments[user_id]
        )
    
    def get_assignment_info(self, user_id: str) -> Dict[str, str]:
        """Get all experiment assignments for a user."""
        return self.assignments.get(user_id, {})
    
    def force_variant(
        self,
        user_id: str,
        experiment_id: str,
        variant: str
    ) -> None:
        """Force a specific variant for a user (useful for testing)."""
        if user_id not in self.assignments:
            self.assignments[user_id] = {}
        self.assignments[user_id][experiment_id] = variant
        logger.info(f"Forced user {user_id[:8]}... to variant '{variant}'")
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get summary of experiment assignments."""
        if experiment_id not in self.experiments:
            return {}
        
        config = self.experiments[experiment_id]
        variant_counts = {v: 0 for v in config.variants}
        
        for user_assignments in self.assignments.values():
            if experiment_id in user_assignments:
                variant = user_assignments[experiment_id]
                if variant in variant_counts:
                    variant_counts[variant] += 1
        
        total = sum(variant_counts.values())
        return {
            "experiment_id": experiment_id,
            "enabled": config.enabled,
            "variants": config.variants,
            "weights": config.weights,
            "total_users": total,
            "variant_distribution": {
                v: count / total if total > 0 else 0
                for v, count in variant_counts.items()
            }
        }


# Default experiment configuration
DEFAULT_RERANKER_EXPERIMENT = ExperimentConfig(
    experiment_id="reranker_ab_test",
    feature_key="reranker_variant",
    variants=["baseline", "linear", "hybrid"],
    weights=[0.33, 0.33, 0.34],
    enabled=True,
    description="A/B test comparing baseline BM25, linear personalization, and hybrid CF+reranker"
)

# Optional A/B test that includes the GBDT model trained offline
DEFAULT_GBDT_EXPERIMENT = ExperimentConfig(
    experiment_id="reranker_ab_test_gbdt",
    feature_key="reranker_variant_gbdt",
    variants=["baseline", "gbdt"],
    weights=[0.5, 0.5],
    enabled=True,
    description="A/B test comparing baseline BM25 with offline-trained GBDT reranker"
)


def create_default_ab_manager() -> ABTestingManager:
    """Create A/B testing manager with default experiment."""
    manager = ABTestingManager()
    manager.register_experiment(DEFAULT_RERANKER_EXPERIMENT)
    # Also register a GBDT-focused experiment so users can compare baseline vs GBDT
    manager.register_experiment(DEFAULT_GBDT_EXPERIMENT)
    return manager


# Singleton instance
_ab_manager: Optional[ABTestingManager] = None


def get_ab_manager() -> ABTestingManager:
    """Get or create singleton A/B testing manager."""
    global _ab_manager
    if _ab_manager is None:
        _ab_manager = create_default_ab_manager()
    return _ab_manager


def reset_ab_manager() -> None:
    """Reset the singleton A/B testing manager."""
    global _ab_manager
    _ab_manager = None
