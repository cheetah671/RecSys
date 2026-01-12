"""
Abstract base class for rerankers.

This module defines the interface that all rerankers must implement,
enabling easy extension and swapping of ranking strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RerankerConfig:
    """Configuration for a reranker instance."""
    name: str = "baseline"
    model_path: Optional[str] = None
    learning_rate: float = 0.01
    regularization: float = 0.001
    feature_dim: int = 10
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """
    Abstract base class for all rerankers.
    
    Subclasses must implement:
    - rerank(): Core ranking logic
    - update(): Learning from feedback
    - extract_features(): Feature extraction from documents
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self.name = self.config.name
    
    @abstractmethod
    def rerank(
        self,
        hits: List[Dict[str, Any]],
        user_id: str,
        query_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank a list of search hits based on user preferences.
        
        Args:
            hits: List of Elasticsearch hits (each with _id, _score, _source)
            user_id: The unique identifier of the user
            query_text: The search query text
            context: Additional context (query_id, etc.)
            
        Returns:
            Re-ranked list of hits
        """
        pass
    
    @abstractmethod
    def update(
        self,
        user_id: str,
        query_text: str,
        ranked_articles: List[Dict[str, Any]],
        actions: List[List[Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the model based on user feedback.
        
        Args:
            user_id: The unique identifier of the user
            query_text: The search query text
            ranked_articles: The articles that were shown (in ranked order)
            actions: User actions for each article (clicks, dwell time, etc.)
            context: Additional context
        """
        pass
    
    @abstractmethod
    def extract_features(
        self,
        article: Dict[str, Any],
        user_id: str,
        query_text: str,
        position: int
    ) -> Dict[str, float]:
        """
        Extract features from an article for ranking.
        
        Args:
            article: Article document (Elasticsearch hit)
            user_id: User identifier
            query_text: Query text
            position: Original position in results
            
        Returns:
            Dictionary of feature name -> feature value
        """
        pass
    
    def compute_reward(self, actions: List[Any]) -> float:
        """
        Convert user actions into a numerical reward score.
        
        Default scoring formula based on project description:
        Score = (Click × 1) + (Like × 5) + (Share × 10) + (Bookmark × 3) + (DwellTime × 0.1)
        
        Args:
            actions: List of actions for a single article
            
        Returns:
            Numerical reward score
        """
        if not actions:
            return 0.0
        
        score = 0.0
        for action in actions:
            if action == "Click":
                score += 1.0
            elif action == "Like":
                score += 5.0
            elif action == "Share":
                score += 10.0
            elif action == "Bookmark":
                score += 3.0
            elif isinstance(action, dict) and "Dwell" in action:
                dwell = action["Dwell"]
                secs = dwell.get("secs", 0)
                nanos = dwell.get("nanos", 0)
                total_secs = secs + nanos / 1e9
                score += total_secs * 0.1
        
        return score
    
    def save(self, path: str) -> None:
        """Save model state to disk. Override in subclasses."""
        pass
    
    def load(self, path: str) -> None:
        """Load model state from disk. Override in subclasses."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics. Override in subclasses."""
        return {"name": self.name}
