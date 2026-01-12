"""
Baseline reranker - returns Elasticsearch results unchanged.

This serves as the control group for A/B testing experiments.
"""

from typing import Any, Dict, List, Optional

from .base import BaseReranker, RerankerConfig
from .factory import register_reranker


@register_reranker("baseline")
class BaselineReranker(BaseReranker):
    """
    Baseline reranker that returns results in original ES order.
    
    This is the control group - no personalization applied.
    Useful for establishing baseline metrics in A/B tests.
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        super().__init__(config)
    
    def rerank(
        self,
        hits: List[Dict[str, Any]],
        user_id: str,
        query_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Return hits unchanged (baseline BM25 ordering)."""
        return hits
    
    def update(
        self,
        user_id: str,
        query_text: str,
        ranked_articles: List[Dict[str, Any]],
        actions: List[List[Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Baseline doesn't learn - no-op."""
        pass
    
    def extract_features(
        self,
        article: Dict[str, Any],
        user_id: str,
        query_text: str,
        position: int
    ) -> Dict[str, float]:
        """Return minimal features for baseline."""
        return {
            "es_score": article.get("_score", 0.0),
            "position": float(position),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "baseline",
            "description": "No personalization - ES default ordering"
        }
