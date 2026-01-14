"""
Hybrid Reranker combining content-based learning with collaborative filtering.

This reranker blends scores from:
1. A base content-based reranker (linear, GBDT, etc.)
2. Collaborative filtering signals (user-based, item-based)
3. Elasticsearch BM25 scores

The mixing weights are adaptive based on data availability and user history.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseReranker, RerankerConfig
from .collaborative import CollaborativeScorer, get_collaborative_scorer
from .factory import register_reranker, RerankerFactory
from .features import (
    extract_all_features,
    get_feature_names,
    features_to_vector,
)
from feedback_logger import get_feedback_logger

logger = logging.getLogger(__name__)


@register_reranker("hybrid")
class HybridReranker(BaseReranker):
    """
    Hybrid reranker that blends content-based and collaborative filtering scores.
    
    Architecture:
        final_score = α * reranker_score + β * cf_score + γ * es_score
    
    Where:
        - reranker_score: From base content-based model (e.g., LinearReranker)
        - cf_score: Combined collaborative filtering score
        - es_score: Normalized Elasticsearch BM25 score
    
    The mixing weights (α, β, γ) adapt based on:
        - User history (cold-start users rely more on ES/CF)
        - Data availability (new system relies more on ES)
        - Per-user learned preferences
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        super().__init__(config or RerankerConfig(name="hybrid"))
        
        # Base reranker type (linear, gbdt, etc.)
        self.base_reranker_type = self.config.extra_params.get("base_reranker", "linear")
        
        # Initialize base reranker
        base_config = RerankerConfig(
            name=self.base_reranker_type,
            model_path=self.config.model_path,
            extra_params=self.config.extra_params.get("base_params", {})
        )
        self.base_reranker = RerankerFactory.create(self.base_reranker_type, config=base_config)
        
        # Initialize collaborative scorer
        cf_params = self.config.extra_params.get("cf_params", {})
        self.cf_scorer = CollaborativeScorer(**cf_params)
        
        # Default mixing weights (should sum to 1)
        self.default_reranker_weight = self.config.extra_params.get("reranker_weight", 0.5)
        self.default_cf_weight = self.config.extra_params.get("cf_weight", 0.3)
        self.default_es_weight = self.config.extra_params.get("es_weight", 0.2)
        
        # Adaptive weight parameters
        self.min_reranker_weight = self.config.extra_params.get("min_reranker_weight", 0.2)
        self.max_reranker_weight = self.config.extra_params.get("max_reranker_weight", 0.7)
        self.warmup_interactions = self.config.extra_params.get("warmup_interactions", 50)
        self.cf_warmup_interactions = self.config.extra_params.get("cf_warmup_interactions", 20)
        
        # Per-user weight learning
        self.learn_user_weights = self.config.extra_params.get("learn_user_weights", True)
        self.user_weights: Dict[str, Dict[str, float]] = {}  # user_id -> {alpha, beta, gamma}
        self.user_weight_momentum: Dict[str, Dict[str, float]] = {}
        self.weight_learning_rate = self.config.extra_params.get("weight_lr", 0.01)
        
        # Feature configuration (includes CF features)
        self.feature_names = get_feature_names()
        self.n_features = len(self.feature_names)
        
        # Stats tracking
        self.n_updates = 0
        self.total_reward = 0.0
        
        # Feedback logger
        self.feedback_logger = get_feedback_logger()
        
        # Update item similarities periodically
        self.similarity_update_interval = self.config.extra_params.get("similarity_update_interval", 500)
        self._queries_since_similarity_update = 0
        
        # Load model if path provided
        if self.config.model_path and os.path.exists(self.config.model_path):
            self.load(self.config.model_path)
    
    def _get_adaptive_weights(self, user_id: str) -> Tuple[float, float, float]:
        """
        Get adaptive mixing weights based on user history and data availability.
        
        Returns:
            Tuple of (reranker_weight, cf_weight, es_weight)
        """
        user_profile = self.feedback_logger.get_user_profile(user_id)
        user_interactions = user_profile.get("total_interactions", 0) if user_profile else 0
        
        # Check if user has learned weights
        if self.learn_user_weights and user_id in self.user_weights:
            weights = self.user_weights[user_id]
            return weights["alpha"], weights["beta"], weights["gamma"]
        
        # Cold-start: new users rely more on ES and CF
        if user_interactions < self.warmup_interactions:
            # Gradually increase reranker weight as we learn
            progress = user_interactions / self.warmup_interactions
            reranker_w = self.min_reranker_weight + progress * (self.default_reranker_weight - self.min_reranker_weight)
            
            # CF weight also increases with interactions (need some data for CF)
            if user_interactions < self.cf_warmup_interactions:
                cf_progress = user_interactions / self.cf_warmup_interactions
                cf_w = cf_progress * self.default_cf_weight
            else:
                cf_w = self.default_cf_weight
            
            # ES fills the rest
            es_w = 1.0 - reranker_w - cf_w
            es_w = max(0.1, es_w)  # Keep at least 10% ES
            
            # Renormalize to sum to 1
            total = reranker_w + cf_w + es_w
            return reranker_w / total, cf_w / total, es_w / total
        
        return self.default_reranker_weight, self.default_cf_weight, self.default_es_weight
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation to bound scores to [0, 1]."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
    
    def rerank(
        self,
        hits: List[Dict[str, Any]],
        user_id: str,
        query_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank hits using blended content-based and CF scores.
        """
        if not hits:
            return hits
        
        # Get adaptive weights for this user
        alpha, beta, gamma = self._get_adaptive_weights(user_id)
        
        # Pre-compute CF scores for all articles (batch is more efficient)
        article_ids = [hit.get("_id", "") for hit in hits]
        cf_scores_batch = self.cf_scorer.batch_predict(user_id, article_ids, context)
        
        # Normalize ES scores
        es_scores = [hit.get("_score", 0.0) or 0.0 for hit in hits]
        max_es = max(es_scores) if es_scores else 1.0
        
        # Get user profile for feature extraction
        user_profile = self.feedback_logger.get_user_profile(user_id)
        
        # Score each hit
        scored_hits = []
        for i, hit in enumerate(hits):
            article_id = hit.get("_id", "")
            
            # Get CF scores for this article
            cf_scores = cf_scores_batch.get(article_id, {})
            cf_combined = cf_scores.get("cf_combined_score", 0.0)
            
            # Extract features with CF scores included
            features = extract_all_features(
                article=hit,
                user_id=user_id,
                query_text=query_text,
                position=i,
                user_profile=user_profile,
                total_results=len(hits),
                cf_scorer=None,  # Already have scores
                cf_scores=cf_scores
            )
            feature_vec = np.array(features_to_vector(features, self.feature_names))
            
            # Get base reranker score
            # We need to call the base reranker's internal scoring
            base_score = self._get_base_reranker_score(hit, user_id, query_text, i, user_profile, len(hits))
            
            # Normalize ES score
            es_score_norm = es_scores[i] / max_es if max_es > 0 else 0.0
            
            # Blend scores
            final_score = (
                alpha * self._sigmoid(base_score) +
                beta * cf_combined +
                gamma * es_score_norm
            )
            
            scored_hits.append({
                "hit": hit,
                "final_score": final_score,
                "base_score": base_score,
                "cf_score": cf_combined,
                "es_score": es_score_norm,
                "features": feature_vec,
                "cf_details": cf_scores,
            })
        
        # Sort by final score (descending)
        scored_hits.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Update similarity periodically
        self._queries_since_similarity_update += 1
        if self._queries_since_similarity_update >= self.similarity_update_interval:
            self.cf_scorer.update_item_similarities(max_pairs=500)
            self._queries_since_similarity_update = 0
        
        # Return re-ranked hits with metadata
        result = []
        for item in scored_hits:
            hit = item["hit"]
            hit["_hybrid_scores"] = {
                "final": item["final_score"],
                "base": item["base_score"],
                "cf": item["cf_score"],
                "es": item["es_score"],
            }
            result.append(hit)
        
        return result
    
    def _get_base_reranker_score(
        self,
        hit: Dict[str, Any],
        user_id: str,
        query_text: str,
        position: int,
        user_profile: Optional[Dict[str, Any]],
        total_results: int
    ) -> float:
        """Get score from base reranker's internal model."""
        # Access base reranker's scoring mechanism
        if hasattr(self.base_reranker, "_score") and hasattr(self.base_reranker, "feature_names"):
            # LinearReranker style
            features = extract_all_features(
                article=hit,
                user_id=user_id,
                query_text=query_text,
                position=position,
                user_profile=user_profile,
                total_results=total_results
            )
            feature_vec = np.array(features_to_vector(features, self.base_reranker.feature_names))
            return self.base_reranker._score(feature_vec, user_id)
        else:
            # Fallback: use ES score
            return hit.get("_score", 0.0) or 0.0
    
    def update(
        self,
        user_id: str,
        query_text: str,
        ranked_articles: List[Dict[str, Any]],
        actions: List[List[Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update both base reranker and adaptive weights based on feedback.
        """
        if not ranked_articles or not actions:
            return
        
        # Update base reranker
        self.base_reranker.update(user_id, query_text, ranked_articles, actions, context)
        
        # Compute rewards
        rewards = [self.compute_reward(a) for a in actions]
        total_reward = sum(rewards)
        self.total_reward += total_reward
        self.n_updates += 1
        
        # Learn user-specific weights if enabled
        if self.learn_user_weights and total_reward > 0:
            self._update_user_weights(user_id, ranked_articles, rewards, context)
    
    def _update_user_weights(
        self,
        user_id: str,
        ranked_articles: List[Dict[str, Any]],
        rewards: List[float],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Learn optimal mixing weights for a user based on which component
        predicted the actual engagement better.
        """
        # Initialize user weights if needed
        if user_id not in self.user_weights:
            self.user_weights[user_id] = {
                "alpha": self.default_reranker_weight,
                "beta": self.default_cf_weight,
                "gamma": self.default_es_weight,
            }
            self.user_weight_momentum[user_id] = {
                "alpha": 0.0, "beta": 0.0, "gamma": 0.0
            }
        
        # Get hybrid scores from context (stored during reranking)
        # For now, use a simple heuristic: reward weights that correlated with engagement
        
        weights = self.user_weights[user_id]
        momentum = self.user_weight_momentum[user_id]
        
        # Compute gradient: increase weight for component that correlated with reward
        # This is a simplified online learning approach
        
        total_reward = sum(rewards)
        if total_reward == 0:
            return
        
        # Get scores for articles (would need to store these during rerank)
        # Simplified: adjust weights based on whether top-ranked items got engagement
        
        top_3_reward = sum(rewards[:3])
        bottom_reward = sum(rewards[3:]) if len(rewards) > 3 else 0
        
        if top_3_reward > bottom_reward:
            # Current weights are working, reinforce them slightly
            pass
        else:
            # Top items didn't get engagement, try shifting weights
            # Increase CF weight slightly (explore collaborative signal)
            grad_beta = 0.02
            grad_alpha = -0.01
            grad_gamma = -0.01
            
            # Apply momentum
            momentum["alpha"] = 0.9 * momentum["alpha"] + self.weight_learning_rate * grad_alpha
            momentum["beta"] = 0.9 * momentum["beta"] + self.weight_learning_rate * grad_beta
            momentum["gamma"] = 0.9 * momentum["gamma"] + self.weight_learning_rate * grad_gamma
            
            weights["alpha"] = np.clip(weights["alpha"] + momentum["alpha"], 0.1, 0.8)
            weights["beta"] = np.clip(weights["beta"] + momentum["beta"], 0.1, 0.5)
            weights["gamma"] = np.clip(weights["gamma"] + momentum["gamma"], 0.1, 0.5)
            
            # Renormalize
            total = weights["alpha"] + weights["beta"] + weights["gamma"]
            weights["alpha"] /= total
            weights["beta"] /= total
            weights["gamma"] /= total
    
    def extract_features(
        self,
        article: Dict[str, Any],
        user_id: str,
        query_text: str,
        position: int
    ) -> Dict[str, float]:
        """Extract features including CF features."""
        user_profile = self.feedback_logger.get_user_profile(user_id)
        article_id = article.get("_id", "")
        cf_scores = self.cf_scorer.predict_cf_score(user_id, article_id)
        
        return extract_all_features(
            article=article,
            user_id=user_id,
            query_text=query_text,
            position=position,
            user_profile=user_profile,
            cf_scorer=None,
            cf_scores=cf_scores
        )
    
    def save(self, path: str) -> None:
        """Save hybrid model state."""
        # Save base reranker
        base_path = path.replace(".json", f"_base_{self.base_reranker_type}.json")
        self.base_reranker.save(base_path)
        
        # Save hybrid-specific state
        state = {
            "base_reranker_type": self.base_reranker_type,
            "base_model_path": base_path,
            "default_weights": {
                "reranker": self.default_reranker_weight,
                "cf": self.default_cf_weight,
                "es": self.default_es_weight,
            },
            "user_weights": self.user_weights,
            "user_weight_momentum": self.user_weight_momentum,
            "n_updates": self.n_updates,
            "total_reward": self.total_reward,
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved hybrid model to {path}")
    
    def load(self, path: str) -> None:
        """Load hybrid model state."""
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return
        
        with open(path, "r") as f:
            state = json.load(f)
        
        # Load base reranker
        base_path = state.get("base_model_path")
        if base_path and os.path.exists(base_path):
            self.base_reranker.load(base_path)
        
        # Load hybrid state
        self.user_weights = state.get("user_weights", {})
        self.user_weight_momentum = state.get("user_weight_momentum", {})
        self.n_updates = state.get("n_updates", 0)
        self.total_reward = state.get("total_reward", 0.0)
        
        default_weights = state.get("default_weights", {})
        self.default_reranker_weight = default_weights.get("reranker", self.default_reranker_weight)
        self.default_cf_weight = default_weights.get("cf", self.default_cf_weight)
        self.default_es_weight = default_weights.get("es", self.default_es_weight)
        
        logger.info(f"Loaded hybrid model from {path} ({self.n_updates} updates)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        base_stats = self.base_reranker.get_stats()
        
        return {
            "name": self.name,
            "type": "hybrid",
            "base_reranker": self.base_reranker_type,
            "base_stats": base_stats,
            "n_updates": self.n_updates,
            "total_reward": self.total_reward,
            "n_users_with_learned_weights": len(self.user_weights),
            "default_weights": {
                "reranker": self.default_reranker_weight,
                "cf": self.default_cf_weight,
                "es": self.default_es_weight,
            },
        }
    
    def get_user_weights(self, user_id: str) -> Dict[str, float]:
        """Get mixing weights for a specific user."""
        alpha, beta, gamma = self._get_adaptive_weights(user_id)
        return {
            "reranker_weight": alpha,
            "cf_weight": beta,
            "es_weight": gamma,
        }
