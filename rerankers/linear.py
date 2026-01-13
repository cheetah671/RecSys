"""
Linear reranker with online learning from user feedback.

Implements a linear model that learns user preferences using
stochastic gradient descent with L2 regularization.
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseReranker, RerankerConfig
from .factory import register_reranker
from .features import (
    extract_all_features,
    get_feature_names,
    features_to_vector,
)
from feedback_logger import get_feedback_logger

logger = logging.getLogger(__name__)


@register_reranker("linear")
class LinearReranker(BaseReranker):
    """
    Linear reranker with online learning.
    
    Uses SGD to learn per-user weights that predict engagement.
    Supports both global weights and per-user personalization.
    
    Key improvements:
    - Exploration-exploitation balance during cold start
    - Decaying learning rate for stability
    - Momentum-based updates
    - Conservative ES score blending for new users
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        super().__init__(config or RerankerConfig(name="linear"))
        
        # Feature configuration
        self.feature_names = get_feature_names()
        self.n_features = len(self.feature_names)
        
        # Hyperparameters (tuned for better convergence)
        self.initial_learning_rate = self.config.extra_params.get("learning_rate", 0.005)
        self.learning_rate = self.initial_learning_rate
        self.regularization = self.config.extra_params.get("regularization", 0.0005)
        self.use_per_user_weights = self.config.extra_params.get("per_user", True)
        self.global_weight = self.config.extra_params.get("global_weight", 0.5)  # More weight to global initially
        self.momentum = self.config.extra_params.get("momentum", 0.9)
        
        # Exploration parameters
        self.exploration_rounds = self.config.extra_params.get("exploration_rounds", 500)
        self.min_es_weight = self.config.extra_params.get("min_es_weight", 0.3)
        self.max_es_weight = self.config.extra_params.get("max_es_weight", 0.8)
        
        # Model weights
        self.global_weights = np.zeros(self.n_features)
        self.global_bias = 0.0
        self.user_weights: Dict[str, np.ndarray] = {}
        self.user_biases: Dict[str, float] = {}
        
        # Momentum accumulators
        self.global_velocity = np.zeros(self.n_features)
        self.global_bias_velocity = 0.0
        self.user_velocities: Dict[str, np.ndarray] = {}
        
        # Per-user update counts (for adaptive learning)
        self.user_update_counts: Dict[str, int] = {}
        
        # Training stats
        self.n_updates = 0
        self.total_loss = 0.0
        self.recent_rewards: List[float] = []  # Track recent rewards for adaptation
        
        # Feedback logger for user profiles
        self.feedback_logger = get_feedback_logger()
        
        # Load model if path provided
        if self.config.model_path and os.path.exists(self.config.model_path):
            self.load(self.config.model_path)
    
    def _get_user_weights(self, user_id: str) -> Tuple[np.ndarray, float]:
        """Get weights for a user, initializing if needed."""
        if user_id not in self.user_weights:
            # Initialize with global weights (transfer learning from global model)
            self.user_weights[user_id] = self.global_weights.copy()
            self.user_biases[user_id] = self.global_bias
            self.user_velocities[user_id] = np.zeros(self.n_features)
            self.user_update_counts[user_id] = 0
        return self.user_weights[user_id], self.user_biases[user_id]
    
    def _get_adaptive_es_weight(self, user_id: str) -> float:
        """
        Get adaptive ES score weight based on user history.
        
        New users rely more on ES scores (exploitation of proven ranking).
        As we learn more about a user, we trust our model more.
        """
        user_updates = self.user_update_counts.get(user_id, 0)
        global_confidence = min(self.n_updates / self.exploration_rounds, 1.0)
        user_confidence = min(user_updates / 50, 1.0)  # Trust user model after ~50 interactions
        
        # Blend: start high (trust ES), decrease as we learn
        combined_confidence = 0.3 * global_confidence + 0.7 * user_confidence
        es_weight = self.max_es_weight - (self.max_es_weight - self.min_es_weight) * combined_confidence
        
        return es_weight
    
    def _get_adaptive_learning_rate(self, user_id: str) -> float:
        """
        Decaying learning rate for stability.
        
        Higher LR initially for fast learning, lower as model stabilizes.
        """
        user_updates = self.user_update_counts.get(user_id, 0)
        # Decay schedule: LR = initial_LR / (1 + decay * updates)
        decay = 0.001
        return self.initial_learning_rate / (1 + decay * (self.n_updates + user_updates))
    
    def _score(self, features: np.ndarray, user_id: str) -> float:
        """Compute ranking score for a feature vector."""
        user_w, user_b = self._get_user_weights(user_id)
        
        if self.use_per_user_weights:
            # Blend global and user-specific weights
            weights = (
                self.global_weight * self.global_weights + 
                (1 - self.global_weight) * user_w
            )
            bias = (
                self.global_weight * self.global_bias + 
                (1 - self.global_weight) * user_b
            )
        else:
            weights = self.global_weights
            bias = self.global_bias
        
        return float(np.dot(weights, features) + bias)
    
    def rerank(
        self,
        hits: List[Dict[str, Any]],
        user_id: str,
        query_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank hits based on learned user preferences.
        
        Computes scores using the linear model and sorts by predicted engagement.
        Uses adaptive ES weight based on confidence in learned model.
        """
        if not hits:
            return hits
        
        # Get user profile for feature extraction
        user_profile = self.feedback_logger.get_user_profile(user_id)
        
        # Get adaptive ES weight (more ES for new users, less as we learn)
        es_weight = self._get_adaptive_es_weight(user_id)
        
        # Normalize ES scores for proper blending
        es_scores = [hit.get("_score", 0.0) or 0.0 for hit in hits]
        max_es = max(es_scores) if es_scores else 1.0
        
        # Score each hit
        scored_hits = []
        for i, hit in enumerate(hits):
            # Extract features WITHOUT position bias (use original ES position)
            features = extract_all_features(
                article=hit,
                user_id=user_id,
                query_text=query_text,
                position=i,  # Original position for context
                user_profile=user_profile,
                total_results=len(hits)
            )
            feature_vec = np.array(features_to_vector(features, self.feature_names))
            
            # Compute personalized score
            personal_score = self._score(feature_vec, user_id)
            
            # Normalize ES score to [0, 1] range for fair blending
            es_score_norm = es_scores[i] / max_es if max_es > 0 else 0.0
            
            # Combine: adaptive blend of ES and personalized scores
            combined_score = es_weight * es_score_norm + (1 - es_weight) * self._sigmoid(personal_score)
            
            scored_hits.append((combined_score, hit, feature_vec))
        
        # Sort by combined score (descending)
        scored_hits.sort(key=lambda x: x[0], reverse=True)
        
        # Return re-ranked hits
        return [hit for _, hit, _ in scored_hits]
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation to bound scores to [0, 1]."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
    
    def update(
        self,
        user_id: str,
        query_text: str,
        ranked_articles: List[Dict[str, Any]],
        actions: List[List[Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update model weights using SGD with momentum based on user feedback.
        
        Uses pairwise learning: articles with higher rewards should score higher
        than articles with lower rewards.
        """
        if not ranked_articles or not actions:
            return
        
        user_profile = self.feedback_logger.get_user_profile(user_id)
        
        # Extract features and rewards
        samples = []
        for i, (article, article_actions) in enumerate(zip(ranked_articles, actions)):
            features = extract_all_features(
                article=article,
                user_id=user_id,
                query_text=query_text,
                position=i,
                user_profile=user_profile,
                total_results=len(ranked_articles)
            )
            feature_vec = np.array(features_to_vector(features, self.feature_names))
            reward = self.compute_reward(article_actions)
            samples.append((feature_vec, reward))
        
        # Get rewards and check if any positive signal
        rewards = np.array([r for _, r in samples])
        total_reward = rewards.sum()
        self.recent_rewards.append(total_reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        
        # Skip update if no engagement signal (avoid learning from noise)
        if total_reward == 0:
            return
        
        # Get adaptive learning rate
        lr = self._get_adaptive_learning_rate(user_id)
        
        # Ensure user weights exist
        user_w, user_b = self._get_user_weights(user_id)
        
        # Pairwise learning: compare articles with different rewards
        for i, (feat_i, reward_i) in enumerate(samples):
            for j, (feat_j, reward_j) in enumerate(samples):
                if reward_i <= reward_j:
                    continue  # Only learn from pairs where i has higher reward
                
                # We want score(i) > score(j), so push score(i) up and score(j) down
                score_i = self._score(feat_i, user_id)
                score_j = self._score(feat_j, user_id)
                
                # Margin: how much higher should i score than j?
                margin = 0.1 * (reward_i - reward_j)
                
                # If already correctly ranked with margin, skip
                if score_i - score_j >= margin:
                    continue
                
                # Gradient for pairwise hinge loss
                diff_vec = feat_i - feat_j
                grad = -diff_vec + self.regularization * self.global_weights
                
                # Update global weights with momentum
                self.global_velocity = self.momentum * self.global_velocity - lr * grad
                self.global_weights += self.global_velocity
                self.global_bias += lr * 0.1  # Small bias update
                
                # Update user-specific weights with momentum
                if self.use_per_user_weights:
                    user_grad = -diff_vec + self.regularization * user_w
                    self.user_velocities[user_id] = (
                        self.momentum * self.user_velocities[user_id] - lr * user_grad
                    )
                    self.user_weights[user_id] += self.user_velocities[user_id]
                    self.user_biases[user_id] += lr * 0.1
                
                # Track stats
                self.n_updates += 1
                loss = max(0, margin - (score_i - score_j))
                self.total_loss += loss
        
        # Increment user update count
        self.user_update_counts[user_id] = self.user_update_counts.get(user_id, 0) + 1
    
    def extract_features(
        self,
        article: Dict[str, Any],
        user_id: str,
        query_text: str,
        position: int
    ) -> Dict[str, float]:
        """Extract features for a single article."""
        user_profile = self.feedback_logger.get_user_profile(user_id)
        return extract_all_features(
            article=article,
            user_id=user_id,
            query_text=query_text,
            position=position,
            user_profile=user_profile
        )
    
    def save(self, path: str) -> None:
        """Save model state to JSON file."""
        state = {
            "global_weights": self.global_weights.tolist(),
            "global_bias": self.global_bias,
            "global_velocity": self.global_velocity.tolist(),
            "user_weights": {
                uid: w.tolist() for uid, w in self.user_weights.items()
            },
            "user_biases": self.user_biases,
            "user_velocities": {
                uid: v.tolist() for uid, v in self.user_velocities.items()
            },
            "user_update_counts": self.user_update_counts,
            "n_updates": self.n_updates,
            "feature_names": self.feature_names,
            "config": {
                "learning_rate": self.initial_learning_rate,
                "regularization": self.regularization,
                "use_per_user_weights": self.use_per_user_weights,
                "global_weight": self.global_weight,
                "momentum": self.momentum,
            }
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved linear model to {path}")
    
    def load(self, path: str) -> None:
        """Load model state from JSON file."""
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return
        
        with open(path, "r") as f:
            state = json.load(f)
        
        self.global_weights = np.array(state["global_weights"])
        self.global_bias = state.get("global_bias", 0.0)
        self.global_velocity = np.array(state.get("global_velocity", np.zeros(self.n_features)))
        self.user_weights = {
            uid: np.array(w) for uid, w in state.get("user_weights", {}).items()
        }
        self.user_biases = state.get("user_biases", {})
        self.user_velocities = {
            uid: np.array(v) for uid, v in state.get("user_velocities", {}).items()
        }
        self.user_update_counts = state.get("user_update_counts", {})
        self.n_updates = state.get("n_updates", 0)
        
        # Verify feature compatibility
        if state.get("feature_names") and state["feature_names"] != self.feature_names:
            logger.warning("Feature names mismatch - weights may be incompatible")
        
        logger.info(f"Loaded linear model from {path} ({self.n_updates} updates)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        avg_recent_reward = (
            sum(self.recent_rewards) / len(self.recent_rewards)
            if self.recent_rewards else 0.0
        )
        return {
            "name": self.name,
            "type": "linear",
            "n_features": self.n_features,
            "n_updates": self.n_updates,
            "n_users": len(self.user_weights),
            "avg_loss": self.total_loss / max(self.n_updates, 1),
            "avg_recent_reward": avg_recent_reward,
            "current_lr": self.learning_rate,
            "top_global_features": self._get_top_features(self.global_weights, k=5),
        }
    
    def _get_top_features(self, weights: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Get top-k features by absolute weight."""
        indices = np.argsort(np.abs(weights))[-k:][::-1]
        return [
            {"name": self.feature_names[i], "weight": float(weights[i])}
            for i in indices
        ]
    
    def get_user_top_features(self, user_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get top features for a specific user."""
        if user_id not in self.user_weights:
            return self._get_top_features(self.global_weights, k)
        return self._get_top_features(self.user_weights[user_id], k)
