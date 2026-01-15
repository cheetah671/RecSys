"""
GBDT (Gradient Boosted Decision Trees) Reranker using LightGBM.

This reranker uses batch-trained LightGBM for learning-to-rank.
It supports:
- Offline batch training on historical data
- Periodic retraining during online serving
- Pairwise ranking loss (LambdaRank)
"""

import json
import logging
import os
import pickle
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

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. GBDT reranker will use fallback linear model.")


@register_reranker("gbdt")
class GBDTReranker(BaseReranker):
    """
    GBDT-based reranker using LightGBM's LambdaRank.
    
    This is a batch-trained model that:
    - Learns non-linear feature interactions
    - Uses pairwise ranking loss for better ranking quality
    - Supports periodic retraining from accumulated data
    
    For online updates, it accumulates data and retrains periodically
    rather than updating per-query like LinearReranker.
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        super().__init__(config or RerankerConfig(name="gbdt"))
        
        # Feature configuration
        self.feature_names = get_feature_names()
        self.n_features = len(self.feature_names)
        
        # LightGBM parameters
        self.lgb_params = self.config.extra_params.get("lgb_params", {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3, 5, 10],
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 50,
            "verbose": -1,
        })
        self.num_boost_round = self.config.extra_params.get("num_boost_round", 300)
        self.early_stopping_rounds = self.config.extra_params.get("early_stopping_rounds", 20)
        
        # Model state
        self.model: Optional[lgb.Booster] = None if HAS_LIGHTGBM else None
        self._fallback_weights: Optional[np.ndarray] = None  # Fallback if no LightGBM
        
        # Batch training settings
        self.min_samples_to_train = self.config.extra_params.get("min_samples_to_train", 100)
        self.retrain_interval = self.config.extra_params.get("retrain_interval", 1000)
        
        # Data accumulator for periodic retraining
        self._accumulated_data: List[Dict[str, Any]] = []
        self._n_updates_since_train = 0
        self._is_trained = False
        
        # ES score blending (for cold-start before model is trained)
        self.min_es_weight = self.config.extra_params.get("min_es_weight", 0.2)
        self.max_es_weight = self.config.extra_params.get("max_es_weight", 0.9)
        
        # Stats
        self.n_updates = 0
        self.n_trains = 0
        self.last_train_samples = 0
        
        # Feedback logger
        self.feedback_logger = get_feedback_logger()
        
        # Load model if path provided
        if self.config.model_path and os.path.exists(self.config.model_path):
            self.load(self.config.model_path)
    
    def _get_es_weight(self) -> float:
        """Get ES score weight based on model training status."""
        if not self._is_trained:
            return self.max_es_weight
        # Decrease ES weight as model gets more training
        confidence = min(self.last_train_samples / 1000, 1.0)
        return self.max_es_weight - (self.max_es_weight - self.min_es_weight) * confidence
    
    def rerank(
        self,
        hits: List[Dict[str, Any]],
        user_id: str,
        query_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank hits using GBDT model predictions.
        """
        if not hits:
            return hits
        
        # Get user profile for feature extraction
        user_profile = self.feedback_logger.get_user_profile(user_id)
        
        # Extract features for all hits
        feature_matrix = []
        for i, hit in enumerate(hits):
            features = extract_all_features(
                article=hit,
                user_id=user_id,
                query_text=query_text,
                position=i,
                user_profile=user_profile,
                total_results=len(hits)
            )
            feature_vec = features_to_vector(features, self.feature_names)
            feature_matrix.append(feature_vec)
        
        X = np.array(feature_matrix)
        
        # Get model predictions
        if self._is_trained and self.model is not None:
            model_scores = self.model.predict(X)
        elif self._fallback_weights is not None:
            model_scores = X @ self._fallback_weights
        else:
            # No model yet - use ES scores only
            model_scores = np.zeros(len(hits))
        
        # Normalize model scores to [0, 1]
        if model_scores.max() > model_scores.min():
            model_scores_norm = (model_scores - model_scores.min()) / (model_scores.max() - model_scores.min())
        else:
            model_scores_norm = np.zeros_like(model_scores)
        
        # Normalize ES scores
        es_scores = np.array([hit.get("_score", 0.0) or 0.0 for hit in hits])
        if es_scores.max() > 0:
            es_scores_norm = es_scores / es_scores.max()
        else:
            es_scores_norm = np.zeros_like(es_scores)
        
        # Blend scores
        es_weight = self._get_es_weight()
        final_scores = es_weight * es_scores_norm + (1 - es_weight) * model_scores_norm
        
        # Sort by final score
        sorted_indices = np.argsort(final_scores)[::-1]
        
        return [hits[i] for i in sorted_indices]
    
    def update(
        self,
        user_id: str,
        query_text: str,
        ranked_articles: List[Dict[str, Any]],
        actions: List[List[Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Accumulate training data for periodic batch retraining.
        
        Unlike LinearReranker, this doesn't update per-query.
        Instead, it accumulates data and retrains when enough samples.
        """
        if not ranked_articles or not actions:
            return
        
        user_profile = self.feedback_logger.get_user_profile(user_id)
        
        # Accumulate samples
        for i, (article, article_actions) in enumerate(zip(ranked_articles, actions)):
            reward = self.compute_reward(article_actions)
            
            features = extract_all_features(
                article=article,
                user_id=user_id,
                query_text=query_text,
                position=i,
                user_profile=user_profile,
                total_results=len(ranked_articles)
            )
            
            self._accumulated_data.append({
                "features": features_to_vector(features, self.feature_names),
                "reward": reward,
                "query_id": context.get("query_id", "") if context else "",
                "user_id": user_id,
                "has_action": reward > 0,  # Flag for sparse action handling
            })
        
        self.n_updates += 1
        self._n_updates_since_train += 1
        
        # Check if should retrain
        if self._should_retrain():
            self._retrain()
    
    def _should_retrain(self) -> bool:
        """Check if we should trigger retraining."""
        # Need minimum samples
        if len(self._accumulated_data) < self.min_samples_to_train:
            return False
        
        # Retrain at intervals
        if self._n_updates_since_train >= self.retrain_interval:
            return True
        
        # Also retrain if we have enough positive samples
        positive_samples = sum(1 for d in self._accumulated_data if d["has_action"])
        if positive_samples >= 50 and not self._is_trained:
            return True
        
        return False
    
    def _retrain(self) -> None:
        """Retrain the model on accumulated data."""
        if not HAS_LIGHTGBM:
            self._retrain_fallback()
            return
        
        logger.info(f"Retraining GBDT model on {len(self._accumulated_data)} samples...")
        
        try:
            # Prepare training data
            X, y, groups = self._prepare_training_data()
            
            if len(X) < self.min_samples_to_train:
                logger.warning("Not enough data after preparation, skipping retrain")
                return
            
            # Create dataset
            train_data = lgb.Dataset(X, label=y, group=groups)
            
            # Train model
            self.model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=self.num_boost_round,
                valid_sets=[train_data],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),  # Suppress logging
                ]
            )
            
            self._is_trained = True
            self.n_trains += 1
            self.last_train_samples = len(X)
            self._n_updates_since_train = 0
            
            logger.info(f"GBDT model trained successfully (train #{self.n_trains}, {len(X)} samples)")
            
        except Exception as e:
            logger.error(f"Error training GBDT model: {e}")
            self._retrain_fallback()
    
    def _retrain_fallback(self) -> None:
        """Fallback training using simple linear regression."""
        logger.info("Using fallback linear training...")
        
        X, y, _ = self._prepare_training_data()
        
        if len(X) < 10:
            return
        
        # Simple ridge regression
        X = np.array(X)
        y = np.array(y)
        
        # Add regularization
        reg = 0.01
        XtX = X.T @ X + reg * np.eye(X.shape[1])
        Xty = X.T @ y
        
        try:
            self._fallback_weights = np.linalg.solve(XtX, Xty)
            self._is_trained = True
            self.n_trains += 1
            self.last_train_samples = len(X)
            self._n_updates_since_train = 0
            logger.info(f"Fallback model trained ({len(X)} samples)")
        except np.linalg.LinAlgError:
            logger.error("Failed to train fallback model")
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare training data for LambdaRank.
        
        Handles sparse actions by:
        1. Oversampling positive samples (with actions)
        2. Creating balanced query groups
        3. Using importance weighting
        
        Returns:
            X: Feature matrix
            y: Relevance labels (rewards)
            groups: Query group sizes for LambdaRank
        """
        # Group by query
        query_data: Dict[str, List[Dict]] = {}
        for sample in self._accumulated_data:
            qid = sample["query_id"]
            if qid not in query_data:
                query_data[qid] = []
            query_data[qid].append(sample)
        
        X_list = []
        y_list = []
        groups = []
        
        # Process each query group
        for qid, samples in query_data.items():
            if len(samples) < 2:
                continue  # Need at least 2 samples for pairwise learning
            
            # Check if this query has any positive feedback
            has_positive = any(s["has_action"] for s in samples)
            
            # For queries with positive feedback, include all samples
            # For queries without, subsample to reduce noise
            if has_positive:
                # Include all samples from queries with engagement
                for s in samples:
                    X_list.append(s["features"])
                    y_list.append(s["reward"])
                groups.append(len(samples))
            else:
                # Subsample negative-only queries (reduce noise from random impressions)
                # Keep only 20% of queries without any engagement
                if np.random.random() < 0.2:
                    for s in samples:
                        X_list.append(s["features"])
                        y_list.append(s["reward"])
                    groups.append(len(samples))
        
        # Finalize arrays
        X_array = np.array(X_list) if X_list else np.array([]).reshape(0, self.n_features)
        y_array = np.array(y_list) if y_list else np.array([])

        # For LambdaRank, labels must be integer relevance grades.
        # Map reward to graded relevance to give more signal than binary.
        # Simple bins: 0 -> 0, (0,5) -> 1, [5,10) -> 2, >=10 -> 3
        if y_array.size > 0:
            grades = np.zeros_like(y_array, dtype=int)
            grades[(y_array > 0) & (y_array < 5)] = 1
            grades[(y_array >= 5) & (y_array < 10)] = 2
            grades[(y_array >= 10)] = 3
            y_array = grades
        
        return X_array, y_array, groups
    
    def train_on_data(
        self,
        training_data: List[Dict[str, Any]],
        validation_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train model on provided data (for offline training).
        
        Args:
            training_data: List of interaction records from feedback_logger
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics
        """
        if not HAS_LIGHTGBM:
            logger.warning("LightGBM not available, using fallback")
            return self._train_fallback_on_data(training_data)
        
        logger.info(f"Training GBDT on {len(training_data)} records...")
        
        # Convert to accumulated format
        self._accumulated_data = []
        for record in training_data:
            features = record.get("features")
            if features is None:
                # Need to re-extract features
                continue
            
            self._accumulated_data.append({
                "features": features,
                "reward": record.get("reward", 0),
                "query_id": record.get("query_id", ""),
                "user_id": record.get("user_id", ""),
                "has_action": record.get("reward", 0) > 0,
            })
        
        # Prepare data
        X, y, groups = self._prepare_training_data()
        
        if len(X) < self.min_samples_to_train:
            return {"error": "Not enough training data", "samples": len(X)}
        
        # Split for validation
        n_val_groups = max(1, int(len(groups) * validation_split))
        n_train_groups = len(groups) - n_val_groups
        
        train_end = sum(groups[:n_train_groups])
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:], y[train_end:]
        groups_train = groups[:n_train_groups]
        groups_val = groups[n_train_groups:]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
        val_data = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_data)
        
        # Track metrics
        evals_result = {}
        
        # Train
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=10),
                lgb.record_evaluation(evals_result),
            ]
        )
        
        self._is_trained = True
        self.n_trains += 1
        self.last_train_samples = len(X)
        
        # Get feature importance
        importance = dict(zip(
            self.feature_names,
            self.model.feature_importance(importance_type="gain").tolist()
        ))
        
        return {
            "samples": len(X),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "num_queries": len(groups),
            "best_iteration": self.model.best_iteration,
            "feature_importance": importance,
            "metrics": evals_result,
        }
    
    def _train_fallback_on_data(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback training without LightGBM."""
        self._accumulated_data = []
        for record in training_data:
            if "features" in record:
                self._accumulated_data.append({
                    "features": record["features"],
                    "reward": record.get("reward", 0),
                    "query_id": record.get("query_id", ""),
                    "user_id": record.get("user_id", ""),
                    "has_action": record.get("reward", 0) > 0,
                })
        
        self._retrain_fallback()
        return {"samples": len(self._accumulated_data), "method": "fallback_linear"}
    
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
        """Save model state."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        state = {
            "n_updates": self.n_updates,
            "n_trains": self.n_trains,
            "last_train_samples": self.last_train_samples,
            "is_trained": self._is_trained,
            "feature_names": self.feature_names,
            "lgb_params": self.lgb_params,
        }
        
        # Save LightGBM model separately
        if self.model is not None:
            model_path = path.replace(".json", ".lgb")
            self.model.save_model(model_path)
            state["lgb_model_path"] = model_path
        
        # Save fallback weights
        if self._fallback_weights is not None:
            state["fallback_weights"] = self._fallback_weights.tolist()
        
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved GBDT model to {path}")
    
    def load(self, path: str) -> None:
        """Load model state."""
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return
        
        with open(path, "r") as f:
            state = json.load(f)
        
        self.n_updates = state.get("n_updates", 0)
        self.n_trains = state.get("n_trains", 0)
        self.last_train_samples = state.get("last_train_samples", 0)
        self._is_trained = state.get("is_trained", False)
        
        # Load LightGBM model
        lgb_model_path = state.get("lgb_model_path")
        if lgb_model_path and os.path.exists(lgb_model_path) and HAS_LIGHTGBM:
            self.model = lgb.Booster(model_file=lgb_model_path)
            logger.info(f"Loaded LightGBM model from {lgb_model_path}")
        
        # Load fallback weights
        if "fallback_weights" in state:
            self._fallback_weights = np.array(state["fallback_weights"])
        
        logger.info(f"Loaded GBDT model from {path} ({self.n_trains} trains)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        stats = {
            "name": self.name,
            "type": "gbdt",
            "n_features": self.n_features,
            "n_updates": self.n_updates,
            "n_trains": self.n_trains,
            "last_train_samples": self.last_train_samples,
            "is_trained": self._is_trained,
            "accumulated_samples": len(self._accumulated_data),
            "has_lightgbm": HAS_LIGHTGBM,
        }
        
        if self.model is not None and HAS_LIGHTGBM:
            importance = dict(zip(
                self.feature_names,
                self.model.feature_importance(importance_type="gain").tolist()
            ))
            # Top 5 features
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            stats["top_features"] = [{"name": k, "importance": v} for k, v in sorted_features]
        
        return stats
