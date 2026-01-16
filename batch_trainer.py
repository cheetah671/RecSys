"""
Batch Training Pipeline for Personalized Ranking Models.

This module provides offline training capabilities:
- Load historical interaction data from feedback.db
- Feature extraction and analysis
- Handle sparse feedback (low action rates)
- Train GBDT/Linear models
- Precompute CF similarities
- Feature importance analysis for hidden feature discovery
"""

import json
import logging
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
import os
import gzip
import json

import numpy as np

from feedback_logger import FeedbackLogger, get_feedback_logger
from rerankers.features import (
    extract_all_features,
    get_feature_names,
    features_to_vector,
    KNOWN_TOPICS,
)
from rerankers.collaborative import CollaborativeScorer, get_collaborative_scorer

# Optional progress bar (fallback to no-op if not installed)
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm optional
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else []

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for batch training."""
    # Data loading
    min_interactions: int = 100  # Minimum interactions to train
    max_interactions: Optional[int] = None  # Limit data for faster training
    
    # Sparse data handling
    positive_sample_weight: float = 3.0  # Weight for samples with actions
    negative_subsample_rate: float = 0.3  # Keep 30% of zero-reward samples
    min_positive_samples: int = 20  # Minimum positive samples to train
    
    # Feature engineering
    add_ctr_features: bool = True  # Add click-through rate features
    add_position_bias_correction: bool = True  # Correct for position bias
    
    # Model training
    validation_split: float = 0.15
    
    # CF precomputation
    precompute_similarities: bool = True
    max_similarity_pairs: int = 5000

    # Performance/UX
    show_progress: bool = True
    # Number of workers to parallelize per-user CF batching (0/1 disables)
    cf_batch_workers: int = 0
    # Try to use GPU for LightGBM training if available (requires GPU-enabled LightGBM)
    use_gpu_for_gbdt: bool = False
    # Cache for prepared features to skip repeat extraction on subsequent runs
    feature_cache_path: Optional[str] = "models/prepared_features.jsonl.gz"


class BatchTrainer:
    """
    Batch training pipeline for ranking models.
    
    Handles the full offline training workflow:
    1. Load and prepare data from feedback.db
    2. Handle sparse feedback (oversample positives, subsample negatives)
    3. Extract and engineer features
    4. Train models (GBDT, Linear)
    5. Precompute CF similarities
    6. Analyze feature importance
    """
    
    def __init__(
        self,
        feedback_logger: Optional[FeedbackLogger] = None,
        config: Optional[TrainingConfig] = None
    ):
        self.feedback_logger = feedback_logger or get_feedback_logger()
        self.config = config or TrainingConfig()
        self.feature_names = get_feature_names()
        
        # Training statistics
        self.stats = {
            "total_interactions": 0,
            "positive_interactions": 0,
            "unique_users": 0,
            "unique_queries": 0,
            "unique_articles": 0,
            "action_rate": 0.0,
        }
    
    def load_training_data(
        self,
        limit: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load interaction data from feedback database.
        
        Returns:
            List of interaction records with features
        """
        limit = limit or self.config.max_interactions
        data = self.feedback_logger.get_training_data(limit=limit, user_id=user_id)
        
        logger.info(f"Loaded {len(data)} interaction records")
        
        # Compute statistics
        users = set()
        queries = set()
        articles = set()
        positive_count = 0
        
        for record in data:
            users.add(record["user_id"])
            queries.add(record["query_id"])
            articles.add(record["article_id"])
            if record["reward"] > 0:
                positive_count += 1
        
        self.stats = {
            "total_interactions": len(data),
            "positive_interactions": positive_count,
            "unique_users": len(users),
            "unique_queries": len(queries),
            "unique_articles": len(articles),
            "action_rate": positive_count / len(data) if data else 0,
        }
        
        logger.info(f"Data stats: {self.stats}")
        
        return data
    
    def prepare_features(
        self,
        data: List[Dict[str, Any]],
        include_cf: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract features for all interaction records.
        
        Handles sparse data by:
        1. Computing position-debiased features
        2. Adding CTR (click-through rate) features
        3. Adding user/article aggregate features
        """
        logger.info("Extracting features (batched + cached)...")

        n = len(data)
        feature_names = self.feature_names

        # Precompute aggregates for CTR features (vectorizable part)
        article_stats = self._compute_article_stats(data)
        user_stats = self._compute_user_stats(data)

        # Cache user profiles (avoid per-record DB calls)
        logger.debug("Caching user profiles...")
        unique_users = list({d["user_id"] for d in data})
        user_profiles: Dict[str, Optional[Dict[str, Any]]] = {}
        for uid in unique_users:
            user_profiles[uid] = self.feedback_logger.get_user_profile(uid)

        # Batch CF scores per user to reduce DB and similarity work
        cf_scorer = get_collaborative_scorer() if include_cf else None
        cf_scores_cache: Dict[Tuple[str, str], Dict[str, float]] = {}

        if include_cf and cf_scorer is not None:
            # Build per-user unique article id lists
            logger.debug("Batching CF predictions per user...")
            user_to_articles: Dict[str, List[str]] = defaultdict(list)
            seen_pairs: Set[Tuple[str, str]] = set()
            for rec in data:
                key = (rec["user_id"], rec["article_id"]) 
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                user_to_articles[rec["user_id"]].append(rec["article_id"])

            # Optionally parallelize across users
            def _predict_for_user(uid: str) -> Tuple[str, Dict[str, Dict[str, float]]]:
                article_ids = user_to_articles.get(uid, [])
                scores = cf_scorer.batch_predict(uid, article_ids) if article_ids else {}
                return uid, scores

            user_ids = list(user_to_articles.keys())
            scores_by_user: Dict[str, Dict[str, Dict[str, float]]] = {}

            if self.config.cf_batch_workers and self.config.cf_batch_workers > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                workers = self.config.cf_batch_workers
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(_predict_for_user, uid): uid for uid in user_ids}
                    for _ in tqdm(as_completed(futures), total=len(futures), desc="CF batches") if self.config.show_progress else as_completed(futures):
                        pass  # progress only; we'll gather after loop below
                    # Collect results (iterate futures again to get results)
                    for fut in futures:
                        uid, sc = fut.result()
                        scores_by_user[uid] = sc
            else:
                iterator = tqdm(user_ids, desc="CF batches") if self.config.show_progress else user_ids
                for uid in iterator:
                    uid, sc = _predict_for_user(uid)
                    scores_by_user[uid] = sc

            # Flatten into (user, article) -> scores
            for uid, mapping in scores_by_user.items():
                for aid, sc in mapping.items():
                    cf_scores_cache[(uid, aid)] = sc

        # Build output with a progress bar in original order
        prepared_data: List[Dict[str, Any]] = []
        iterator = tqdm(range(n), desc="Extracting features") if self.config.show_progress else range(n)
        for i in iterator:
            record = data[i]

            uid = record["user_id"]
            aid = record["article_id"]
            user_profile = user_profiles.get(uid)

            # Article skeleton for feature extraction (no full text in logs)
            article = {
                "_id": aid,
                "_score": record.get("es_score", 0.0),
                "_source": {
                    "text": "",  # Text not stored in logs
                    "topics": record.get("article_topics", []),
                }
            }

            # Use precomputed CF scores if available; avoid per-record DB in extract
            pre_cf = cf_scores_cache.get((uid, aid)) if include_cf else None
            features = extract_all_features(
                article=article,
                user_id=uid,
                query_text=record["query_text"],
                position=record["position"],
                user_profile=user_profile,
                total_results=10,
                cf_scorer=None if pre_cf is not None else cf_scorer,
                cf_scores=pre_cf,
            )

            # Add CTR/aggregate features (constant-time lookups)
            if self.config.add_ctr_features:
                astats = article_stats.get(aid)
                if astats:
                    features["article_ctr"] = astats["ctr"]
                    features["article_avg_reward"] = astats["avg_reward"]
                    features["article_impression_count"] = min(astats["impressions"] / 100, 1.0)
                else:
                    features["article_ctr"] = 0.0
                    features["article_avg_reward"] = 0.0
                    features["article_impression_count"] = 0.0

                ustats = user_stats.get(uid)
                if ustats:
                    features["user_ctr"] = ustats["ctr"]
                    features["user_action_rate"] = ustats["action_rate"]
                else:
                    features["user_ctr"] = 0.0
                    features["user_action_rate"] = 0.0

            # Position bias correction (cheap)
            if self.config.add_position_bias_correction:
                pos = record["position"]
                propensity = 1.0 / (1.0 + pos * 0.5)  # simple decay model
                features["position_propensity"] = propensity

            prepared_data.append({
                **record,
                "features": features_to_vector(features, feature_names),
                "feature_names": feature_names,
                "feature_dict": features,
            })

        logger.info(
            f"Prepared {len(prepared_data)} records with {len(feature_names) if prepared_data else 0} features"
        )

        return prepared_data
    
    def _compute_article_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compute per-article statistics for CTR features."""
        stats = defaultdict(lambda: {"impressions": 0, "clicks": 0, "total_reward": 0})
        
        for record in data:
            article_id = record["article_id"]
            stats[article_id]["impressions"] += 1
            if record["reward"] > 0:
                stats[article_id]["clicks"] += 1
            stats[article_id]["total_reward"] += record["reward"]
        
        # Compute derived stats
        for article_id, s in stats.items():
            s["ctr"] = s["clicks"] / s["impressions"] if s["impressions"] > 0 else 0
            s["avg_reward"] = s["total_reward"] / s["impressions"] if s["impressions"] > 0 else 0
        
        return dict(stats)
    
    def _compute_user_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compute per-user statistics."""
        stats = defaultdict(lambda: {"impressions": 0, "actions": 0, "clicks": 0})
        
        for record in data:
            user_id = record["user_id"]
            stats[user_id]["impressions"] += 1
            if record["reward"] > 0:
                stats[user_id]["actions"] += 1
            # Check for clicks in actions
            actions = record.get("actions", [])
            if isinstance(actions, str):
                actions = json.loads(actions)
            if "Click" in actions:
                stats[user_id]["clicks"] += 1
        
        # Compute derived stats
        for user_id, s in stats.items():
            s["action_rate"] = s["actions"] / s["impressions"] if s["impressions"] > 0 else 0
            s["ctr"] = s["clicks"] / s["impressions"] if s["impressions"] > 0 else 0
        
        return dict(stats)
    
    def handle_sparse_data(
        self,
        data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Handle sparse feedback by oversampling positives and subsampling negatives.
        
        Returns:
            Tuple of (balanced_data, sample_weights)
        """
        positive_samples = [d for d in data if d["reward"] > 0]
        negative_samples = [d for d in data if d["reward"] == 0]
        
        logger.info(f"Before balancing: {len(positive_samples)} positive, {len(negative_samples)} negative")
        
        # Subsample negatives
        n_negative_keep = int(len(negative_samples) * self.config.negative_subsample_rate)
        if n_negative_keep > 0:
            np.random.shuffle(negative_samples)
            negative_samples = negative_samples[:n_negative_keep]
        
        # Combine
        balanced_data = positive_samples + negative_samples
        np.random.shuffle(balanced_data)
        
        # Create sample weights
        weights = np.array([
            self.config.positive_sample_weight if d["reward"] > 0 else 1.0
            for d in balanced_data
        ])
        
        logger.info(f"After balancing: {len(positive_samples)} positive, {len(negative_samples)} negative")
        
        return balanced_data, weights
    
    def analyze_features(
        self,
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze feature distributions and correlations with reward.
        
        This helps discover hidden patterns and important features.
        """
        if not data or "feature_dict" not in data[0]:
            return {}
        
        feature_names = list(data[0]["feature_dict"].keys())
        
        # Compute feature statistics
        feature_stats = {}
        for feature_name in feature_names:
            values = [d["feature_dict"].get(feature_name, 0) for d in data]
            rewards = [d["reward"] for d in data]
            
            # Basic stats
            values_array = np.array(values)
            rewards_array = np.array(rewards)
            
            # Correlation with reward
            if np.std(values_array) > 0 and np.std(rewards_array) > 0:
                correlation = np.corrcoef(values_array, rewards_array)[0, 1]
            else:
                correlation = 0.0
            
            # Mean for positive vs negative samples
            positive_mask = rewards_array > 0
            mean_positive = values_array[positive_mask].mean() if positive_mask.sum() > 0 else 0
            mean_negative = values_array[~positive_mask].mean() if (~positive_mask).sum() > 0 else 0
            
            feature_stats[feature_name] = {
                "mean": float(values_array.mean()),
                "std": float(values_array.std()),
                "min": float(values_array.min()),
                "max": float(values_array.max()),
                "correlation_with_reward": float(correlation),
                "mean_when_positive": float(mean_positive),
                "mean_when_negative": float(mean_negative),
                "lift": float(mean_positive / mean_negative) if mean_negative > 0 else float('inf'),
            }
        
        # Sort by absolute correlation
        sorted_features = sorted(
            feature_stats.items(),
            key=lambda x: abs(x[1]["correlation_with_reward"]),
            reverse=True
        )
        
        # Top predictive features
        top_features = sorted_features[:10]
        
        # Topic analysis
        topic_rewards = defaultdict(list)
        for d in data:
            topics = d.get("article_topics", [])
            if isinstance(topics, str):
                topics = json.loads(topics)
            reward = d["reward"]
            for topic in topics:
                topic_rewards[topic].append(reward)
        
        topic_stats = {
            topic: {
                "count": len(rewards),
                "mean_reward": np.mean(rewards),
                "action_rate": np.mean([1 if r > 0 else 0 for r in rewards]),
            }
            for topic, rewards in topic_rewards.items()
        }
        
        return {
            "feature_stats": feature_stats,
            "top_predictive_features": [
                {"name": name, **stats} for name, stats in top_features
            ],
            "topic_stats": topic_stats,
            "data_stats": self.stats,
        }
    
    def precompute_similarities(
        self,
        max_pairs: int = 5000
    ) -> int:
        """
        Precompute item-item similarities for collaborative filtering.
        
        This is more accurate than incremental updates during serving.
        """
        logger.info("Precomputing item-item similarities...")
        
        cf_scorer = get_collaborative_scorer()
        
        # Force cache refresh
        cf_scorer._query_count = cf_scorer.cache_ttl
        cf_scorer._refresh_cache_if_needed()
        
        # Update similarities
        pairs_updated = cf_scorer.update_item_similarities(max_pairs=max_pairs)
        
        logger.info(f"Precomputed {pairs_updated} similarity pairs")
        
        return pairs_updated
    
    def train_gbdt(
        self,
        data: List[Dict[str, Any]],
        model_path: str = "models/gbdt_model.json"
    ) -> Dict[str, Any]:
        """
        Train GBDT model on prepared data.
        """
        from rerankers.gbdt import GBDTReranker, HAS_LIGHTGBM
        
        if not HAS_LIGHTGBM:
            logger.warning("LightGBM not available, using fallback training")
        
        # Handle sparse data
        balanced_data, weights = self.handle_sparse_data(data)
        
        if len([d for d in balanced_data if d["reward"] > 0]) < self.config.min_positive_samples:
            return {"error": "Not enough positive samples", "positive_count": len([d for d in balanced_data if d["reward"] > 0])}
        
        # Create and configure reranker
        from rerankers import RerankerConfig
        config = RerankerConfig(name="gbdt", model_path=model_path)
        reranker = GBDTReranker(config)

        # Optionally enable GPU for LightGBM if supported
        try:
            import os
            use_gpu = self.config.use_gpu_for_gbdt or os.getenv("LGBM_USE_GPU", "0") == "1"
        except Exception:
            use_gpu = False
        if HAS_LIGHTGBM and use_gpu:
            # Update only specific parameters to keep other defaults
            gpu_overrides = {
                "device": "gpu",      # Requires LightGBM built with GPU support
                "gpu_use_dp": False,   # single precision to save VRAM
                "max_bin": 255,
            }
            reranker.lgb_params.update({k: v for k, v in gpu_overrides.items() if k not in reranker.lgb_params or reranker.lgb_params.get(k) != v})
            logger.info("Enabled GPU acceleration for LightGBM (if available)")

        # Train on data
        result = reranker.train_on_data(balanced_data, validation_split=self.config.validation_split)
        
        # Save model
        reranker.save(model_path)
        
        return result
    
    def run_full_pipeline(
        self,
        model_path: str = "models/gbdt_model.json",
        analysis_path: str = "models/feature_analysis.json"
    ) -> Dict[str, Any]:
        """
        Run the full batch training pipeline.
        
        1. Load data
        2. Prepare features
        3. Analyze features (for hidden feature discovery)
        4. Handle sparse data
        5. Train model
        6. Precompute CF similarities
        7. Save results
        """
        results = {}
        
        # Step 1: Load data
        logger.info("=" * 60)
        logger.info("STEP 1: Loading data")
        logger.info("=" * 60)
        data = self.load_training_data()
        results["data_stats"] = self.stats
        
        if len(data) < self.config.min_interactions:
            return {
                "error": f"Not enough data: {len(data)} < {self.config.min_interactions}",
                "data_stats": self.stats
            }
        
        # Step 2: Prepare or load cached features
        logger.info("=" * 60)
        logger.info("STEP 2: Preparing features")
        logger.info("=" * 60)
        prepared_data: List[Dict[str, Any]]
        cache_used = False
        if self.config.feature_cache_path and os.path.exists(self.config.feature_cache_path):
            try:
                prepared_data = self._load_prepared_features(self.config.feature_cache_path)
                cache_used = True
                logger.info(f"Loaded prepared features from cache: {self.config.feature_cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load feature cache ({e}), recomputing...")
                prepared_data = self.prepare_features(data, include_cf=True)
        else:
            prepared_data = self.prepare_features(data, include_cf=True)
            # Save cache for future runs
            try:
                if self.config.feature_cache_path:
                    os.makedirs(os.path.dirname(self.config.feature_cache_path) or ".", exist_ok=True)
                    self._save_prepared_features(prepared_data, self.config.feature_cache_path)
                    logger.info(f"Saved prepared features to cache: {self.config.feature_cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save feature cache: {e}")
        results["feature_count"] = len(prepared_data[0]["feature_names"]) if prepared_data else 0
        
        # Step 3: Analyze features
        logger.info("=" * 60)
        logger.info("STEP 3: Analyzing features")
        logger.info("=" * 60)
        analysis = self.analyze_features(prepared_data)
        results["analysis"] = analysis
        
        # Save analysis
        os.makedirs(os.path.dirname(analysis_path) if os.path.dirname(analysis_path) else ".", exist_ok=True)
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Saved feature analysis to {analysis_path}")
        
        # Print top features
        if "top_predictive_features" in analysis:
            logger.info("\nTop predictive features:")
            for feat in analysis["top_predictive_features"][:5]:
                logger.info(f"  {feat['name']}: corr={feat['correlation_with_reward']:.3f}, lift={feat.get('lift', 0):.2f}")
        
        # Step 4: Train model
        logger.info("=" * 60)
        logger.info("STEP 4: Training GBDT model")
        logger.info("=" * 60)
        train_result = self.train_gbdt(prepared_data, model_path)
        results["training"] = train_result
        
        # Step 5: Precompute CF similarities
        if self.config.precompute_similarities:
            logger.info("=" * 60)
            logger.info("STEP 5: Precomputing CF similarities")
            logger.info("=" * 60)
            pairs = self.precompute_similarities(max_pairs=self.config.max_similarity_pairs)
            results["cf_pairs_computed"] = pairs
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return results

    # ----------------- Feature cache helpers -----------------
    def _save_prepared_features(self, data: List[Dict[str, Any]], path: str) -> None:
        """Save prepared features to a compressed JSONL file.

        Format:
        - First line: {"type": "meta", "feature_names": [...]} 
        - Next lines: {"type": "data", "user_id": ..., "query_id": ..., "article_id": ..., "reward": ..., "position": ..., "es_score": ..., "article_topics": [...], "features": [...]}.
        """
        if not data:
            return
        feature_names = data[0].get("feature_names", [])
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"type": "meta", "feature_names": feature_names}) + "\n")
            for rec in data:
                out = {
                    "type": "data",
                    "user_id": rec.get("user_id"),
                    "query_id": rec.get("query_id"),
                    "article_id": rec.get("article_id"),
                    "reward": rec.get("reward", 0),
                    "position": rec.get("position", 0),
                    "es_score": rec.get("es_score", 0.0),
                    "article_topics": rec.get("article_topics", []),
                    "features": rec.get("features", []),
                }
                f.write(json.dumps(out) + "\n")

    def _load_prepared_features(self, path: str) -> List[Dict[str, Any]]:
        """Load prepared features from a compressed JSONL cache and reconstruct records.
        Returns list in the same structure as prepare_features output (contains feature_dict reconstructed).
        """
        feature_names: List[str] = []
        records: List[Dict[str, Any]] = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("type") == "meta":
                    feature_names = obj.get("feature_names", [])
                elif obj.get("type") == "data":
                    vec = obj.get("features", [])
                    feature_dict = {name: vec[i] if i < len(vec) else 0.0 for i, name in enumerate(feature_names)}
                    rec = {
                        "user_id": obj.get("user_id"),
                        "query_id": obj.get("query_id"),
                        "article_id": obj.get("article_id"),
                        "reward": obj.get("reward", 0),
                        "position": obj.get("position", 0),
                        "es_score": obj.get("es_score", 0.0),
                        "article_topics": obj.get("article_topics", []),
                        "feature_names": feature_names,
                        "features": vec,
                        "feature_dict": feature_dict,
                    }
                    records.append(rec)
        return records


class PeriodicRetrainer:
    """
    Background retrainer that periodically updates models during online serving.
    
    Runs in a separate thread and triggers retraining when:
    - Enough new data has accumulated
    - A time interval has passed
    """
    
    def __init__(
        self,
        trainer: BatchTrainer,
        reranker,
        retrain_interval: int = 1000,  # interactions
        time_interval: int = 3600,  # seconds
        model_path: str = "models/gbdt_model.json"
    ):
        self.trainer = trainer
        self.reranker = reranker
        self.retrain_interval = retrain_interval
        self.time_interval = time_interval
        self.model_path = model_path
        
        self._last_retrain_count = 0
        self._last_retrain_time = datetime.now()
        self._is_running = False
        self._thread = None
    
    def check_and_retrain(self, current_count: int) -> bool:
        """
        Check if retraining is needed and trigger if so.
        
        Returns:
            True if retrained, False otherwise
        """
        # Check interaction count
        interactions_since_retrain = current_count - self._last_retrain_count
        time_since_retrain = (datetime.now() - self._last_retrain_time).total_seconds()
        
        should_retrain = (
            interactions_since_retrain >= self.retrain_interval or
            time_since_retrain >= self.time_interval
        )
        
        if should_retrain:
            logger.info(f"Triggering periodic retrain ({interactions_since_retrain} interactions, {time_since_retrain:.0f}s)")
            self._do_retrain()
            return True
        
        return False
    
    def _do_retrain(self) -> None:
        """Execute retraining."""
        try:
            # Load recent data
            data = self.trainer.load_training_data(limit=10000)
            
            if len(data) < self.trainer.config.min_interactions:
                logger.warning("Not enough data for retraining")
                return
            
            # Prepare and train
            prepared_data = self.trainer.prepare_features(data, include_cf=True)
            result = self.trainer.train_gbdt(prepared_data, self.model_path)
            
            # Reload model into reranker
            if "error" not in result:
                self.reranker.load(self.model_path)
                logger.info(f"Periodic retrain complete: {result.get('samples', 0)} samples")
            
            self._last_retrain_count = self.trainer.stats["total_interactions"]
            self._last_retrain_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Periodic retrain failed: {e}")
    
    def start_background(self) -> None:
        """Start background retraining thread."""
        import threading
        
        def _background_loop():
            import time
            while self._is_running:
                self.check_and_retrain(self.trainer.stats.get("total_interactions", 0))
                time.sleep(60)  # Check every minute
        
        self._is_running = True
        self._thread = threading.Thread(target=_background_loop, daemon=True)
        self._thread.start()
        logger.info("Started background retrainer")
    
    def stop(self) -> None:
        """Stop background retraining."""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Stopped background retrainer")
