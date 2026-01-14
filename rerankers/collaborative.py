"""
Collaborative Filtering Scorer for personalized ranking.

Implements user-based and item-based collaborative filtering
to provide CF scores that can be blended with content-based rerankers.
"""

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from feedback_logger import FeedbackLogger, get_feedback_logger

logger = logging.getLogger(__name__)


class CollaborativeScorer:
    """
    Collaborative filtering scorer for computing user-item affinity.
    
    Supports:
    - User-based CF: Find similar users, predict based on their preferences
    - Item-based CF: Find similar items, predict based on co-engagement
    - Popularity-based: Global item popularity as fallback
    
    Scores are normalized to [0, 1] for easy blending with other signals.
    """
    
    def __init__(
        self,
        feedback_logger: Optional[FeedbackLogger] = None,
        user_similarity_weight: float = 0.4,
        item_similarity_weight: float = 0.4,
        popularity_weight: float = 0.2,
        min_common_items: int = 2,
        min_common_users: int = 2,
        cache_ttl: int = 100,  # Refresh cache every N queries
    ):
        """
        Initialize the collaborative scorer.
        
        Args:
            feedback_logger: FeedbackLogger instance for data access
            user_similarity_weight: Weight for user-based CF score
            item_similarity_weight: Weight for item-based CF score
            popularity_weight: Weight for popularity-based score
            min_common_items: Minimum common items for user similarity
            min_common_users: Minimum common users for item similarity
            cache_ttl: Number of queries before refreshing cached data
        """
        self.feedback_logger = feedback_logger or get_feedback_logger()
        
        # Weights for score blending (should sum to 1)
        self.user_sim_weight = user_similarity_weight
        self.item_sim_weight = item_similarity_weight
        self.popularity_weight = popularity_weight
        
        # Similarity thresholds
        self.min_common_items = min_common_items
        self.min_common_users = min_common_users
        
        # Caching
        self.cache_ttl = cache_ttl
        self._query_count = 0
        self._user_article_matrix: Optional[Dict[str, Dict[str, float]]] = None
        self._article_stats: Optional[Dict[str, Dict[str, Any]]] = None
        self._user_similarity_cache: Dict[Tuple[str, str], float] = {}
        self._max_popularity: float = 1.0
    
    def _refresh_cache_if_needed(self) -> None:
        """Refresh cached data periodically."""
        self._query_count += 1
        if self._query_count >= self.cache_ttl or self._user_article_matrix is None:
            self._user_article_matrix = self.feedback_logger.get_user_article_matrix()
            self._article_stats = self.feedback_logger.get_all_article_stats()
            self._user_similarity_cache.clear()
            
            # Compute max popularity for normalization
            if self._article_stats:
                self._max_popularity = max(
                    (s.get("total_reward", 0) for s in self._article_stats.values()),
                    default=1.0
                ) or 1.0
            
            self._query_count = 0
            logger.debug(f"Refreshed CF cache: {len(self._user_article_matrix)} users, {len(self._article_stats)} articles")
    
    def compute_user_similarity(self, user_id_1: str, user_id_2: str) -> float:
        """
        Compute cosine similarity between two users based on their ratings.
        
        Uses topic preference vectors from user profiles.
        
        Returns:
            Similarity score in [0, 1]
        """
        # Check cache
        cache_key = tuple(sorted([user_id_1, user_id_2]))
        if cache_key in self._user_similarity_cache:
            return self._user_similarity_cache[cache_key]
        
        self._refresh_cache_if_needed()
        
        if not self._user_article_matrix:
            return 0.0
        
        ratings_1 = self._user_article_matrix.get(user_id_1, {})
        ratings_2 = self._user_article_matrix.get(user_id_2, {})
        
        if not ratings_1 or not ratings_2:
            return 0.0
        
        # Find common items
        common_items = set(ratings_1.keys()) & set(ratings_2.keys())
        
        if len(common_items) < self.min_common_items:
            return 0.0
        
        # Compute cosine similarity on common items
        vec_1 = np.array([ratings_1[item] for item in common_items])
        vec_2 = np.array([ratings_2[item] for item in common_items])
        
        norm_1 = np.linalg.norm(vec_1)
        norm_2 = np.linalg.norm(vec_2)
        
        if norm_1 == 0 or norm_2 == 0:
            return 0.0
        
        similarity = np.dot(vec_1, vec_2) / (norm_1 * norm_2)
        similarity = (similarity + 1) / 2  # Normalize to [0, 1]
        
        # Cache result
        self._user_similarity_cache[cache_key] = similarity
        return similarity
    
    def get_similar_users(
        self,
        user_id: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find top-k most similar users.
        
        Returns:
            List of (user_id, similarity) tuples, sorted by similarity desc
        """
        self._refresh_cache_if_needed()
        
        if not self._user_article_matrix:
            return []
        
        similarities = []
        for other_user in self._user_article_matrix.keys():
            if other_user == user_id:
                continue
            sim = self.compute_user_similarity(user_id, other_user)
            if sim > 0:
                similarities.append((other_user, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def predict_user_cf_score(
        self,
        user_id: str,
        article_id: str,
        similar_users: Optional[List[Tuple[str, float]]] = None
    ) -> float:
        """
        Predict user-item affinity using user-based collaborative filtering.
        
        Uses weighted average of similar users' ratings.
        
        Returns:
            Predicted score in [0, 1]
        """
        self._refresh_cache_if_needed()
        
        if similar_users is None:
            similar_users = self.get_similar_users(user_id, top_k=20)
        
        if not similar_users:
            return 0.0
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for other_user, similarity in similar_users:
            other_ratings = self._user_article_matrix.get(other_user, {})
            if article_id in other_ratings:
                rating = other_ratings[article_id]
                weighted_sum += similarity * rating
                weight_total += similarity
        
        if weight_total == 0:
            return 0.0
        
        # Normalize to [0, 1] - assuming ratings are roughly in [0, 20] range
        raw_score = weighted_sum / weight_total
        return min(raw_score / 20.0, 1.0)
    
    def predict_item_cf_score(
        self,
        user_id: str,
        article_id: str
    ) -> float:
        """
        Predict user-item affinity using item-based collaborative filtering.
        
        Uses similarity to items the user has previously engaged with.
        
        Returns:
            Predicted score in [0, 1]
        """
        self._refresh_cache_if_needed()
        
        # Get articles the user has engaged with
        user_articles = self.feedback_logger.get_articles_liked_by_user(user_id, min_reward=1.0)
        
        if not user_articles:
            return 0.0
        
        # Get co-engagement scores for similar articles
        similar_articles = self.feedback_logger.get_similar_articles(article_id, limit=50)
        
        if not similar_articles:
            # Fallback: check if article is in user's liked set
            user_ratings = self._user_article_matrix.get(user_id, {})
            if article_id in user_ratings:
                return min(user_ratings[article_id] / 20.0, 1.0)
            return 0.0
        
        # Compute weighted score based on similarity to user's liked articles
        weighted_sum = 0.0
        weight_total = 0.0
        
        user_article_set = set(user_articles)
        for sim_article in similar_articles:
            sim_id = sim_article["similar_article_id"]
            sim_score = sim_article["similarity_score"]
            
            if sim_id in user_article_set:
                # User has engaged with a similar article
                user_ratings = self._user_article_matrix.get(user_id, {})
                user_rating = user_ratings.get(sim_id, 0)
                weighted_sum += sim_score * user_rating
                weight_total += sim_score
        
        if weight_total == 0:
            return 0.0
        
        raw_score = weighted_sum / weight_total
        return min(raw_score / 20.0, 1.0)
    
    def get_popularity_score(self, article_id: str) -> float:
        """
        Get normalized popularity score for an article.
        
        Returns:
            Popularity score in [0, 1]
        """
        self._refresh_cache_if_needed()
        
        if not self._article_stats or article_id not in self._article_stats:
            return 0.0
        
        stats = self._article_stats[article_id]
        total_reward = stats.get("total_reward", 0)
        
        return min(total_reward / self._max_popularity, 1.0)
    
    def predict_cf_score(
        self,
        user_id: str,
        article_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Compute combined CF score for a user-article pair.
        
        Returns:
            Dictionary with individual scores and combined score
        """
        self._refresh_cache_if_needed()
        
        # Get similar users (cache for this session)
        similar_users = self.get_similar_users(user_id, top_k=20)
        
        # Compute individual scores
        user_cf_score = self.predict_user_cf_score(user_id, article_id, similar_users)
        item_cf_score = self.predict_item_cf_score(user_id, article_id)
        popularity_score = self.get_popularity_score(article_id)
        
        # Weighted combination
        combined_score = (
            self.user_sim_weight * user_cf_score +
            self.item_sim_weight * item_cf_score +
            self.popularity_weight * popularity_score
        )
        
        return {
            "cf_user_score": user_cf_score,
            "cf_item_score": item_cf_score,
            "cf_popularity_score": popularity_score,
            "cf_combined_score": combined_score,
        }
    
    def batch_predict(
        self,
        user_id: str,
        article_ids: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Batch predict CF scores for multiple articles.
        
        More efficient than calling predict_cf_score individually.
        
        Returns:
            Dict mapping article_id -> score dict
        """
        self._refresh_cache_if_needed()
        
        # Pre-compute similar users once
        similar_users = self.get_similar_users(user_id, top_k=20)
        
        # Pre-fetch user's liked articles
        user_articles = self.feedback_logger.get_articles_liked_by_user(user_id, min_reward=1.0)
        user_article_set = set(user_articles)
        
        results = {}
        for article_id in article_ids:
            user_cf_score = self.predict_user_cf_score(user_id, article_id, similar_users)
            item_cf_score = self.predict_item_cf_score(user_id, article_id)
            popularity_score = self.get_popularity_score(article_id)
            
            combined_score = (
                self.user_sim_weight * user_cf_score +
                self.item_sim_weight * item_cf_score +
                self.popularity_weight * popularity_score
            )
            
            results[article_id] = {
                "cf_user_score": user_cf_score,
                "cf_item_score": item_cf_score,
                "cf_popularity_score": popularity_score,
                "cf_combined_score": combined_score,
            }
        
        return results
    
    def update_item_similarities(self, max_pairs: int = 1000) -> int:
        """
        Batch update item-item similarities based on co-engagement.
        
        Should be called periodically (e.g., after every N sessions).
        
        Returns:
            Number of pairs updated
        """
        self._refresh_cache_if_needed()
        
        if not self._user_article_matrix:
            return 0
        
        # Build item -> users mapping
        item_users: Dict[str, Set[str]] = defaultdict(set)
        for user_id, ratings in self._user_article_matrix.items():
            for article_id, rating in ratings.items():
                if rating > 0:
                    item_users[article_id].add(user_id)
        
        # Find co-engaged pairs and compute similarity
        pairs_updated = 0
        items = list(item_users.keys())
        
        for i, item_1 in enumerate(items):
            if pairs_updated >= max_pairs:
                break
            
            users_1 = item_users[item_1]
            if len(users_1) < self.min_common_users:
                continue
            
            for item_2 in items[i+1:]:
                if pairs_updated >= max_pairs:
                    break
                
                users_2 = item_users[item_2]
                common_users = users_1 & users_2
                
                if len(common_users) < self.min_common_users:
                    continue
                
                # Compute Jaccard similarity
                union_size = len(users_1 | users_2)
                similarity = len(common_users) / union_size if union_size > 0 else 0.0
                
                if similarity > 0.1:  # Only store meaningful similarities
                    self.feedback_logger.update_article_similarity(
                        item_1, item_2, similarity, len(common_users)
                    )
                    pairs_updated += 1
        
        logger.info(f"Updated {pairs_updated} item-item similarity pairs")
        return pairs_updated


# Singleton instance
_collaborative_scorer: Optional[CollaborativeScorer] = None


def get_collaborative_scorer(**kwargs) -> CollaborativeScorer:
    """Get or create singleton collaborative scorer."""
    global _collaborative_scorer
    if _collaborative_scorer is None:
        _collaborative_scorer = CollaborativeScorer(**kwargs)
    return _collaborative_scorer
