"""
Feature extraction utilities for reranking models.

Provides consistent feature extraction across all rerankers.
"""

import hashlib
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .collaborative import CollaborativeScorer


# Define known topics from the dataset
KNOWN_TOPICS = [
    "science and technology",
    "lifestyle and leisure", 
    "health",
    "sports",
    "politics",
    "business",
    "entertainment",
    "education",
    "environment",
    "crime",
]


def extract_text_features(text: str) -> Dict[str, float]:
    """Extract basic text features from article content."""
    if not text:
        return {
            "text_length": 0.0,
            "word_count": 0.0,
            "avg_word_length": 0.0,
            "sentence_count": 0.0,
            "avg_sentence_length": 0.0,
            "question_count": 0.0,
            "exclamation_count": 0.0,
            "uppercase_ratio": 0.0,
            "digit_ratio": 0.0,
        }
    
    words = text.split()
    word_count = len(words)
    
    # Sentence detection (simple)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Character analysis
    uppercase_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    total_chars = len(text) or 1
    
    return {
        "text_length": min(len(text) / 10000.0, 1.0),  # Normalize to ~1
        "word_count": min(word_count / 1000.0, 1.0),
        "avg_word_length": sum(len(w) for w in words) / max(word_count, 1) / 10.0,
        "sentence_count": min(sentence_count / 50.0, 1.0),
        "avg_sentence_length": word_count / max(sentence_count, 1) / 30.0,
        "question_count": text.count("?") / 10.0,
        "exclamation_count": text.count("!") / 10.0,
        "uppercase_ratio": uppercase_count / total_chars,
        "digit_ratio": digit_count / total_chars,
    }


def extract_topic_features(
    topics: List[str],
    user_topic_prefs: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Extract topic-based features.
    
    Args:
        topics: List of topics for the article
        user_topic_prefs: User's topic preference weights (from feedback)
    
    Returns:
        Topic feature dictionary
    """
    features = {}
    
    # One-hot encoding for known topics
    topic_set = set(t.lower() for t in topics)
    for topic in KNOWN_TOPICS:
        features[f"topic_{topic.replace(' ', '_')}"] = 1.0 if topic in topic_set else 0.0
    
    # Topic count
    features["topic_count"] = len(topics) / 5.0
    
    # User preference match
    if user_topic_prefs:
        total_pref = sum(user_topic_prefs.values()) or 1.0
        pref_score = 0.0
        for topic in topics:
            topic_lower = topic.lower()
            pref_score += user_topic_prefs.get(topic_lower, 0) / total_pref
        features["user_topic_match"] = min(pref_score, 1.0)
    else:
        features["user_topic_match"] = 0.0
    
    return features


def extract_position_features(
    position: int,
    es_score: float,
    total_results: int = 10
) -> Dict[str, float]:
    """Extract position and score-based features."""
    return {
        "position": position / max(total_results, 1),
        "position_inverse": 1.0 / (position + 1),
        "es_score": min(es_score / 20.0, 1.0) if es_score else 0.0,  # Normalize ES score
        "es_score_log": math.log1p(es_score) / 5.0 if es_score else 0.0,
        "is_top_3": 1.0 if position < 3 else 0.0,
        "is_top_5": 1.0 if position < 5 else 0.0,
    }


def extract_query_features(
    query_text: str,
    article_text: str,
    article_topics: List[str]
) -> Dict[str, float]:
    """Extract query-document match features."""
    query_words = set(query_text.lower().split())
    article_words = set(article_text.lower().split())
    
    # Word overlap
    overlap = query_words & article_words
    overlap_ratio = len(overlap) / max(len(query_words), 1)
    
    # Query terms in topics
    topic_text = " ".join(article_topics).lower()
    topic_overlap = sum(1 for w in query_words if w in topic_text)
    topic_overlap_ratio = topic_overlap / max(len(query_words), 1)
    
    return {
        "query_word_overlap": overlap_ratio,
        "query_topic_overlap": topic_overlap_ratio,
        "query_length": len(query_words) / 10.0,
    }


def extract_cf_features(
    article_id: str,
    user_id: str,
    cf_scorer: Optional["CollaborativeScorer"] = None,
    cf_scores: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Extract collaborative filtering features.
    
    Args:
        article_id: Article identifier
        user_id: User identifier
        cf_scorer: Optional CollaborativeScorer instance
        cf_scores: Pre-computed CF scores (if available)
        
    Returns:
        CF feature dictionary
    """
    if cf_scores:
        return {
            "cf_user_score": cf_scores.get("cf_user_score", 0.0),
            "cf_item_score": cf_scores.get("cf_item_score", 0.0),
            "cf_popularity_score": cf_scores.get("cf_popularity_score", 0.0),
            "cf_combined_score": cf_scores.get("cf_combined_score", 0.0),
        }
    
    if cf_scorer is not None:
        scores = cf_scorer.predict_cf_score(user_id, article_id)
        return {
            "cf_user_score": scores.get("cf_user_score", 0.0),
            "cf_item_score": scores.get("cf_item_score", 0.0),
            "cf_popularity_score": scores.get("cf_popularity_score", 0.0),
            "cf_combined_score": scores.get("cf_combined_score", 0.0),
        }
    
    # Default: no CF data available
    return {
        "cf_user_score": 0.0,
        "cf_item_score": 0.0,
        "cf_popularity_score": 0.0,
        "cf_combined_score": 0.0,
    }


def extract_all_features(
    article: Dict[str, Any],
    user_id: str,
    query_text: str,
    position: int,
    user_profile: Optional[Dict[str, Any]] = None,
    total_results: int = 10,
    cf_scorer: Optional["CollaborativeScorer"] = None,
    cf_scores: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Extract all features for an article.
    
    Args:
        article: Elasticsearch hit document
        user_id: User identifier
        query_text: Search query
        position: Position in result list
        user_profile: User profile with topic preferences
        total_results: Total number of results
        cf_scorer: Optional CollaborativeScorer for CF features
        cf_scores: Pre-computed CF scores (more efficient for batch)
        
    Returns:
        Complete feature dictionary
    """
    source = article.get("_source", {})
    text = source.get("text", "")
    topics = source.get("topics", [])
    es_score = article.get("_score", 0.0)
    article_id = article.get("_id", "")
    
    # Get user topic preferences
    user_topic_prefs = None
    if user_profile and "topic_counts" in user_profile:
        user_topic_prefs = user_profile["topic_counts"]
    
    features = {}
    
    # Combine all feature types
    features.update(extract_text_features(text))
    features.update(extract_topic_features(topics, user_topic_prefs))
    features.update(extract_position_features(position, es_score, total_results))
    features.update(extract_query_features(query_text, text, topics))
    
    # Add CF features
    features.update(extract_cf_features(article_id, user_id, cf_scorer, cf_scores))
    
    # Add user-based features
    if user_profile:
        features["user_total_interactions"] = min(
            user_profile.get("total_interactions", 0) / 100.0, 1.0
        )
        features["user_avg_reward"] = (
            user_profile.get("total_reward", 0) / 
            max(user_profile.get("total_interactions", 1), 1)
        ) / 10.0
    else:
        features["user_total_interactions"] = 0.0
        features["user_avg_reward"] = 0.0
    
    return features


def get_feature_names() -> List[str]:
    """Get ordered list of feature names for vectorization."""
    # This ensures consistent feature ordering
    dummy_features = extract_all_features(
        article={"_source": {"text": "test", "topics": []}, "_score": 1.0},
        user_id="test",
        query_text="test",
        position=0,
        user_profile=None
    )
    # Include offline-only aggregate features so training/inference dimensions align
    offline_extra = {
        "article_ctr": 0.0,
        "article_avg_reward": 0.0,
        "article_impression_count": 0.0,
        "user_ctr": 0.0,
        "user_action_rate": 0.0,
        "position_propensity": 0.0,
    }
    dummy_features.update(offline_extra)
    return sorted(dummy_features.keys())


def features_to_vector(features: Dict[str, float], feature_names: List[str]) -> List[float]:
    """Convert feature dict to ordered vector."""
    return [features.get(name, 0.0) for name in feature_names]


def vector_to_features(vector: List[float], feature_names: List[str]) -> Dict[str, float]:
    """Convert ordered vector back to feature dict."""
    return dict(zip(feature_names, vector))
