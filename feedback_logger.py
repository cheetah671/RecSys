"""
Feedback logger for storing and retrieving user interactions.

Logs all search sessions with user feedback for:
- Training personalized ranking models
- Offline analysis and experimentation
- A/B test result tracking
"""

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InteractionLog:
    """Single user-article interaction."""
    timestamp: str
    user_id: str
    query_id: str
    query_text: str
    article_id: str
    position: int
    es_score: float
    reward: float
    actions: List[Any]
    article_topics: List[str]
    article_text_length: int
    experiment_id: Optional[str] = None
    variant: Optional[str] = None
    extra_features: Optional[Dict[str, Any]] = None


class FeedbackLogger:
    """
    Thread-safe feedback logger using SQLite.
    
    Stores interaction data for model training and analysis.
    """
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._local = threading.local()
        # Global write lock to avoid concurrent write transactions causing SQLITE_BUSY
        self._write_lock = threading.Lock()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            # Increase timeout so SQLite waits if the DB is busy
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
            self._local.conn.row_factory = sqlite3.Row
            try:
                # Enable WAL journal for better concurrent read/write
                self._local.conn.execute("PRAGMA journal_mode=WAL;")
                # Reduce fsyncs while keeping reasonable durability
                self._local.conn.execute("PRAGMA synchronous=NORMAL;")
                # Set a busy timeout at the connection level
                self._local.conn.execute("PRAGMA busy_timeout=10000;")
            except Exception:
                pass
        return self._local.conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                query_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                article_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                es_score REAL,
                reward REAL NOT NULL,
                actions TEXT NOT NULL,
                article_topics TEXT,
                article_text_length INTEGER,
                experiment_id TEXT,
                variant TEXT,
                extra_features TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_user_id ON interactions(user_id);
            CREATE INDEX IF NOT EXISTS idx_article_id ON interactions(article_id);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_experiment ON interactions(experiment_id, variant);
            
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                topic_counts TEXT NOT NULL,
                total_interactions INTEGER DEFAULT 0,
                total_reward REAL DEFAULT 0.0,
                last_updated TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS experiment_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                variant TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                total_queries INTEGER DEFAULT 0,
                total_clicks INTEGER DEFAULT 0,
                total_reward REAL DEFAULT 0.0,
                avg_position_clicked REAL,
                UNIQUE(experiment_id, variant)
            );
            
            -- Collaborative filtering tables
            CREATE TABLE IF NOT EXISTS user_article_ratings (
                user_id TEXT NOT NULL,
                article_id TEXT NOT NULL,
                total_reward REAL DEFAULT 0.0,
                interaction_count INTEGER DEFAULT 0,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (user_id, article_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_uar_user ON user_article_ratings(user_id);
            CREATE INDEX IF NOT EXISTS idx_uar_article ON user_article_ratings(article_id);
            
            CREATE TABLE IF NOT EXISTS article_similarity (
                article_id_1 TEXT NOT NULL,
                article_id_2 TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                coengagement_count INTEGER DEFAULT 0,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (article_id_1, article_id_2)
            );
            
            CREATE INDEX IF NOT EXISTS idx_as_article1 ON article_similarity(article_id_1);
            CREATE INDEX IF NOT EXISTS idx_as_article2 ON article_similarity(article_id_2);
            
            CREATE TABLE IF NOT EXISTS article_stats (
                article_id TEXT PRIMARY KEY,
                total_impressions INTEGER DEFAULT 0,
                total_clicks INTEGER DEFAULT 0,
                total_reward REAL DEFAULT 0.0,
                avg_reward REAL DEFAULT 0.0,
                last_updated TEXT NOT NULL
            );
        """)
        conn.commit()
    
    def log_interaction(
        self,
        user_id: str,
        query_id: str,
        query_text: str,
        article: Dict[str, Any],
        position: int,
        actions: List[Any],
        reward: float,
        experiment_id: Optional[str] = None,
        variant: Optional[str] = None,
        extra_features: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a single user-article interaction."""
        conn = self._get_conn()
        
        source = article.get("_source", {})
        topics = source.get("topics", [])
        text = source.get("text", "")
        
        log = InteractionLog(
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            query_id=query_id,
            query_text=query_text,
            article_id=article.get("_id", ""),
            position=position,
            es_score=article.get("_score", 0.0),
            reward=reward,
            actions=actions,
            article_topics=topics if isinstance(topics, list) else [topics],
            article_text_length=len(text),
            experiment_id=experiment_id,
            variant=variant,
            extra_features=extra_features
        )
        
        # Use a write lock to serialize write operations and reduce SQLITE_BUSY
        with self._write_lock:
            conn.execute("""
            INSERT INTO interactions 
            (timestamp, user_id, query_id, query_text, article_id, position,
             es_score, reward, actions, article_topics, article_text_length,
             experiment_id, variant, extra_features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log.timestamp,
            log.user_id,
            log.query_id,
            log.query_text,
            log.article_id,
            log.position,
            log.es_score,
            log.reward,
            json.dumps(log.actions),
            json.dumps(log.article_topics),
            log.article_text_length,
            log.experiment_id,
            log.variant,
            json.dumps(log.extra_features) if log.extra_features else None
        ))
            conn.commit()
    
    def log_session(
        self,
        user_id: str,
        query_id: str,
        query_text: str,
        ranked_articles: List[Dict[str, Any]],
        actions: List[List[Any]],
        compute_reward_fn,
        experiment_id: Optional[str] = None,
        variant: Optional[str] = None
    ) -> None:
        """Log an entire search session (all articles and their feedback)."""
        for position, (article, article_actions) in enumerate(zip(ranked_articles, actions)):
            reward = compute_reward_fn(article_actions)
            # Serialize each insertion to avoid contention
            self.log_interaction(
                user_id=user_id,
                query_id=query_id,
                query_text=query_text,
                article=article,
                position=position,
                actions=article_actions,
                reward=reward,
                experiment_id=experiment_id,
                variant=variant
            )
            
            # Update CF tables
            article_id = article.get("_id", "")
            if article_id:
                clicked = any(act == "Click" for act in article_actions)
                self.update_user_article_rating(user_id, article_id, reward)
                self.update_article_stats(article_id, reward, clicked)
        
        # Update user profile
        self._update_user_profile(user_id, ranked_articles, actions, compute_reward_fn)
        
        # Update experiment metrics if applicable
        if experiment_id and variant:
            self._update_experiment_metrics(
                experiment_id, variant, actions, compute_reward_fn
            )
    
    def _update_user_profile(
        self,
        user_id: str,
        ranked_articles: List[Dict[str, Any]],
        actions: List[List[Any]],
        compute_reward_fn
    ) -> None:
        """Update user profile with topic preferences."""
        conn = self._get_conn()
        
        # Get existing profile
        row = conn.execute(
            "SELECT topic_counts, total_interactions, total_reward FROM user_profiles WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        
        if row:
            topic_counts = json.loads(row["topic_counts"])
            total_interactions = row["total_interactions"]
            total_reward = row["total_reward"]
        else:
            topic_counts = {}
            total_interactions = 0
            total_reward = 0.0
        
        # Update with new interactions
        for article, article_actions in zip(ranked_articles, actions):
            reward = compute_reward_fn(article_actions)
            if reward > 0:
                source = article.get("_source", {})
                topics = source.get("topics", [])
                for topic in topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + reward
            total_interactions += 1
            total_reward += reward
        
        # Upsert profile
        with self._write_lock:
            conn.execute("""
            INSERT INTO user_profiles (user_id, topic_counts, total_interactions, total_reward, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                topic_counts = excluded.topic_counts,
                total_interactions = excluded.total_interactions,
                total_reward = excluded.total_reward,
                last_updated = excluded.last_updated
        """, (
            user_id,
            json.dumps(topic_counts),
            total_interactions,
            total_reward,
            datetime.utcnow().isoformat()
        ))
        conn.commit()
    
    def _update_experiment_metrics(
        self,
        experiment_id: str,
        variant: str,
        actions: List[List[Any]],
        compute_reward_fn
    ) -> None:
        """Update experiment metrics for A/B testing."""
        conn = self._get_conn()
        
        total_clicks = sum(1 for a in actions if any(act == "Click" for act in a))
        total_reward = sum(compute_reward_fn(a) for a in actions)
        
        # Calculate average position of clicked articles
        clicked_positions = [i for i, a in enumerate(actions) if any(act == "Click" for act in a)]
        avg_pos = sum(clicked_positions) / len(clicked_positions) if clicked_positions else None
        
        with self._write_lock:
            conn.execute("""
            INSERT INTO experiment_metrics (experiment_id, variant, timestamp, total_queries, total_clicks, total_reward, avg_position_clicked)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(experiment_id, variant) DO UPDATE SET
                total_queries = experiment_metrics.total_queries + 1,
                total_clicks = experiment_metrics.total_clicks + excluded.total_clicks,
                total_reward = experiment_metrics.total_reward + excluded.total_reward,
                avg_position_clicked = CASE 
                    WHEN excluded.avg_position_clicked IS NOT NULL 
                    THEN (COALESCE(experiment_metrics.avg_position_clicked, 0) * experiment_metrics.total_queries + excluded.avg_position_clicked) / (experiment_metrics.total_queries + 1)
                    ELSE experiment_metrics.avg_position_clicked
                END,
                timestamp = excluded.timestamp
        """, (
            experiment_id,
            variant,
            datetime.utcnow().isoformat(),
            total_clicks,
            total_reward,
            avg_pos
        ))
        conn.commit()
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile with topic preferences."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return {
            "user_id": row["user_id"],
            "topic_counts": json.loads(row["topic_counts"]),
            "total_interactions": row["total_interactions"],
            "total_reward": row["total_reward"],
            "last_updated": row["last_updated"]
        }
    
    def get_user_interactions(
        self,
        user_id: str,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get recent interactions for a user."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM interactions 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (user_id, limit)).fetchall()
        
        return [dict(row) for row in rows]
    
# In feedback_logger.py, update the get_training_data method:

    def get_training_data(
        self,
        limit: Optional[int] = None,
        user_id: Optional[str] = None,
        since_id: Optional[int] = None  # <--- NEW PARAMETER
    ) -> List[Dict[str, Any]]:
        """Get interaction data for model training."""
        conn = self._get_conn()
        
        # Start building query
        query = "SELECT * FROM interactions"
        params = []
        conditions = []

        # Add filters
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if since_id is not None:
            conditions.append("id > ?")  # <--- Filter for new data
            params.append(since_id)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Sort by ID for incremental stability
        if since_id is not None:
            query += " ORDER BY id ASC" 
        else:
            query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        rows = conn.execute(query, params).fetchall()
        
        result = []
        for row in rows:
            data = dict(row)
            # ... existing json loads ...
            data["actions"] = json.loads(data["actions"])
            data["article_topics"] = json.loads(data["article_topics"])
            if data["extra_features"]:
                data["extra_features"] = json.loads(data["extra_features"])
            result.append(data)
        
        return result    
    def get_experiment_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get metrics for all variants in an experiment."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM experiment_metrics 
            WHERE experiment_id = ?
        """, (experiment_id,)).fetchall()
        
        return [dict(row) for row in rows]
    
    # =========================================================================
    # Collaborative Filtering Methods
    # =========================================================================
    
    def update_user_article_rating(
        self,
        user_id: str,
        article_id: str,
        reward: float
    ) -> None:
        """Update user-article rating for collaborative filtering."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO user_article_ratings (user_id, article_id, total_reward, interaction_count, last_updated)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(user_id, article_id) DO UPDATE SET
                total_reward = user_article_ratings.total_reward + excluded.total_reward,
                interaction_count = user_article_ratings.interaction_count + 1,
                last_updated = excluded.last_updated
        """, (user_id, article_id, reward, datetime.utcnow().isoformat()))
        conn.commit()
    
    def update_article_stats(
        self,
        article_id: str,
        reward: float,
        clicked: bool
    ) -> None:
        """Update global article statistics."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO article_stats (article_id, total_impressions, total_clicks, total_reward, avg_reward, last_updated)
            VALUES (?, 1, ?, ?, ?, ?)
            ON CONFLICT(article_id) DO UPDATE SET
                total_impressions = article_stats.total_impressions + 1,
                total_clicks = article_stats.total_clicks + excluded.total_clicks,
                total_reward = article_stats.total_reward + excluded.total_reward,
                avg_reward = (article_stats.total_reward + excluded.total_reward) / (article_stats.total_impressions + 1),
                last_updated = excluded.last_updated
        """, (article_id, 1 if clicked else 0, reward, reward, datetime.utcnow().isoformat()))
        conn.commit()
    
    def get_user_article_ratings(self, user_id: str) -> Dict[str, float]:
        """Get all article ratings for a user."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT article_id, total_reward, interaction_count 
            FROM user_article_ratings 
            WHERE user_id = ?
        """, (user_id,)).fetchall()
        
        return {
            row["article_id"]: row["total_reward"] / max(row["interaction_count"], 1)
            for row in rows
        }
    
    def get_article_stats(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get global statistics for an article."""
        conn = self._get_conn()
        row = conn.execute("""
            SELECT * FROM article_stats WHERE article_id = ?
        """, (article_id,)).fetchone()
        
        return dict(row) if row else None
    
    def get_all_article_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all articles."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM article_stats").fetchall()
        return {row["article_id"]: dict(row) for row in rows}
    
    def get_users_who_liked_article(self, article_id: str, min_reward: float = 1.0) -> List[str]:
        """Get users who positively engaged with an article."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT user_id FROM user_article_ratings 
            WHERE article_id = ? AND total_reward >= ?
        """, (article_id, min_reward)).fetchall()
        
        return [row["user_id"] for row in rows]
    
    def get_articles_liked_by_user(self, user_id: str, min_reward: float = 1.0) -> List[str]:
        """Get articles positively engaged by a user."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT article_id FROM user_article_ratings 
            WHERE user_id = ? AND total_reward >= ?
            ORDER BY total_reward DESC
        """, (user_id, min_reward)).fetchall()
        
        return [row["article_id"] for row in rows]
    
    def get_user_article_matrix(
        self,
        min_interactions: int = 1
    ) -> Dict[str, Dict[str, float]]:
        """
        Get user-article rating matrix for collaborative filtering.
        
        Returns:
            Dict mapping user_id -> {article_id: avg_reward}
        """
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT user_id, article_id, total_reward, interaction_count
            FROM user_article_ratings
            WHERE interaction_count >= ?
        """, (min_interactions,)).fetchall()
        
        matrix: Dict[str, Dict[str, float]] = {}
        for row in rows:
            user_id = row["user_id"]
            if user_id not in matrix:
                matrix[user_id] = {}
            matrix[user_id][row["article_id"]] = row["total_reward"] / max(row["interaction_count"], 1)
        
        return matrix
    
    def update_article_similarity(
        self,
        article_id_1: str,
        article_id_2: str,
        similarity_score: float,
        coengagement_count: int = 1
    ) -> None:
        """Update precomputed article-article similarity."""
        conn = self._get_conn()
        # Ensure consistent ordering
        if article_id_1 > article_id_2:
            article_id_1, article_id_2 = article_id_2, article_id_1
        
        conn.execute("""
            INSERT INTO article_similarity (article_id_1, article_id_2, similarity_score, coengagement_count, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(article_id_1, article_id_2) DO UPDATE SET
                similarity_score = excluded.similarity_score,
                coengagement_count = excluded.coengagement_count,
                last_updated = excluded.last_updated
        """, (article_id_1, article_id_2, similarity_score, coengagement_count, datetime.utcnow().isoformat()))
        conn.commit()
    
    def get_similar_articles(self, article_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get articles similar to a given article."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT 
                CASE WHEN article_id_1 = ? THEN article_id_2 ELSE article_id_1 END as similar_article_id,
                similarity_score,
                coengagement_count
            FROM article_similarity
            WHERE article_id_1 = ? OR article_id_2 = ?
            ORDER BY similarity_score DESC
            LIMIT ?
        """, (article_id, article_id, article_id, limit)).fetchall()
        
        return [dict(row) for row in rows]
    
    def get_coengaged_articles(self, article_ids: List[str]) -> Dict[str, float]:
        """
        Get articles co-engaged with a set of articles.
        
        Args:
            article_ids: List of article IDs the user has engaged with
            
        Returns:
            Dict mapping article_id -> aggregated similarity score
        """
        if not article_ids:
            return {}
        
        conn = self._get_conn()
        placeholders = ",".join("?" * len(article_ids))
        
        rows = conn.execute(f"""
            SELECT 
                CASE WHEN article_id_1 IN ({placeholders}) THEN article_id_2 ELSE article_id_1 END as related_article_id,
                SUM(similarity_score * coengagement_count) as agg_score
            FROM article_similarity
            WHERE article_id_1 IN ({placeholders}) OR article_id_2 IN ({placeholders})
            GROUP BY related_article_id
            ORDER BY agg_score DESC
        """, article_ids + article_ids + article_ids).fetchall()
        
        return {row["related_article_id"]: row["agg_score"] for row in rows}
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Singleton instance
_feedback_logger: Optional[FeedbackLogger] = None


def get_feedback_logger(db_path: str = "feedback.db") -> FeedbackLogger:
    """Get or create singleton feedback logger."""
    global _feedback_logger
    if _feedback_logger is None:
        _feedback_logger = FeedbackLogger(db_path)
    return _feedback_logger
