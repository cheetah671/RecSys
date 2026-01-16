"""
Personalized Search System - Main Entry Point

A modular, extensible framework for personalized news article ranking
with A/B testing support and multithreaded execution.

Usage:
    # Index articles
    python main.py index --articles articles.jsonl
    
    # Run baseline loop (no personalization)
    python main.py loop --reranker baseline --rounds 10
    
    # Run with linear personalization
    python main.py loop --reranker linear --rounds 10
    
    # Run A/B test between baseline and linear
    python main.py loop --ab-test --rounds 100
    
    # Run with custom threading settings
    python main.py loop --ab-test --rounds 1000 --workers 8 --prefetch 20
    
    # Collect baseline data (for batch training)
    python main.py collect --rounds 2000
    
    # Train GBDT model on collected data
    python main.py train --model gbdt --output models/gbdt_model.json
    
    # View experiment results
    python main.py stats --experiment reranker_ab_test
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from es_client import ESConfig, ElasticsearchClient
from feedback_logger import get_feedback_logger, FeedbackLogger
from ab_testing import (
    get_ab_manager,
    ABTestingManager,
    ExperimentConfig,
    DEFAULT_RERANKER_EXPERIMENT
)
from rerankers import (
    BaseReranker,
    RerankerFactory,
    RerankerConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# Environment defaults
SIMULATOR_URL_DEFAULT = os.getenv("SIM_URL", "http://localhost:3000")
ARTICLES_PATH_DEFAULT = os.getenv("ARTICLES_PATH", "articles.jsonl")
MODEL_PATH_DEFAULT = os.getenv("MODEL_PATH", "models/linear_model.json")

# Thread pool settings
DEFAULT_WORKERS = int(os.getenv("WORKERS", "4"))
DEFAULT_PREFETCH = int(os.getenv("PREFETCH_SIZE", "10"))


def create_session_with_retries(
    retries: int = 3,
    backoff_factor: float = 0.3,
    pool_connections: int = 10,
    pool_maxsize: int = 10
) -> requests.Session:
    """Create a requests session with connection pooling and retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class PersonalizedSearchSystem:
    """
    Main system orchestrating search, re-ranking, and learning.
    
    Supports:
    - Multiple reranker strategies via factory pattern
    - A/B testing between strategies
    - Online learning from user feedback
    - Comprehensive logging for offline analysis
    """
    
    def __init__(
        self,
        es_client: ElasticsearchClient,
        feedback_logger: FeedbackLogger,
        ab_manager: Optional[ABTestingManager] = None,
        default_reranker: str = "baseline",
        model_path: Optional[str] = None,
    ):
        self.es_client = es_client
        self.feedback_logger = feedback_logger
        self.ab_manager = ab_manager
        self.default_reranker = default_reranker
        self.model_path = model_path
        
        # Cache of reranker instances
        self._rerankers: Dict[str, BaseReranker] = {}
        
        # Initialize default reranker
        self._get_or_create_reranker(default_reranker)
    
    def _get_or_create_reranker(self, name: str) -> BaseReranker:
        """Get or create a reranker instance by name."""
        if name not in self._rerankers:
            config = RerankerConfig(
                name=name,
                model_path=self.model_path,
                extra_params={"per_user": True, "es_weight": 0.5}
            )
            self._rerankers[name] = RerankerFactory.create(name, config)
            logger.info(f"Created reranker: {name}")
        return self._rerankers[name]
    
    def get_reranker_for_user(
        self,
        user_id: str,
        experiment_id: Optional[str] = None
    ) -> BaseReranker:
        """
        Get the appropriate reranker for a user.
        
        If A/B testing is enabled, uses experiment assignment.
        Otherwise, uses default reranker.
        """
        if self.ab_manager and experiment_id:
            reranker_name = self.ab_manager.get_reranker_for_user(
                user_id, experiment_id
            )
        else:
            reranker_name = self.default_reranker
        
        return self._get_or_create_reranker(reranker_name)
    
    def search_and_rerank(
        self,
        query_text: str,
        user_id: str,
        top_k: int = 10,
        experiment_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """
        Search and re-rank articles for a user query.
        
        Returns:
            Tuple of (reranked_hits, reranker_name, experiment_variant)
        """
        # Get base results from Elasticsearch
        hits = self.es_client.search(query_text=query_text, size=top_k)
        
        if not hits:
            logger.warning(f"No results for query: {query_text}")
            return [], self.default_reranker, None
        
        # Get appropriate reranker
        reranker = self.get_reranker_for_user(user_id, experiment_id)
        
        # Get experiment variant info
        variant = None
        if self.ab_manager and experiment_id:
            variant = self.ab_manager.get_variant(user_id, experiment_id)
        
        # Re-rank results
        reranked = reranker.rerank(
            hits=hits,
            user_id=user_id,
            query_text=query_text,
            context=context
        )
        
        return reranked, reranker.name, variant
    
    def process_feedback(
        self,
        user_id: str,
        query_id: str,
        query_text: str,
        ranked_articles: List[Dict[str, Any]],
        actions: List[List[Any]],
        reranker_name: str,
        experiment_id: Optional[str] = None,
        variant: Optional[str] = None
    ) -> None:
        """
        Process user feedback: log and update model.
        
        Args:
            user_id: User identifier
            query_id: Query identifier
            query_text: Original query text
            ranked_articles: Articles that were shown (in ranked order)
            actions: User actions for each article
            reranker_name: Name of reranker used
            experiment_id: Experiment ID (if in A/B test)
            variant: Variant assignment (if in A/B test)
        """
        reranker = self._get_or_create_reranker(reranker_name)
        
        # Log the session
        self.feedback_logger.log_session(
            user_id=user_id,
            query_id=query_id,
            query_text=query_text,
            ranked_articles=ranked_articles,
            actions=actions,
            compute_reward_fn=reranker.compute_reward,
            experiment_id=experiment_id,
            variant=variant
        )
        
        # Update the model with feedback
        reranker.update(
            user_id=user_id,
            query_text=query_text,
            ranked_articles=ranked_articles,
            actions=actions,
            context={"query_id": query_id}
        )
        
        # Compute session stats
        total_reward = sum(reranker.compute_reward(a) for a in actions)
        clicks = sum(1 for a in actions if "Click" in a)
        
        logger.debug(
            f"Feedback: user={user_id[:8]}... reranker={reranker_name} "
            f"clicks={clicks} reward={total_reward:.2f}"
        )
    
    def save_models(self) -> None:
        """Save all reranker models to disk."""
        for name, reranker in self._rerankers.items():
            if self.model_path:
                # Create model-specific path
                base, ext = os.path.splitext(self.model_path)
                path = f"{base}_{name}{ext}"
                reranker.save(path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "rerankers": {
                name: reranker.get_stats() 
                for name, reranker in self._rerankers.items()
            }
        }
        
        if self.ab_manager:
            for exp_id in self.ab_manager.experiments:
                stats[f"experiment_{exp_id}"] = self.ab_manager.get_experiment_summary(exp_id)
        
        return stats


def call_simulator_query(base_url: str, session: Optional[requests.Session] = None) -> dict:
    """Get a query from the simulator."""
    s = session or requests
    r = s.get(f"{base_url}/query", timeout=10)
    r.raise_for_status()
    return r.json()


def call_simulator_ranklist(
    base_url: str,
    query_id: str,
    user_id: str,
    ranked_article_ids: List[str],
    session: Optional[requests.Session] = None
) -> dict:
    """Submit ranked list to simulator and get feedback."""
    s = session or requests
    payload = {
        "query_id": query_id,
        "user_id": user_id,
        "ranked_article_ids": ranked_article_ids,
    }
    r = s.post(f"{base_url}/ranklist", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def index_articles(
    client: ElasticsearchClient,
    articles_path: str,
    recreate: bool = False
) -> None:
    """Index articles into Elasticsearch."""
    client.ensure_index(recreate=recreate)
    client.bulk_index_jsonl(articles_path)


class QueryPrefetcher:
    """
    Prefetches queries from the simulator in a background thread.
    
    This reduces latency by having queries ready before they're needed.
    """
    
    def __init__(
        self,
        sim_url: str,
        prefetch_size: int = 10,
        session: Optional[requests.Session] = None
    ):
        self.sim_url = sim_url
        self.prefetch_size = prefetch_size
        self.session = session or create_session_with_retries()
        self.queue: Queue = Queue(maxsize=prefetch_size)
        self._stop = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
        self._future = None
    
    def start(self) -> None:
        """Start the prefetching background task."""
        self._stop = False
        self._future = self._executor.submit(self._prefetch_loop)
        logger.debug("Query prefetcher started")
    
    def stop(self) -> None:
        """Stop the prefetching background task."""
        self._stop = True
        if self._future:
            self._future.result(timeout=5)
        logger.debug("Query prefetcher stopped")
    
    def _prefetch_loop(self) -> None:
        """Background loop that keeps the queue filled."""
        while not self._stop:
            if self.queue.qsize() < self.prefetch_size:
                try:
                    query = call_simulator_query(self.sim_url, self.session)
                    self.queue.put(query, timeout=1)
                except Exception as e:
                    logger.debug(f"Prefetch error: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.05)
    
    def get_query(self, timeout: float = 10) -> Optional[dict]:
        """Get a prefetched query, or fetch one directly if queue is empty."""
        try:
            return self.queue.get(timeout=timeout)
        except Exception:
            # Fallback to direct fetch
            return call_simulator_query(self.sim_url, self.session)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
        self._executor.shutdown(wait=False)


def run_loop(
    system: PersonalizedSearchSystem,
    sim_url: str,
    top_k: int = 10,
    rounds: int = 5,
    sleep_secs: float = 0.5,
    ab_test: bool = False,
    experiment_id: str = "reranker_ab_test",
    save_interval: int = 50,
    num_workers: int = DEFAULT_WORKERS,
    prefetch_size: int = DEFAULT_PREFETCH,
) -> None:
    """
    Main loop: query simulator, rank, get feedback, learn.
    
    Uses multithreading for improved performance:
    - Query prefetching in background
    - Parallel processing of search and feedback
    - Connection pooling for HTTP requests
    
    Args:
        system: Personalized search system instance
        sim_url: Simulator base URL
        top_k: Number of results to retrieve
        rounds: Number of query rounds
        sleep_secs: Delay between rounds (reduced when using prefetch)
        ab_test: Whether to run A/B testing
        experiment_id: Experiment ID for A/B testing
        save_interval: How often to save models
        num_workers: Number of worker threads for parallel processing
        prefetch_size: Number of queries to prefetch
    """
    logger.info(f"Starting loop: rounds={rounds}, ab_test={ab_test}, workers={num_workers}")
    
    # Create session with connection pooling
    session = create_session_with_retries(pool_connections=num_workers, pool_maxsize=num_workers * 2)
    
    # Thread-safe stats
    stats_lock = Lock()
    stats = {
        "total_clicks": 0,
        "total_reward": 0.0,
        "queries_by_variant": {},
        "completed": 0,
        "errors": 0,
    }
    
    def process_single_round(
        round_num: int,
        query: dict
    ) -> Tuple[int, float, str, Optional[str]]:
        """Process a single round. Returns (clicks, reward, variant_key, error_msg)."""
        try:
            user_id = query["user_id"]
            query_id = query["query_id"]
            query_text = query["query_text"]
            
            logger.info(f"Round {round_num}/{rounds}: user={user_id[:8]}... query='{query_text}'")
            
            # Search and re-rank
            exp_id = experiment_id if ab_test else None
            hits, reranker_name, variant = system.search_and_rerank(
                query_text=query_text,
                user_id=user_id,
                top_k=top_k,
                experiment_id=exp_id,
                context={"query_id": query_id}
            )
            
            if not hits:
                logger.warning("No results found, skipping round")
                return 0, 0.0, reranker_name, None
            
            # Submit to simulator with simple retries
            ranked_ids = [h["_id"] for h in hits]
            resp = None
            for attempt in range(3):
                try:
                    resp = call_simulator_ranklist(sim_url, query_id, user_id, ranked_ids, session)
                    break
                except Exception as se:
                    logger.warning(f"Simulator call failed (attempt {attempt+1}/3) for round {round_num}: {se}")
                    time.sleep(0.5 * (attempt + 1))
            if resp is None:
                raise RuntimeError("Simulator call failed after 3 attempts")
            actions = resp.get("actions", [])
            
            # Process feedback (thread-safe due to per-user model updates)
            system.process_feedback(
                user_id=user_id,
                query_id=query_id,
                query_text=query_text,
                ranked_articles=hits,
                actions=actions,
                reranker_name=reranker_name,
                experiment_id=exp_id,
                variant=variant
            )
            
            # Calculate stats
            reranker = system._get_or_create_reranker(reranker_name)
            round_reward = sum(reranker.compute_reward(a) for a in actions)
            round_clicks = sum(1 for a in actions if any(act == "Click" for act in a))
            
            variant_key = variant or reranker_name
            
            # Log progress
            if round_clicks > 0:
                logger.info(
                    f"  [{reranker_name}] clicks={round_clicks} reward={round_reward:.2f} "
                    f"actions={_summarize_actions(actions)}"
                )
            
            return round_clicks, round_reward, variant_key, None
            
        except Exception as e:
            logger.error(f"Error in round {round_num}: {e}", exc_info=True)
            return 0, 0.0, "error", str(e)
    
    # Use prefetcher for queries
    with QueryPrefetcher(sim_url, prefetch_size=prefetch_size, session=session) as prefetcher:
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="worker") as executor:
            futures = []
            
            for i in range(rounds):
                try:
                    # Get prefetched query (or fetch directly)
                    query = prefetcher.get_query(timeout=10)
                    if query is None:
                        logger.warning(f"Could not get query for round {i+1}")
                        continue
                    
                    # Submit to thread pool
                    future = executor.submit(process_single_round, i + 1, query)
                    futures.append((i + 1, future))
                    
                    # Small delay to avoid overwhelming the simulator
                    if sleep_secs > 0 and len(futures) % num_workers == 0:
                        time.sleep(sleep_secs / num_workers)
                    
                except Exception as e:
                    logger.error(f"Error submitting round {i+1}: {e}")
            
            # Collect results as they complete (robust to slow/failed workers)
            for round_num, future in futures:
                try:
                    # Allow more time for slower simulator responses
                    clicks, reward, variant_key, error = future.result(timeout=180)
                    
                    with stats_lock:
                        if error:
                            stats["errors"] += 1
                        else:
                            stats["completed"] += 1
                            stats["total_clicks"] += clicks
                            stats["total_reward"] += reward
                            
                            if variant_key not in stats["queries_by_variant"]:
                                stats["queries_by_variant"][variant_key] = {
                                    "queries": 0, "clicks": 0, "reward": 0.0
                                }
                            stats["queries_by_variant"][variant_key]["queries"] += 1
                            stats["queries_by_variant"][variant_key]["clicks"] += clicks
                            stats["queries_by_variant"][variant_key]["reward"] += reward
                        
                        # Periodic save
                        if stats["completed"] % save_interval == 0 and stats["completed"] > 0:
                            system.save_models()
                            logger.info(f"Saved models at {stats['completed']} completed rounds")
                    
                except TimeoutError:
                    # Cancel stalled task and continue
                    logger.error(f"Timeout collecting result for round {round_num}; cancelling task")
                    try:
                        future.cancel()
                    except Exception:
                        pass
                    with stats_lock:
                        stats["errors"] += 1
                except Exception as e:
                    logger.error(f"Error collecting result for round {round_num}: {e}")
                    with stats_lock:
                        stats["errors"] += 1
    
    # Final save and stats
    system.save_models()
    session.close()
    _print_final_stats(stats, rounds)


def _summarize_actions(actions: List[List[Any]]) -> str:
    """Create a brief summary of actions."""
    clicks = sum(1 for a in actions if "Click" in a)
    likes = sum(1 for a in actions if "Like" in a)
    shares = sum(1 for a in actions if "Share" in a)
    bookmarks = sum(1 for a in actions if "Bookmark" in a)
    return f"C={clicks}/L={likes}/S={shares}/B={bookmarks}"


def _print_final_stats(stats: Dict[str, Any], rounds: int) -> None:
    """Print final statistics."""
    completed = stats.get("completed", rounds)
    errors = stats.get("errors", 0)
    
    logger.info("=" * 60)
    logger.info("FINAL STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total rounds: {rounds}")
    logger.info(f"Completed: {completed}, Errors: {errors}")
    logger.info(f"Total clicks: {stats['total_clicks']}")
    logger.info(f"Total reward: {stats['total_reward']:.2f}")
    logger.info(f"Avg reward/round: {stats['total_reward']/max(completed,1):.2f}")
    
    if stats["queries_by_variant"]:
        logger.info("\nPer-variant breakdown:")
        for variant, vs in stats["queries_by_variant"].items():
            logger.info(
                f"  {variant}: queries={vs['queries']} clicks={vs['clicks']} "
                f"reward={vs['reward']:.2f} avg={vs['reward']/max(vs['queries'],1):.2f}"
            )


def show_stats(
    feedback_logger: FeedbackLogger,
    ab_manager: ABTestingManager,
    experiment_id: Optional[str] = None
) -> None:
    """Show statistics from logged data."""
    print("\n" + "=" * 60)
    print("EXPERIMENT STATISTICS")
    print("=" * 60)
    
    if experiment_id:
        results = feedback_logger.get_experiment_results(experiment_id)
        if results:
            print(f"\nExperiment: {experiment_id}")
            for r in results:
                print(f"\n  Variant: {r['variant']}")
                print(f"    Total queries: {r['total_queries']}")
                print(f"    Total clicks: {r['total_clicks']}")
                print(f"    Total reward: {r['total_reward']:.2f}")
                print(f"    Click rate: {r['total_clicks']/max(r['total_queries'],1):.2%}")
                print(f"    Avg reward: {r['total_reward']/max(r['total_queries'],1):.2f}")
                if r['avg_position_clicked']:
                    print(f"    Avg click position: {r['avg_position_clicked']:.2f}")
        else:
            print(f"No data for experiment: {experiment_id}")
    
    # Show A/B manager summary
    for exp_id in ab_manager.experiments:
        summary = ab_manager.get_experiment_summary(exp_id)
        print(f"\nA/B Test Status: {exp_id}")
        print(f"  Enabled: {summary['enabled']}")
        print(f"  Total users: {summary['total_users']}")
        print(f"  Variants: {summary['variants']}")
        print(f"  Distribution: {summary['variant_distribution']}")


def train_model(
    feedback_logger: FeedbackLogger,
    model_type: str = "gbdt",
    output_path: str = "models/gbdt_model.json",
    analysis_path: str = "models/feature_analysis.json",
    min_interactions: int = 100,
    negative_subsample: float = 0.3,
    min_positive_samples: int = 50,
    positive_weight: float = 3.0,
    precompute_cf: bool = True,
    use_gpu_for_gbdt: bool = False,
    cf_batch_workers: int = 0,
    show_progress: bool = True,
) -> None:
    """
    Train a ranking model on collected interaction data.
    
    This implements the batch training pipeline:
    1. Load data from feedback.db
    2. Handle sparse feedback (actions are rare)
    3. Extract and analyze features
    4. Train model (GBDT or Linear)
    5. Optionally precompute CF similarities
    """
    from batch_trainer import BatchTrainer, TrainingConfig
    
    print("\n" + "=" * 60)
    print("BATCH TRAINING PIPELINE")
    print("=" * 60)
    
    # Create training config
    config = TrainingConfig(
        min_interactions=min_interactions,
        negative_subsample_rate=negative_subsample,
        min_positive_samples=min_positive_samples,
        positive_sample_weight=positive_weight,
        precompute_similarities=precompute_cf,
        use_gpu_for_gbdt=use_gpu_for_gbdt,
        cf_batch_workers=cf_batch_workers,
        show_progress=show_progress,
    )
    
    # Create trainer
    trainer = BatchTrainer(feedback_logger=feedback_logger, config=config)
    
    # Run full pipeline
    results = trainer.run_full_pipeline(
        model_path=output_path,
        analysis_path=analysis_path
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return
    
    print(f"\nData Statistics:")
    if "data_stats" in results:
        stats = results["data_stats"]
        print(f"  Total interactions: {stats.get('total_interactions', 0)}")
        print(f"  Positive interactions: {stats.get('positive_interactions', 0)}")
        print(f"  Action rate: {stats.get('action_rate', 0):.2%}")
        print(f"  Unique users: {stats.get('unique_users', 0)}")
        print(f"  Unique articles: {stats.get('unique_articles', 0)}")
    
    print(f"\nTraining Results:")
    if "training" in results:
        train = results["training"]
        print(f"  Samples used: {train.get('samples', 0)}")
        print(f"  Train/Val split: {train.get('train_samples', 0)}/{train.get('val_samples', 0)}")
        if "best_iteration" in train:
            print(f"  Best iteration: {train['best_iteration']}")
        if "feature_importance" in train:
            print(f"\n  Top 5 Features:")
            sorted_features = sorted(
                train["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for name, importance in sorted_features:
                print(f"    {name}: {importance:.2f}")
    
    if "cf_pairs_computed" in results:
        print(f"\nCF Similarities: {results['cf_pairs_computed']} pairs computed")
    
    print(f"\nModel saved to: {output_path}")
    print(f"Analysis saved to: {analysis_path}")


def collect_data(
    system: PersonalizedSearchSystem,
    sim_url: str,
    rounds: int = 1000,
    sleep_secs: float = 0.5,
    num_workers: int = DEFAULT_WORKERS,
    prefetch_size: int = DEFAULT_PREFETCH,
) -> None:
    """
    Collect interaction data using baseline ranking (for batch training).
    
    This is Phase 1 of the batch training pipeline:
    - Uses baseline (BM25) ranking to avoid bias
    - Logs all interactions to feedback.db
    - Does NOT update any models during collection
    """
    print("\n" + "=" * 60)
    print("DATA COLLECTION PHASE")
    print("=" * 60)
    print(f"Collecting {rounds} rounds of interaction data using baseline ranking")
    print("This data will be used for batch training.")
    print("=" * 60 + "\n")
    
    # Override reranker to baseline for unbiased data collection
    original_reranker = system.default_reranker
    system.default_reranker = "baseline"
    
    # Disable model updates during collection
    baseline_reranker = system._get_or_create_reranker("baseline")
    
    # Run collection loop (same as regular loop but with baseline)
    run_loop(
        system=system,
        sim_url=sim_url,
        top_k=10,
        rounds=rounds,
        sleep_secs=sleep_secs,
        ab_test=False,  # No A/B testing during collection
        save_interval=rounds + 1,  # Don't save during collection
        num_workers=num_workers,
        prefetch_size=prefetch_size,
    )
    
    # Restore original reranker
    system.default_reranker = original_reranker
    
    # Print collection stats
    stats = system.feedback_logger.get_training_data(limit=1)
    total = len(system.feedback_logger.get_training_data(limit=rounds * 10))
    
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total interactions logged: {total}")
    print("Run 'python main.py train' to train a model on this data.")


def collect_data_current(
    system: PersonalizedSearchSystem,
    sim_url: str,
    rounds: int = 1000,
    sleep_secs: float = 0.5,
    num_workers: int = DEFAULT_WORKERS,
    prefetch_size: int = DEFAULT_PREFETCH,
) -> None:
    """
    Collect interaction data using the system's current reranker (for iterative training).
    """
    print("\n" + "=" * 60)
    print("DATA COLLECTION (CURRENT RERANKER)")
    print("=" * 60)
    print(f"Collecting {rounds} rounds of interaction data using reranker: {system.default_reranker}")
    print("This data will be used for iterative training.")
    print("=" * 60 + "\n")

    run_loop(
        system=system,
        sim_url=sim_url,
        top_k=10,
        rounds=rounds,
        sleep_secs=sleep_secs,
        ab_test=False,
        experiment_id="",
        save_interval=rounds + 1,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Personalized Search System with A/B Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py index --articles articles.jsonl --recreate
  python main.py loop --reranker linear --rounds 100
  python main.py loop --ab-test --rounds 500
  python main.py collect --rounds 2000
  python main.py train --model gbdt --output models/gbdt_model.json
  python main.py stats --experiment reranker_ab_test
        """
    )
    
    p.add_argument(
        "command",
        choices=["index", "loop", "stats", "collect", "collect-current", "train"],
        help="Action to perform"
    )
    
    # Elasticsearch options
    p.add_argument(
        "--host", dest="es_host",
        default=os.getenv("ES_HOST", "http://localhost:9200"),
        help="Elasticsearch host URL"
    )
    p.add_argument(
        "--index", dest="es_index",
        default=os.getenv("ES_INDEX", "articles"),
        help="Elasticsearch index name"
    )
    
    # Data options
    p.add_argument(
        "--articles", dest="articles_path",
        default=ARTICLES_PATH_DEFAULT,
        help="Path to articles.jsonl file"
    )
    p.add_argument(
        "--recreate", action="store_true",
        help="Recreate index before indexing"
    )
    
    # Simulator options
    p.add_argument(
        "--sim-url", dest="sim_url",
        default=SIMULATOR_URL_DEFAULT,
        help="Simulator base URL"
    )
    
    # Reranker options
    p.add_argument(
        "--reranker", dest="reranker",
        default="baseline",
        choices=RerankerFactory.list_available() or ["baseline", "linear"],
        help="Reranker to use (if not using A/B testing)"
    )
    p.add_argument(
        "--model-path", dest="model_path",
        default=MODEL_PATH_DEFAULT,
        help="Path to save/load model"
    )
    
    # A/B testing options
    p.add_argument(
        "--ab-test", dest="ab_test",
        action="store_true",
        help="Enable A/B testing between rerankers"
    )
    p.add_argument(
        "--experiment", dest="experiment_id",
        default="reranker_ab_test",
        help="Experiment ID for A/B testing"
    )
    
    # Loop options
    p.add_argument(
        "--top-k", dest="top_k",
        type=int, default=10,
        help="Number of results to retrieve"
    )
    p.add_argument(
        "--rounds", dest="rounds",
        type=int, default=5,
        help="Number of query rounds"
    )
    p.add_argument(
        "--sleep", dest="sleep_secs",
        type=float, default=0.5,
        help="Delay between rounds (seconds)"
    )
    p.add_argument(
        "--save-interval", dest="save_interval",
        type=int, default=50,
        help="Save models every N rounds"
    )
    
    # Threading options
    p.add_argument(
        "--workers", dest="num_workers",
        type=int, default=DEFAULT_WORKERS,
        help=f"Number of worker threads (default: {DEFAULT_WORKERS})"
    )
    p.add_argument(
        "--prefetch", dest="prefetch_size",
        type=int, default=DEFAULT_PREFETCH,
        help=f"Number of queries to prefetch (default: {DEFAULT_PREFETCH})"
    )
    
    # Logging options
    p.add_argument(
        "--feedback-db", dest="feedback_db",
        default="feedback.db",
        help="Path to feedback database"
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Training options (for 'train' command)
    p.add_argument(
        "--model", dest="train_model",
        default="gbdt",
        choices=["gbdt", "linear"],
        help="Model type to train (for train command)"
    )
    p.add_argument(
        "--output", dest="output_path",
        default="models/gbdt_model.json",
        help="Output path for trained model"
    )
    p.add_argument(
        "--analysis-output", dest="analysis_path",
        default="models/feature_analysis.json",
        help="Output path for feature analysis"
    )
    p.add_argument(
        "--min-interactions", dest="min_interactions",
        type=int, default=100,
        help="Minimum interactions required for training"
    )
    p.add_argument(
        "--negative-subsample", dest="negative_subsample",
        type=float, default=0.3,
        help="Fraction of negative samples to keep (for sparse data)"
    )
    p.add_argument(
        "--min-samples", dest="min_positive_samples",
        type=int, default=50,
        help="Minimum positive samples (with actions) required for training"
    )
    p.add_argument(
        "--positive-weight", dest="positive_weight",
        type=float, default=3.0,
        help="Weight multiplier for positive samples (handles sparse data)"
    )
    p.add_argument(
        "--precompute-cf", dest="precompute_cf",
        action="store_true",
        help="Precompute CF similarities during training"
    )

    # Performance options (train)
    p.add_argument(
        "--gbdt-gpu",
        dest="use_gpu_gbdt",
        action="store_true",
        help="Use GPU for LightGBM training (requires GPU-enabled LightGBM)"
    )
    p.add_argument(
        "--cf-workers",
        dest="cf_workers",
        type=int,
        default=0,
        help="Parallel workers for batched CF scoring during feature extraction (0 disables)"
    )
    p.add_argument(
        "--no-progress",
        dest="no_progress",
        action="store_true",
        help="Disable progress bars during feature extraction"
    )
    
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize Elasticsearch client
    config = ESConfig(host=args.es_host, index_name=args.es_index)
    es_client = ElasticsearchClient(config=config)
    
    # Initialize feedback logger
    feedback_logger = get_feedback_logger(args.feedback_db)
    
    # Initialize A/B testing manager
    ab_manager = get_ab_manager()
    
    try:
        if args.command == "index":
            index_articles(es_client, args.articles_path, recreate=args.recreate)
            
        elif args.command == "loop":
            # Create personalized search system
            system = PersonalizedSearchSystem(
                es_client=es_client,
                feedback_logger=feedback_logger,
                ab_manager=ab_manager if args.ab_test else None,
                default_reranker=args.reranker,
                model_path=args.model_path,
            )
            
            run_loop(
                system=system,
                sim_url=args.sim_url,
                top_k=args.top_k,
                rounds=args.rounds,
                sleep_secs=args.sleep_secs,
                ab_test=args.ab_test,
                experiment_id=args.experiment_id,
                save_interval=args.save_interval,
                num_workers=args.num_workers,
                prefetch_size=args.prefetch_size,
            )
        
        elif args.command == "collect":
            # Collect data using baseline for batch training
            system = PersonalizedSearchSystem(
                es_client=es_client,
                feedback_logger=feedback_logger,
                ab_manager=None,
                default_reranker="baseline",
                model_path=None,
            )
            
            collect_data(
                system=system,
                sim_url=args.sim_url,
                rounds=args.rounds,
                sleep_secs=args.sleep_secs,
                num_workers=args.num_workers,
                prefetch_size=args.prefetch_size,
            )
        elif args.command == "collect-current":
            # Collect data using the selected reranker (for iterative training)
            system = PersonalizedSearchSystem(
                es_client=es_client,
                feedback_logger=feedback_logger,
                ab_manager=None,
                default_reranker=args.reranker,
                model_path=args.model_path,
            )

            collect_data_current(
                system=system,
                sim_url=args.sim_url,
                rounds=args.rounds,
                sleep_secs=args.sleep_secs,
                num_workers=args.num_workers,
                prefetch_size=args.prefetch_size,
            )
        
        elif args.command == "train":
            # Batch train a model on collected data
            train_model(
                feedback_logger=feedback_logger,
                model_type=args.train_model,
                output_path=args.output_path,
                analysis_path=args.analysis_path,
                min_interactions=args.min_interactions,
                negative_subsample=args.negative_subsample,
                min_positive_samples=args.min_positive_samples,
                positive_weight=args.positive_weight,
                precompute_cf=args.precompute_cf,
                use_gpu_for_gbdt=args.use_gpu_gbdt,
                cf_batch_workers=args.cf_workers,
                show_progress=not args.no_progress,
            )
            
        elif args.command == "stats":
            show_stats(feedback_logger, ab_manager, args.experiment_id)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except requests.RequestException as e:
        logger.error(f"Request error: {e}")
        sys.exit(1)
    finally:
        es_client.close()
        feedback_logger.close()


if __name__ == "__main__":
    main()
