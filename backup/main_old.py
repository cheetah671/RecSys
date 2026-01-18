import argparse
import logging
import os
import time
from typing import List

import requests

from es_client import ESConfig, ElasticsearchClient

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


SIMULATOR_URL_DEFAULT = os.getenv("SIM_URL", "http://localhost:3000")
ARTICLES_PATH_DEFAULT = os.getenv("ARTICLES_PATH", "articles.jsonl")


def call_simulator_query(base_url: str) -> dict:
    r = requests.get(f"{base_url}/query", timeout=10)
    r.raise_for_status()
    return r.json()


def call_simulator_ranklist(base_url: str, query_id: str, user_id: str, ranked_article_ids: List[str]) -> dict:
    payload = {
        "query_id": query_id,
        "user_id": user_id,
        "ranked_article_ids": ranked_article_ids,
    }
    r = requests.post(f"{base_url}/ranklist", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def index_articles(client: ElasticsearchClient, articles_path: str, recreate: bool = False) -> None:
    client.ensure_index(recreate=recreate)
    client.bulk_index_jsonl(articles_path)


def run_baseline_loop(client: ElasticsearchClient, base_url: str, top_k: int = 10, rounds: int = 5, sleep_secs: float = 0.5) -> None:
    for i in range(rounds):
        q = call_simulator_query(base_url)
        user_id = q["user_id"]
        query_id = q["query_id"]
        query_text = q["query_text"]
        logger.info(f"Round {i+1}: user={user_id} query='{query_text}'")

        hits = client.search(query_text=query_text, size=top_k)
        
        # # --- DEBUG START: Inspect what we are actually finding ---
        # if not hits:
        #     logger.warning("  > 0 hits found! Check your index content.")
        # else:
        #     logger.info(f"  > Found {len(hits)} hits. Top 3 snippets:")
        #     for idx, h in enumerate(hits[:3]):
        #         score = h.get("_score")
        #         doc_id = h.get("_id")
        #         # Grab the text, slice first 100 chars, remove newlines for cleaner logs
        #         snippet = h.get("_source", {}).get("text", "")[:100].replace("\n", " ")
        #         logger.info(f"    #{idx+1} [Score: {score:.2f}] ID: {doc_id} | Text: {snippet}...")
        # # --- DEBUG END ---

        hits = client.rerank(hits, user_id=user_id, context={"query_id": query_id, "query_text": query_text})
        ranked_ids = [h["_id"] for h in hits]
        
        logger.info(f"Submitting {len(ranked_ids)} results to simulator")
        resp = call_simulator_ranklist(base_url, query_id, user_id, ranked_ids)
        actions = resp.get("actions", [])
        logger.info(f"Simulator actions: {actions}")
        time.sleep(sleep_secs)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline BM25 indexing and simulator loop")
    p.add_argument("command", choices=["index", "loop"], help="Action to perform")
    p.add_argument("--host", dest="es_host", default=os.getenv("ES_HOST", "http://localhost:9200"))
    p.add_argument("--index", dest="es_index", default=os.getenv("ES_INDEX", "articles"))
    p.add_argument("--articles", dest="articles_path", default=ARTICLES_PATH_DEFAULT)
    p.add_argument("--sim-url", dest="sim_url", default=SIMULATOR_URL_DEFAULT)
    p.add_argument("--recreate", action="store_true", help="Recreate index before ingest")
    p.add_argument("--top-k", dest="top_k", type=int, default=10)
    p.add_argument("--rounds", dest="rounds", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = ESConfig(host=args.es_host, index_name=args.es_index)
    client = ElasticsearchClient(config=config)

    try:
        if args.command == "index":
            index_articles(client, args.articles_path, recreate=args.recreate)
        elif args.command == "loop":
            run_baseline_loop(client, base_url=args.sim_url, top_k=args.top_k, rounds=args.rounds)
    except requests.RequestException as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
