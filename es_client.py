import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class ESConfig:
    def __init__(
        self,
        host: Optional[str] = None,
        index_name: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_ssl: Optional[bool] = None,
    ) -> None:
        self.host = host or os.getenv("ES_HOST", "http://localhost:9200")
        self.index_name = index_name or os.getenv("ES_INDEX", "articles")
        self.username = username or os.getenv("ES_USERNAME")
        self.password = password or os.getenv("ES_PASSWORD")
        v = os.getenv("ES_VERIFY_SSL")
        self.verify_ssl = (
            verify_ssl if verify_ssl is not None else (False if v == "false" else True)
        )


class ElasticsearchClient:
    def __init__(self, config: Optional[ESConfig] = None) -> None:
        self.config = config or ESConfig()
        self.session = requests.Session()
        if self.config.username and self.config.password:
            self.session.auth = (self.config.username, self.config.password)
        self.verify = self.config.verify_ssl
        self.base = self.config.host.rstrip("/")
        logger.info(f"Using ES REST endpoint at {self.base}")

    def ensure_index(self, recreate: bool = False) -> None:
        index = self.config.index_name
        # HEAD /{index}
        resp = self.session.head(f"{self.base}/{index}", verify=self.verify)
        exists = resp.status_code == 200
        if exists and recreate:
            logger.info(f"Deleting existing index '{index}'")
            del_resp = self.session.delete(f"{self.base}/{index}", verify=self.verify)
            del_resp.raise_for_status()
            exists = False
        if not exists:
            logger.info(f"Creating index '{index}'")
            body = {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": "standard",
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "uuid": {"type": "keyword"},
                        "text": {"type": "text", "analyzer": "standard"},
                        "topics": {"type": "keyword"},
                    }
                },
            }
            create_resp = self.session.put(
                f"{self.base}/{index}",
                json=body,
                verify=self.verify,
            )
            create_resp.raise_for_status()

    def _iter_jsonl(self, path: str) -> Iterable[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line")
                    continue
                yield obj

    def bulk_index_jsonl(self, path: str, id_field: str = "uuid", batch_size: int = 2000) -> Tuple[int, int]:
            index = self.config.index_name
            total, failed = 0, 0

            def actions() -> Iterable[Dict[str, Any]]:
                for doc in self._iter_jsonl(path):
                    # 1. Explicitly use 'uuid' from your sample data
                    doc_id = doc.get("uuid")
                    
                    if not doc_id:
                        logger.warning("Document missing 'uuid'; skipping")
                        nonlocal failed
                        failed += 1
                        continue
                    
                    # 2. Handle 'topics'. Your sample has ["science and technology", "lifestyle and leisure"]
                    topics = doc.get("topics", [])
                    # Ensure topics is a clean list of strings
                    if isinstance(topics, list):
                        topics = [str(t) for t in topics]
                    else:
                        topics = []

                    # 3. Use the 'text' field directly as it contains the full article
                    # We strip() to remove leading/trailing newlines present in your sample
                    text_content = doc.get("text", "").strip()

                    source = {
                        "uuid": str(doc_id),
                        "text": text_content,
                        "topics": topics,
                    }
                    
                    yield {
                        "_index": index,
                        "_id": str(doc_id),
                        "_source": source,
                    }

            logger.info(f"Bulk indexing from {path} into '{index}'")
            
            # ... (The rest of the function involving the _bulk API loop remains exactly the same) ...
            bulk_lines: List[str] = []
            count_in_batch = 0
            for action in actions():
                meta = {"index": {"_index": action["_index"], "_id": action["_id"]}}
                bulk_lines.append(json.dumps(meta))
                bulk_lines.append(json.dumps(action["_source"]))
                count_in_batch += 1
                if count_in_batch >= batch_size:
                    data = "\n".join(bulk_lines) + "\n"
                    resp = self.session.post(
                        f"{self.base}/_bulk",
                        data=data,
                        headers={"Content-Type": "application/x-ndjson"},
                        verify=self.verify,
                    )
                    resp.raise_for_status()
                    res = resp.json()
                    items = res.get("items", [])
                    for item in items:
                        total += 1
                        if item.get("index", {}).get("error"):
                            failed += 1
                    bulk_lines = []
                    count_in_batch = 0

            if bulk_lines:
                data = "\n".join(bulk_lines) + "\n"
                resp = self.session.post(
                    f"{self.base}/_bulk",
                    data=data,
                    headers={"Content-Type": "application/x-ndjson"},
                    verify=self.verify,
                )
                resp.raise_for_status()
                res = resp.json()
                items = res.get("items", [])
                for item in items:
                    total += 1
                    if item.get("index", {}).get("error"):
                        failed += 1

            logger.info(f"Indexed {total - failed}/{total} documents (failed={failed})")
            return total, failed

    def search(self, query_text: str, size: int = 10) -> List[Dict[str, Any]]:
        index = self.config.index_name
        body = {
            "query": {"match": {"text": {"query": query_text}}},
            "size": size,
        }
        resp = self.session.get(
            f"{self.base}/{index}/_search",
            json=body,
            verify=self.verify,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("hits", {}).get("hits", [])

    def close(self) -> None:
        # Nothing to close for requests.Session beyond letting GC handle it.
        try:
            self.session.close()
        except Exception:
            pass

    # Placeholder for future personalized re-ranking inside the client
    def rerank(self, hits: List[Dict[str, Any]], user_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Hook to personalize ranking based on user/context.
        For baseline, return ES order unchanged.
        """
        return hits
