### Running Instructions

```bash
pip install -r requirements.txt
```

To index the articles

```bash
python main.py index --articles articles.jsonl --recreate
```

Run simulator contrainer

```bash
docker load -i ire_project-1.0-amd64.tar
docker run --rm -p 3000:3000 \
  -v $(pwd)/data:/data \
  --tmpfs /tmp:rw,noexec,nosuid \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  ire_project:1.0
```

Start the baseline loop (in another terminal)
```bash
python main.py loop --sim-url http://localhost:3000 --top-k 10 --rounds 5
```

Flags and defaults:

```bash
--host: ES URL (default http://localhost:9200)
--index: index name (default articles)
--articles: path to JSONL (default articles.jsonl)
--recreate: drop & recreate index before ingest (index only)
--sim-url: simulator base URL (default http://localhost:3000)
--top-k: number of results to return to simulator (default 10)
--rounds: number of simulator queries to process (default 5)
```

CPU, with batched CF and progress:
```bash
python main.py train --model gbdt --min-samples 50 --positive-weight 3.0
```
With parallel batched CF (4 threads):
```bash
python main.py train --model gbdt --min-samples 50 --positive-weight 3.0 --cf-workers 4
```
With GPU (if available):
```bash
python main.py train --model gbdt --min-samples 50 --positive-weight 3.0 --gbdt-gpu
```