#!/usr/bin/env bash
set -euo pipefail

# Run A/B tests on available experiments and save results
# Experiments included:
# 1) reranker_ab_test_gbdt: baseline vs gbdt
# 2) reranker_ab_test: baseline vs linear vs hybrid
# If you need additional combos (e.g., hybrid vs gbdt), register an experiment in ab_testing.py.

SIM_URL=${SIM_URL:-http://localhost:3000}
ES_HOST=${ES_HOST:-http://localhost:9200}
ES_INDEX=${ES_INDEX:-articles}
FEEDBACK_DB=${FEEDBACK_DB:-feedback.db}
MODEL_GBDT=${MODEL_GBDT:-models/gbdt_model.json}
MODEL_LINEAR=${MODEL_LINEAR:-models/linear_model.json}
RESULTS_FILE=${RESULTS_FILE:-ab_test_results.txt}
WORKERS=${WORKERS:-8}
PREFETCH=${PREFETCH:-20}
ROUNDS=${ROUNDS:-1000}

# 1) baseline vs gbdt
python main.py loop \
  --ab-test \
  --experiment reranker_ab_test_gbdt \
  --model-path "$MODEL_GBDT" \
  --rounds "$ROUNDS" \
  --sim-url "$SIM_URL" \
  --host "$ES_HOST" \
  --index "$ES_INDEX" \
  --feedback-db "$FEEDBACK_DB" \
  --workers "$WORKERS" \
  --prefetch "$PREFETCH"

python main.py stats --experiment reranker_ab_test_gbdt > "$RESULTS_FILE"

# 2) baseline vs linear vs hybrid
python main.py loop \
  --ab-test \
  --experiment reranker_ab_test \
  --model-path "$MODEL_LINEAR" \
  --rounds "$ROUNDS" \
  --sim-url "$SIM_URL" \
  --host "$ES_HOST" \
  --index "$ES_INDEX" \
  --feedback-db "$FEEDBACK_DB" \
  --workers "$WORKERS" \
  --prefetch "$PREFETCH"

python main.py stats --experiment reranker_ab_test >> "$RESULTS_FILE"

echo "Results saved to $RESULTS_FILE"