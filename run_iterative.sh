#!/usr/bin/env bash
set -euo pipefail

# Iterative training + data collection + A/B test runner
# Steps:
# 1) Train initial GBDT (uses cached features if present)
# 2) Collect 1,000 rounds using GBDT reranker
# 3) Retrain GBDT (ensure feature cache is refreshed)
# 4) Repeat until 15,000 rounds collected via GBDT
# 5) Collect iteratively with Linear (online updates) until 15,000 rounds
# 6) Run A/B tests: baseline vs gbdt, baseline vs linear, linear vs gbdt
# 7) Save stats outputs to results file

SIM_URL=${SIM_URL:-http://localhost:3000}
ES_HOST=${ES_HOST:-http://localhost:9200}
ES_INDEX=${ES_INDEX:-articles}
FEEDBACK_DB=${FEEDBACK_DB:-feedback.db}
MODEL_PATH=${MODEL_PATH:-models/gbdt_model.json}
ROUNDS_PER_ITER=${ROUNDS_PER_ITER:-100}
TOTAL_TARGET=${TOTAL_TARGET:-1000}
WORKERS=${WORKERS:-8}
PREFETCH=${PREFETCH_SIZE:-20}
USE_GPU=${USE_GPU_GBDT:-1}

RESULTS_FILE=${RESULTS_FILE:-ab_results.txt}
echo "Starting iterative training pipeline (ROUNDS_PER_ITER=$ROUNDS_PER_ITER, TOTAL_TARGET=$TOTAL_TARGET)" | tee "$RESULTS_FILE"

# 1) Initial train
python main.py train \
  --model gbdt \
  --output "$MODEL_PATH" \
  --analysis-output models/feature_analysis.json \
  --min-interactions 100 \
  --negative-subsample 0.3 \
  --min-samples 50 \
  --positive-weight 3.0 \
  --precompute-cf \
  $( [ "$USE_GPU" = "1" ] && echo "--gbdt-gpu" ) | tee -a "$RESULTS_FILE"

# Iterative loop: collect + retrain until TOTAL_TARGET
collected=0
while [ $collected -lt $TOTAL_TARGET ]; do
  echo "\n=== Iteration: collected=$collected / target=$TOTAL_TARGET ==="
  # 2) Collect using current GBDT model
  python main.py collect-model \
    --sim-url "$SIM_URL" \
    --reranker gbdt \
    --model-path "$MODEL_PATH" \
    --rounds $ROUNDS_PER_ITER \
    --workers $WORKERS \
    --prefetch $PREFETCH | tee -a "$RESULTS_FILE"

  collected=$((collected + ROUNDS_PER_ITER))

  # 3) Retrain GBDT on accumulated data
  # Refresh feature cache to ensure new data is used
  rm -f models/prepared_features.jsonl.gz || true
  python main.py train \
    --model gbdt \
    --output "$MODEL_PATH" \
    --analysis-output models/feature_analysis.json \
    --min-interactions 100 \
    --negative-subsample 0.3 \
    --min-samples 50 \
    --positive-weight 3.0 \
    --precompute-cf \
    $( [ "$USE_GPU" = "1" ] && echo "--gbdt-gpu" ) | tee -a "$RESULTS_FILE"

  echo "=== Iteration complete ===\n"

done

# Linear iterative collection (online updates happen during collection)
echo "\nStarting iterative collection with LINEAR reranker" | tee -a "$RESULTS_FILE"
collected_linear=0
LINEAR_MODEL_PATH=${LINEAR_MODEL_PATH:-models/linear_model.json}
while [ $collected_linear -lt $TOTAL_TARGET ]; do
  echo "\n=== Linear Iteration: collected=$collected_linear / target=$TOTAL_TARGET ===" | tee -a "$RESULTS_FILE"
  python main.py collect-model \
    --sim-url "$SIM_URL" \
    --reranker linear \
    --model-path "$LINEAR_MODEL_PATH" \
    --rounds $ROUNDS_PER_ITER \
    --workers $WORKERS \
    --prefetch $PREFETCH | tee -a "$RESULTS_FILE"
  collected_linear=$((collected_linear + ROUNDS_PER_ITER))
done

# 5) A/B test baseline vs gbdt
python main.py loop \
  --ab-test \
  --experiment reranker_ab_test_gbdt \
  --model-path "$MODEL_PATH" \
  --rounds 1000 \
  --workers $WORKERS \
  --prefetch $PREFETCH | tee -a "$RESULTS_FILE"

# 6) Show stats
python main.py stats --experiment reranker_ab_test_gbdt | tee -a "$RESULTS_FILE"

# Baseline vs Linear
python main.py loop \
  --ab-test \
  --experiment reranker_ab_test_baseline_linear \
  --rounds 1000 \
  --workers $WORKERS \
  --prefetch $PREFETCH | tee -a "$RESULTS_FILE"
python main.py stats --experiment reranker_ab_test_baseline_linear | tee -a "$RESULTS_FILE"

# Linear vs GBDT
python main.py loop \
  --ab-test \
  --experiment reranker_ab_test_linear_gbdt \
  --model-path "$MODEL_PATH" \
  --rounds 1000 \
  --workers $WORKERS \
  --prefetch $PREFETCH | tee -a "$RESULTS_FILE"
python main.py stats --experiment reranker_ab_test_linear_gbdt | tee -a "$RESULTS_FILE"

echo "Pipeline complete. Model: $MODEL_PATH" | tee -a "$RESULTS_FILE"
