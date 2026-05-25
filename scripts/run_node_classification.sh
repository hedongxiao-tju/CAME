#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/eval_config.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset> [dataset...]" >&2
  echo "Example: $0 product arxiv ele-fashion wikics books-nc" >&2
  exit 1
fi

activate_env
require_file "$CHECKPOINT"

for dataset in "$@"; do
  data_dir="$(resolve_dataset_dir "$dataset" || true)"
  if [[ -z "${data_dir:-}" ]]; then
    echo "Skip $dataset: dataset directory not found under $DATA_ROOT" >&2
    continue
  fi

  log_file="$LOG_DIR/node_${dataset}.log"
  cmd=(
    python "$CAME_ROOT/eval_node_classification.py"
    --data-dir "$data_dir"
    --checkpoint "$CHECKPOINT"
    --device "$DEVICE"
    --embed-type z3
    --use-gate
    --classifier "$NODE_CLASSIFIER"
    --classifier-epochs "$NODE_CLASSIFIER_EPOCHS"
    --log-every "$NODE_LOG_EVERY"
  )

  if [[ "$dataset" == "books-nc" ]]; then
    require_file "$BOOKS_NC_SUBGRAPH_CACHE_PATH"
    require_file "$BOOKS_NC_SUBGRAPH_CACHE_META_PATH"
    cmd+=(
      --mode stream
      --subgraph-cache-path "$BOOKS_NC_SUBGRAPH_CACHE_PATH"
      --subgraph-cache-meta-path "$BOOKS_NC_SUBGRAPH_CACHE_META_PATH"
    )
  else
    cmd+=(--mode full)
  fi

  echo "==> Node classification: $dataset"
  CUDA_VISIBLE_DEVICES="$GPU_ID" run_logged "$log_file" "${cmd[@]}"
done
