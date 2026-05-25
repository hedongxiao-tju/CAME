#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/eval_config.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset> [dataset...]" >&2
  echo "Example: $0 books-lp amazon-cloth amazon-sports" >&2
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

  log_file="$LOG_DIR/link_${dataset}.log"
  cmd=(
    python "$CAME_ROOT/eval_link_prediction.py"
    --data-dir "$data_dir"
    --checkpoint "$CHECKPOINT"
    --device "$DEVICE"
    --embed-type z3
    --use-gate
    --mode full
    --eval-batch-size "$LP_EVAL_BATCH_SIZE"
  )

  echo "==> Link prediction: $dataset"
  CUDA_VISIBLE_DEVICES="$GPU_ID" run_logged "$log_file" "${cmd[@]}"
done
