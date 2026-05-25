#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/eval_config.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset> [dataset...]" >&2
  echo "Example: $0 wn18rr fb15k237" >&2
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

  case "${dataset,,}" in
    wn18rr)
      kg_name="WN18RR"
      ;;
    fb15k237)
      kg_name="FB15K237"
      ;;
    *)
      echo "Skip $dataset: unsupported KG dataset name" >&2
      continue
      ;;
  esac

  log_file="$LOG_DIR/kg_${dataset}.log"
  cmd=(
    python "$CAME_ROOT/eval_kg.py"
    --data-dir "$data_dir"
    --dataset "$kg_name"
    --checkpoint "$CHECKPOINT"
    --device "$DEVICE"
    --embed-type z3
    --use-gate
    --lr-f "$KG_LR_F"
    --weight-decay-f "$KG_WEIGHT_DECAY_F"
    --lp-epochs "$KG_LP_EPOCHS"
  )

  if [[ -n "$KG_EVAL_BATCH_SIZE" ]]; then
    cmd+=(--eval-batch-size "$KG_EVAL_BATCH_SIZE")
  fi

  echo "==> KG evaluation: $dataset"
  CUDA_VISIBLE_DEVICES="$GPU_ID" run_logged "$log_file" "${cmd[@]}"
done
