#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-graphmae2}"

CAME_ROOT="${CAME_ROOT:-$REPO_ROOT}"
DATA_ROOT="${DATA_ROOT:-$CAME_ROOT/data}"
CHECKPOINT="${CHECKPOINT:-$CAME_ROOT/checkpoints/came_multigraph_pretrain_epoch8.pt}"

GPU_ID="${GPU_ID:-2}"
DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="${LOG_DIR:-$CAME_ROOT/logs/eval}"

BOOKS_NC_SUBGRAPH_CACHE_PATH="${BOOKS_NC_SUBGRAPH_CACHE_PATH:-$DATA_ROOT/books-nc/subgraphs_top128_bs128.bin}"
BOOKS_NC_SUBGRAPH_CACHE_META_PATH="${BOOKS_NC_SUBGRAPH_CACHE_META_PATH:-$DATA_ROOT/books-nc/subgraphs_top128_bs128_meta.pt}"

NODE_CLASSIFIER="${NODE_CLASSIFIER:-linear}"
NODE_CLASSIFIER_EPOCHS="${NODE_CLASSIFIER_EPOCHS:-5000}"
NODE_LOG_EVERY="${NODE_LOG_EVERY:-100}"

LP_EVAL_BATCH_SIZE="${LP_EVAL_BATCH_SIZE:-500}"

KG_EVAL_BATCH_SIZE="${KG_EVAL_BATCH_SIZE:}"
KG_LR_F="${KG_LR_F:-0.01}"
KG_WEIGHT_DECAY_F="${KG_WEIGHT_DECAY_F:-0.0}"
KG_LP_EPOCHS="${KG_LP_EPOCHS:-5000}"

mkdir -p "$LOG_DIR"

activate_env() {
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

resolve_dataset_dir() {
  local name="$1"
  local candidates=(
    "$DATA_ROOT/$name"
    "$DATA_ROOT/${name,,}"
    "$DATA_ROOT/${name^^}"
  )

  if [[ "$name" == "product" ]]; then
    candidates+=("$DATA_ROOT/products")
  fi
  if [[ "$name" == "products" ]]; then
    candidates+=("$DATA_ROOT/product")
  fi
  if [[ "$name" == "fb15k237" ]]; then
    candidates+=("$DATA_ROOT/FB15K237")
  fi
  if [[ "$name" == "wn18rr" ]]; then
    candidates+=("$DATA_ROOT/WN18RR")
  fi

  local path
  for path in "${candidates[@]}"; do
    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  {
    echo "[$(date '+%F %T')] Running: $*"
    /usr/bin/time -f $'wall_time=%E\nmax_rss_kb=%M' "$@"
  } 2>&1 | tee "$log_file"
}
