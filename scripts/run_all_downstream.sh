#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/run_node_classification.sh" product arxiv ele-fashion wikics books-nc
"$SCRIPT_DIR/run_link_prediction.sh" books-lp amazon-cloth amazon-sports
"$SCRIPT_DIR/run_kg.sh" wn18rr fb15k237
