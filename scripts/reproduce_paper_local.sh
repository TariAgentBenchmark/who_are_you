#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/reproduce_paper_local.sh <true_metadata_csv> <fake_metadata_csv>"
  exit 1
fi

TRUE_CSV="$1"
FAKE_CSV="$2"

export WRY_STORAGE_BACKEND=file
export WRY_FEATURE_DIR="${WRY_FEATURE_DIR:-outputs/local_features}"
export WRY_FORCE_REPROCESS="${WRY_FORCE_REPROCESS:-0}"

echo "Storage backend: $WRY_STORAGE_BACKEND"
echo "Feature dir: $WRY_FEATURE_DIR"
echo "Force reprocess: $WRY_FORCE_REPROCESS"

mkdir -p "$WRY_FEATURE_DIR"

echo "[1/3] Extracting organic features to local file store (timit_true_extended)"
printf 'n\n\ny\n' | uv run python -u core/handler.py bigram timit_true_extended "$TRUE_CSV"

echo "[2/3] Extracting deepfake features to local file store (real_time_extended)"
printf 'y\n\ny\n' | uv run python -u core/handler.py bigram real_time_extended "$FAKE_CSV"

echo "[3/3] Running threshold extraction and evaluation from local file store"
uv run python -u core/extract_threshold.py

echo "Done."
