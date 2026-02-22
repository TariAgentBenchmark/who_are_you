#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/reproduce_paper.sh <true_metadata_csv> <fake_metadata_csv>"
  exit 1
fi

TRUE_CSV="$1"
FAKE_CSV="$2"

echo "[0/5] Cleaning Mongo collections to avoid mixed runs"
uv run python - <<'PY'
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017")
for db_name in ["exploration", "windows"]:
    db = client[db_name]
    for coll in ["timit_true_extended", "real_time_extended"]:
        db[coll].drop()
        print(f"dropped {db_name}.{coll}")
PY

echo "[1/4] Extracting organic features into Mongo (exploration.timit_true_extended)"
printf 'n\n\ny\n' | uv run python -u core/handler.py bigram timit_true_extended "$TRUE_CSV"

echo "[2/4] Extracting deepfake features into Mongo (exploration.real_time_extended)"
printf 'y\n\ny\n' | uv run python -u core/handler.py bigram real_time_extended "$FAKE_CSV"

echo "[3/4] Copying collections from exploration.* to windows.* for threshold script compatibility"
uv run python - <<'PY'
import pymongo

src_db = "exploration"
dst_db = "windows"
collections = ["timit_true_extended", "real_time_extended"]

client = pymongo.MongoClient("mongodb://localhost:27017")
src = client[src_db]
dst = client[dst_db]

for name in collections:
    docs = list(src[name].find({}, {"_id": 0}))
    dst[name].drop()
    if docs:
        dst[name].insert_many(docs)
    print(f"copied {name}: {len(docs)} docs")
PY

echo "[4/4] Running threshold extraction and evaluation"
uv run python -u core/extract_threshold.py

echo "Done."
