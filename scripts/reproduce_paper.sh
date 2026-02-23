#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_paper.sh [options] <true_metadata_csv> <fake_metadata_csv>

Options:
  --clean                  Drop target Mongo collections before run
  --num-gpus N             Number of GPUs/processes for extraction (default: 1)
  --force-reprocess        Do not skip already-processed filepaths
  -h, --help               Show help
EOF
}

CLEAN=0
NUM_GPUS=1
FORCE_REPROCESS=0

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --force-reprocess) FORCE_REPROCESS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

TRUE_CSV="$1"
FAKE_CSV="$2"

if [[ "$CLEAN" -eq 1 ]]; then
  echo "[0/5] Cleaning Mongo collections"
  uv run python - <<'PY'
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017")
for db_name in ["exploration", "windows"]:
    db = client[db_name]
    for coll in ["timit_true_extended", "real_time_extended"]:
        db[coll].drop()
        print(f"dropped {db_name}.{coll}")
PY
fi

extract_collection() {
  local collection_name="$1"
  local metadata_csv="$2"
  local is_fake="$3"

  if [[ "$is_fake" -eq 1 ]]; then
    local prompt=$'y\n\ny\n'
  else
    local prompt=$'n\n\ny\n'
  fi

  if [[ "$NUM_GPUS" -le 1 ]]; then
    printf '%s' "$prompt" | WRY_FORCE_REPROCESS="$FORCE_REPROCESS" uv run python -u core/handler.py bigram "$collection_name" "$metadata_csv"
    return
  fi

  local shard_dir
  shard_dir="$(mktemp -d)"
  uv run python - "$metadata_csv" "$NUM_GPUS" "$shard_dir" <<'PY'
import csv
import hashlib
import os
import sys

in_csv = sys.argv[1]
num_gpus = int(sys.argv[2])
out_dir = sys.argv[3]

with open(in_csv, newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    header = reader.fieldnames

groups = [[] for _ in range(num_gpus)]
for row in rows:
    key = row["filepath"].encode("utf-8")
    shard_id = int(hashlib.md5(key).hexdigest(), 16) % num_gpus
    groups[shard_id].append(row)

for i, chunk in enumerate(groups):
    out_csv = os.path.join(out_dir, f"shard_{i}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(chunk)
    print(out_csv, len(chunk))
PY

  echo "Using $NUM_GPUS extraction processes for $collection_name"
  local pids=()
  local had_work=0
  for ((i=0; i<NUM_GPUS; i++)); do
    local shard_csv="$shard_dir/shard_$i.csv"
    if [[ $(wc -l < "$shard_csv") -le 1 ]]; then
      continue
    fi
    had_work=1
    (
      export CUDA_VISIBLE_DEVICES="$i"
      export WRY_FORCE_REPROCESS="$FORCE_REPROCESS"
      printf '%s' "$prompt" | uv run python -u core/handler.py bigram "$collection_name" "$shard_csv"
    ) &
    pids+=("$!")
  done

  if [[ "$had_work" -eq 0 ]]; then
    echo "No work items found for $collection_name"
    return
  fi

  local exit_code=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      exit_code=1
    fi
  done
  if [[ "$exit_code" -ne 0 ]]; then
    echo "At least one extraction worker failed for $collection_name"
    exit 1
  fi
}

echo "[1/4] Extracting organic features into Mongo (exploration.timit_true_extended)"
extract_collection "timit_true_extended" "$TRUE_CSV" 0

echo "[2/4] Extracting deepfake features into Mongo (exploration.real_time_extended)"
extract_collection "real_time_extended" "$FAKE_CSV" 1

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
