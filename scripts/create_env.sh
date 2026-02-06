#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="who-are-you"
PY_VER="3.12"
CUDA_VER="12.4"
ENGINE=""

usage() {
  cat <<'EOF'
Usage: scripts/create_env.sh [options]

Options:
  --name NAME         Conda environment name (default: who-are-you)
  --python VER        Python version (default: 3.12)
  --cuda VER          CUDA toolkit version (default: 12.4)
  --engine ENGINE     conda|micromamba (auto-detect if omitted)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) ENV_NAME="$2"; shift 2 ;;
    --python) PY_VER="$2"; shift 2 ;;
    --cuda) CUDA_VER="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$ENGINE" ]]; then
  if command -v micromamba >/dev/null 2>&1; then
    ENGINE="micromamba"
  elif command -v conda >/dev/null 2>&1; then
    ENGINE="conda"
  else
    echo "ERROR: Neither conda nor micromamba found on PATH."
    exit 1
  fi
fi

if [[ "$ENGINE" != "conda" && "$ENGINE" != "micromamba" ]]; then
  echo "ERROR: --engine must be 'conda' or 'micromamba'"
  exit 1
fi

echo "Using engine: $ENGINE"
echo "Creating env: $ENV_NAME (python=$PY_VER, cuda-toolkit=$CUDA_VER)"

"$ENGINE" create -y -n "$ENV_NAME" -c nvidia -c conda-forge \
  "python=$PY_VER" "cuda-toolkit=$CUDA_VER" pip

echo "Installing uv"
if [[ "$ENGINE" == "conda" ]]; then
  conda install -n "$ENV_NAME" -c conda-forge uv -y
else
  micromamba install -n "$ENV_NAME" -c conda-forge uv -y
fi

echo "Exporting environment.yml"
if ! "$ENGINE" env export -n "$ENV_NAME" --no-builds > environment.yml; then
  "$ENGINE" env export -n "$ENV_NAME" > environment.yml
fi

echo "Done. environment.yml written."
