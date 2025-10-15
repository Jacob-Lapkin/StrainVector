#!/usr/bin/env bash
set -euo pipefail

# One-command bootstrap for building a reference DB from the examples dataset
# or from custom inputs via environment overrides.
#
# Usage:
#   bash scripts/bootstrap_index.sh
#
# Optional environment overrides:
#   PYTHON=python3.11 VENV_DIR=.venv INPUT=examples/genomes OUT_DB=refdb \
#   METADATA=examples/metadata.csv NO_INDEX=0 BATCH_SIZE=16 SHARD_SIZE=50000 \
#   WINDOW=5000 STRIDE=5000 MIN_CONTIG_LEN=1000 MAX_NS_FRAC=0.1 \
#   MODEL=zhihan1996/DNABERT-2-117M POOLING=mean DEVICE=auto SEED=42 \
#   FAISS_INDEX=Flat METRIC=cosine

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}

INPUT=${INPUT:-data/genomes}
OUT_DB=${OUT_DB:-refdb}
METADATA=${METADATA:-examples/metadata.csv}
NO_INDEX=${NO_INDEX:-0}  # set to 1 to skip FAISS index

# Track whether user explicitly set these so we can auto-tune for examples/
if [ -z "${WINDOW+x}" ]; then WINDOW_WAS_SET=0; else WINDOW_WAS_SET=1; fi
if [ -z "${STRIDE+x}" ]; then STRIDE_WAS_SET=0; else STRIDE_WAS_SET=1; fi
if [ -z "${MIN_CONTIG_LEN+x}" ]; then MIN_CONTIG_LEN_WAS_SET=0; else MIN_CONTIG_LEN_WAS_SET=1; fi

BATCH_SIZE=${BATCH_SIZE:-16}
SHARD_SIZE=${SHARD_SIZE:-50000}
WINDOW=${WINDOW:-3000}
STRIDE=${STRIDE:-3000}
MIN_CONTIG_LEN=${MIN_CONTIG_LEN:-1000}
MAX_NS_FRAC=${MAX_NS_FRAC:-0.1}

OS_NAME=$(uname -s || echo Unknown)
# Default model: DNABERT-2 on Linux, DNABERT v1 on macOS (Triton not supported)
if [ -z "${MODEL:-}" ]; then
  if [ "$OS_NAME" = "Darwin" ]; then
    MODEL="zhihan1996/DNA_bert_6"
    echo "[StrainVector] macOS detected; defaulting MODEL to $MODEL (CPU-friendly). Override by setting MODEL=..."
  else
    MODEL="zhihan1996/DNABERT-2-117M"
  fi
fi
POOLING=${POOLING:-cls}
NORMALIZE=${NORMALIZE:-false}
DEVICE=${DEVICE:-auto}
SEED=${SEED:-42}
FAISS_INDEX=${FAISS_INDEX:-Flat}
METRIC=${METRIC:-cosine}

# macOS memory optimizations: reduce batch size and shard size to prevent OOM
if [ "$OS_NAME" = "Darwin" ]; then
  if [ -z "${BATCH_SIZE:-}" ] || [ "$BATCH_SIZE" -eq 16 ]; then
    BATCH_SIZE=4
    echo "[StrainVector] macOS detected; reducing BATCH_SIZE to $BATCH_SIZE for memory efficiency. Override by setting BATCH_SIZE=..."
  fi
  if [ -z "${SHARD_SIZE:-}" ] || [ "$SHARD_SIZE" -eq 50000 ]; then
    SHARD_SIZE=10000
    echo "[StrainVector] macOS detected; reducing SHARD_SIZE to $SHARD_SIZE for memory efficiency. Override by setting SHARD_SIZE=..."
  fi
fi

echo "[StrainVector] Using Python: $PYTHON"
echo "[StrainVector] Creating venv at: $VENV_DIR"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[StrainVector] Upgrading build tools"
pip install -q -U pip setuptools wheel

echo "[StrainVector] Installing StrainVector in editable mode (force-reinstall)"
pip install -q -e . --force-reinstall

echo "[StrainVector] Installing minimal runtime deps: torch, transformers, numpy, einops"
if ! pip install -q torch transformers numpy einops; then
  echo "[StrainVector] ERROR: Failed to install torch/transformers/numpy/einops" >&2
  exit 1
fi

INDEX_FLAGS=()
if [ "$NO_INDEX" -eq 0 ]; then
  echo "[StrainVector] Attempting to install FAISS (CPU)"
  if pip install -q faiss-cpu; then
    echo "[StrainVector] FAISS installed. Index will be built."
  else
    echo "[StrainVector] WARNING: Could not install FAISS. Will skip index build."
    INDEX_FLAGS+=(--no-index)
  fi
else
  INDEX_FLAGS+=(--no-index)
fi

echo "[StrainVector] Building reference DB: input=$INPUT out_db=$OUT_DB"

# If using the bundled examples, auto-adjust to small windows unless user overrode
case "$INPUT" in
  examples/*)
    if [ "$WINDOW_WAS_SET" -eq 0 ]; then WINDOW=60; fi
    if [ "$STRIDE_WAS_SET" -eq 0 ]; then STRIDE=60; fi
    if [ "$MIN_CONTIG_LEN_WAS_SET" -eq 0 ]; then MIN_CONTIG_LEN=1; fi
    echo "[StrainVector] Detected examples input; using small windows (WINDOW=$WINDOW, STRIDE=$STRIDE, MIN_CONTIG_LEN=$MIN_CONTIG_LEN). Override via env vars."
    ;;
esac
cmd=(
  strainvector index
  --input "$INPUT"
  --out-db "$OUT_DB"
  --metadata "$METADATA"
  --batch-size "$BATCH_SIZE"
  --shard-size "$SHARD_SIZE"
  --window "$WINDOW"
  --stride "$STRIDE"
  --min-contig-len "$MIN_CONTIG_LEN"
  --max-ns-frac "$MAX_NS_FRAC"
  --model "zhihan1996/DNABERT-2-117M"
  --pooling "$POOLING"
  --device "$DEVICE"
  --seed "$SEED"
  --faiss-index "$FAISS_INDEX"
  --metric "$METRIC"
  --force
)

# Add --no-normalize flag if NORMALIZE is false (better strain-level resolution)
if [ "$NORMALIZE" = "false" ]; then
  cmd+=(--no-normalize)
fi

# Append optional flags safely if any are present
if [ ${#INDEX_FLAGS[@]} -gt 0 ]; then
  cmd+=("${INDEX_FLAGS[@]}")
fi

set -x
"${cmd[@]}"
set +x

echo "[StrainVector] Done. Outputs in $OUT_DB (vectors/, index/, config.json)."
