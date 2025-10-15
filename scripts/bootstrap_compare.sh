#!/usr/bin/env bash
set -euo pipefail

# One-command bootstrap to compare multiple samples.
# Usage:
#   bash scripts/bootstrap_compare.sh
#
# Comparison modes:
#   - centroid: Fast genome-level average (single mean vector per sample)
#   - window: Detailed window-by-window comparison with distribution statistics
#
# Optional env overrides:
#   PYTHON=python3.11 VENV_DIR=.venv SAMPLES="examples/genomes examples/genomes" \
#   OUT=compare.json OUT_TSV=compare.tsv WINDOW=3000 STRIDE=3000 MIN_CONTIG_LEN=1000 \
#   MAX_NS_FRAC=0.1 MODEL=... DEVICE=auto METRIC=cosine \
#   COMPARISON_MODE=window     # or 'centroid' for fast mode \
#   DOWNSAMPLE_RATE=5 MAX_WINDOWS=10000 SAMPLE_FRACTION=0.2

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}
OS_NAME=$(uname -s || echo Unknown)

read -r -a SAMPLE_ARR <<< "${SAMPLES:-data/samples}"
OUT=${OUT:-compare.json}
OUT_TSV=${OUT_TSV:-}

if [ -z "${WINDOW+x}" ]; then WINDOW_WAS_SET=0; else WINDOW_WAS_SET=1; fi
if [ -z "${STRIDE+x}" ]; then STRIDE_WAS_SET=0; else STRIDE_WAS_SET=1; fi
if [ -z "${MIN_CONTIG_LEN+x}" ]; then MIN_CONTIG_LEN_WAS_SET=0; else MIN_CONTIG_LEN_WAS_SET=1; fi

WINDOW=${WINDOW:-3000}
STRIDE=${STRIDE:-3000}
MIN_CONTIG_LEN=${MIN_CONTIG_LEN:-1000}
MAX_NS_FRAC=${MAX_NS_FRAC:-0.1}

# Downsampling options for speed (set to enable)
DOWNSAMPLE_RATE=${DOWNSAMPLE_RATE:-5}      # process 1 out of every N windows (1 = no downsampling)
MAX_WINDOWS=${MAX_WINDOWS:-10000}          # cap at N windows total per sample (empty = no cap)
SAMPLE_FRACTION=${SAMPLE_FRACTION:-}       # random sample fraction 0-1 (empty = no sampling)

if [ -z "${MODEL:-}" ]; then
  if [ "$OS_NAME" = "Darwin" ]; then
    MODEL="zhihan1996/DNA_bert_6"
    echo "[StrainVector] macOS detected; defaulting MODEL to $MODEL (CPU-friendly). Override by setting MODEL=..."
  else
    MODEL="zhihan1996/DNABERT-2-117M"
  fi
fi

DEVICE=${DEVICE:-auto}
POOLING=${POOLING:-cls}
NORMALIZE=${NORMALIZE:-false}
METRIC=${METRIC:-cosine}
COMPARISON_MODE=${COMPARISON_MODE:-window}

echo "[StrainVector] Using Python: $PYTHON"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# macOS OpenMP duplicate runtime workaround
if [ "$OS_NAME" = "Darwin" ]; then
  export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  echo "[StrainVector] macOS: set KMP_DUPLICATE_LIB_OK=$KMP_DUPLICATE_LIB_OK; OMP/MKL/OPENBLAS threads limited to 1"
fi

echo "[StrainVector] Ensuring deps"
pip install -q -U pip setuptools wheel
pip install -q -e . --force-reinstall
pip install -q torch transformers numpy einops || true

echo "[StrainVector] Compare: samples=${SAMPLE_ARR[*]} out=$OUT"

# If all samples are under examples/, auto small windows unless overridden
ALL_EXAMPLES=1
for s in "${SAMPLE_ARR[@]}"; do
  case "$s" in
    examples/*) : ;;
    *) ALL_EXAMPLES=0 ;;
  esac
done
if [ $ALL_EXAMPLES -eq 1 ]; then
  if [ "$WINDOW_WAS_SET" -eq 0 ]; then WINDOW=60; fi
  if [ "$STRIDE_WAS_SET" -eq 0 ]; then STRIDE=60; fi
  if [ "$MIN_CONTIG_LEN_WAS_SET" -eq 0 ]; then MIN_CONTIG_LEN=1; fi
  echo "[StrainVector] Detected examples inputs; using small windows (WINDOW=$WINDOW, STRIDE=$STRIDE, MIN_CONTIG_LEN=$MIN_CONTIG_LEN)."
fi

# Add --no-normalize flag if NORMALIZE is false (must match other commands)
NORMALIZE_FLAG=""
if [ "$NORMALIZE" = "false" ]; then
  NORMALIZE_FLAG="--no-normalize"
fi

# Build downsampling flags
DOWNSAMPLE_FLAGS=""
if [ "$DOWNSAMPLE_RATE" -gt 1 ]; then
  DOWNSAMPLE_FLAGS="$DOWNSAMPLE_FLAGS --downsample-rate $DOWNSAMPLE_RATE"
fi
if [ -n "$MAX_WINDOWS" ]; then
  DOWNSAMPLE_FLAGS="$DOWNSAMPLE_FLAGS --max-windows $MAX_WINDOWS"
fi
if [ -n "$SAMPLE_FRACTION" ]; then
  DOWNSAMPLE_FLAGS="$DOWNSAMPLE_FLAGS --sample-fraction $SAMPLE_FRACTION"
fi

set -x
strainvector compare \
  --samples "${SAMPLE_ARR[@]}" \
  --out "$OUT" \
  ${OUT_TSV:+--out-tsv "$OUT_TSV"} \
  --window "$WINDOW" \
  --stride "$STRIDE" \
  --min-contig-len "$MIN_CONTIG_LEN" \
  --max-ns-frac "$MAX_NS_FRAC" \
  --model "$MODEL" \
  --pooling "$POOLING" \
  --device "$DEVICE" \
  --metric "$METRIC" \
  --comparison-mode "$COMPARISON_MODE" \
  $NORMALIZE_FLAG \
  $DOWNSAMPLE_FLAGS
set +x

echo "[StrainVector] Done. Matrix: $OUT ${OUT_TSV:+| TSV: $OUT_TSV}"
