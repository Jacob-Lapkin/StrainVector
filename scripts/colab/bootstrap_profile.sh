#!/usr/bin/env bash
set -euo pipefail

# Google Colab bootstrap to profile a sample against a reference DB.
# This version skips venv creation (Colab uses system Python)
#
# Usage in Colab:
#   !bash scripts/colab/bootstrap_profile.sh
#
# Optional env overrides:
#   SAMPLE=data/samples/sample.fasta DB=refdb OUT=results.json \
#   WINDOW=5000 STRIDE=5000 MIN_CONTIG_LEN=1000 MAX_NS_FRAC=0.1 TOP_K=50 \
#   SIM_THRESHOLD=0.85 AGG_MODE=top1 WEIGHTING_MODE=inverse MODEL=... DEVICE=auto

PYTHON=${PYTHON:-python3}

SAMPLE=${SAMPLE:-data/samples/MGYA00579260_contigs.fasta}
DB=${DB:-refdb}

# Save profile outputs inside their own folder by default
OUT=${OUT:-profiles/results.json}

# Track whether user set these to auto-tune for examples/
if [ -z "${WINDOW+x}" ]; then WINDOW_WAS_SET=0; else WINDOW_WAS_SET=1; fi
if [ -z "${STRIDE+x}" ]; then STRIDE_WAS_SET=0; else STRIDE_WAS_SET=1; fi
if [ -z "${MIN_CONTIG_LEN+x}" ]; then MIN_CONTIG_LEN_WAS_SET=0; else MIN_CONTIG_LEN_WAS_SET=1; fi

WINDOW=${WINDOW:-3000}
STRIDE=${STRIDE:-3000}
MIN_CONTIG_LEN=${MIN_CONTIG_LEN:-1000}
MAX_NS_FRAC=${MAX_NS_FRAC:-0.1}
TOP_K=${TOP_K:-50}

# New options
SIM_THRESHOLD=${SIM_THRESHOLD:-0.85}
AGG_MODE=${AGG_MODE:-top1}   # options: top1, sum
WEIGHTING_MODE=${WEIGHTING_MODE:-inverse}   # options: uniform, inverse, quadratic

# Downsampling options for speed (set to enable)
DOWNSAMPLE_RATE=${DOWNSAMPLE_RATE:-5}      # process 1 out of every N windows (1 = no downsampling)
MAX_WINDOWS=${MAX_WINDOWS:-10000}                 # cap at N windows total (empty = no cap)
SAMPLE_FRACTION=${SAMPLE_FRACTION:-}         # random sample fraction 0-1 (empty = no sampling)

# Default model: DNABERT-2 (Colab typically has GPU)
MODEL=${MODEL:-zhihan1996/DNABERT-2-117M}

DEVICE=${DEVICE:-auto}
POOLING=${POOLING:-cls}
NORMALIZE=${NORMALIZE:-false}

echo "[StrainVector Colab] Using system Python: $PYTHON"
echo "[StrainVector Colab] Skipping venv creation (using Colab environment)"

echo "[StrainVector Colab] Ensuring deps"
pip install -q -U pip setuptools wheel
pip install -q -e . --force-reinstall
pip install -q torch transformers numpy einops || true
pip install -q faiss-cpu || true

echo "[StrainVector Colab] Profiling sample: $SAMPLE against DB: $DB"

# Adjust windows for examples
case "$SAMPLE" in
  examples/*)
    if [ "$WINDOW_WAS_SET" -eq 0 ]; then WINDOW=60; fi
    if [ "$STRIDE_WAS_SET" -eq 0 ]; then STRIDE=60; fi
    if [ "$MIN_CONTIG_LEN_WAS_SET" -eq 0 ]; then MIN_CONTIG_LEN=1; fi
    echo "[StrainVector Colab] Detected examples input; using small windows (WINDOW=$WINDOW, STRIDE=$STRIDE, MIN_CONTIG_LEN=$MIN_CONTIG_LEN). Override via env vars."
    ;;
esac

# Add --no-normalize flag if NORMALIZE is false (must match reference DB setting)
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
strainvector profile \
  --sample "$SAMPLE" \
  --db "$DB" \
  --out "$OUT" \
  --top-k "$TOP_K" \
  --window "$WINDOW" \
  --stride "$STRIDE" \
  --min-contig-len "$MIN_CONTIG_LEN" \
  --max-ns-frac "$MAX_NS_FRAC" \
  --model "$MODEL" \
  --pooling "$POOLING" \
  --device "$DEVICE" \
  --similarity-threshold "$SIM_THRESHOLD" \
  --aggregate-mode "$AGG_MODE" \
  --weighting-mode "$WEIGHTING_MODE" \
  $NORMALIZE_FLAG \
  $DOWNSAMPLE_FLAGS
set +x

echo "[StrainVector Colab] Done. Summary: $OUT; neighbors: $(dirname "$OUT")/neighbors.jsonl"
