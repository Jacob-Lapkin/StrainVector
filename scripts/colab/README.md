# Google Colab Scripts

These scripts are modified versions of the main bootstrap scripts, optimized for Google Colab environments.

## Key Differences from Main Scripts

- **No virtual environment creation**: Colab uses a system Python environment, so venv creation is skipped
- **Direct pip installation**: Packages are installed directly to the Colab environment
- **GPU support**: Default settings assume CUDA availability (use `DEVICE=cuda` if needed)

## Usage in Google Colab

### 1. Clone the Repository

```python
!git clone https://github.com/jacoblapkin/strainvector.git
%cd strainvector
```

### 2. Run Bootstrap Scripts

#### Index (Build Reference Database)
```python
!bash scripts/colab/bootstrap_index.sh
```

With custom parameters:
```python
!INPUT=data/genomes OUT_DB=refdb BATCH_SIZE=32 bash scripts/colab/bootstrap_index.sh
```

#### Profile (Query Sample Against Reference)
```python
!bash scripts/colab/bootstrap_profile.sh
```

With custom parameters:
```python
!SAMPLE=data/samples/sample.fasta WEIGHTING_MODE=inverse bash scripts/colab/bootstrap_profile.sh
```

#### Compare (Pairwise Genome Comparison)
```python
!bash scripts/colab/bootstrap_compare.sh
```

## Environment Variables

All the same environment variables from the main scripts work here:

**Indexing (bootstrap_index.sh):**
- `INPUT`: Input FASTA directory or file (default: `data/genomes`)
- `OUT_DB`: Output database directory (default: `refdb`)
- `BATCH_SIZE`: Embedding batch size (default: `16`)
- `SHARD_SIZE`: Vectors per shard file (default: `50000`)
- `WINDOW`: Window size in bp (default: `3000`)
- `STRIDE`: Stride between windows (default: `3000`)
- `MODEL`: HuggingFace model name (default: `zhihan1996/DNABERT-2-117M`)
- `DEVICE`: cpu|cuda|auto (default: `auto`)

**Profiling (bootstrap_profile.sh):**
- `SAMPLE`: Sample FASTA to profile (default: `data/samples/MGYA00579260_contigs.fasta`)
- `DB`: Reference database directory (default: `refdb`)
- `OUT`: Output JSON path (default: `profiles/results.json`)
- `TOP_K`: Neighbors per query chunk (default: `50`)
- `SIM_THRESHOLD`: Minimum similarity threshold (default: `0.85`)
- `AGG_MODE`: Aggregation mode - `top1` or `sum` (default: `top1`)
- `WEIGHTING_MODE`: Weighting strategy - `uniform`, `inverse`, `quadratic` (default: `inverse`)
- `DOWNSAMPLE_RATE`: Process 1 in N windows (default: `5`)
- `MAX_WINDOWS`: Cap total windows processed (default: `10000`)

And more (see individual script headers)

## Tips for Colab

- **GPU Runtime**: Enable GPU in Runtime > Change runtime type for faster processing
- **Session persistence**: Colab sessions timeout after inactivity; consider saving outputs to Google Drive
- **Memory limits**: Free Colab has ~12GB RAM; adjust `BATCH_SIZE` if you encounter OOM errors

## Example Workflow

```python
# Clone repo
!git clone https://github.com/jacoblapkin/strainvector.git
%cd strainvector

# Download example genomes (if needed)
!mkdir -p data/genomes
# ... download your genomes ...

# Build reference database
!bash scripts/colab/bootstrap_index.sh

# Profile a sample
!SAMPLE=data/samples/sample.fasta bash scripts/colab/bootstrap_profile.sh

# View results
import json
with open('profiles/results.json') as f:
    results = json.load(f)
print(json.dumps(results, indent=2))
```
