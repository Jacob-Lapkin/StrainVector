# Usage

This guide covers the current CLI workflow for building a reference database and outlines output formats. Core profiling logic is not yet implemented.

## Input Format Requirements

**StrainVector requires assembled sequences in FASTA format:**

- **Supported formats**: `.fa`, `.fasta`, `.fna` (plain text or `.gz` compressed)
- **Supported input types**:
  - Assembled reference genomes
  - Metagenomic contigs or scaffolds (from assembly tools like metaSPAdes, MEGAHIT)
  - Isolate genome assemblies
- **NOT supported**: Raw sequencing reads in FASTQ format

**Important**: If you have raw metagenomic reads, you must assemble them first using a metagenomic assembler before using StrainVector. The tool is designed to work with contiguous sequences (typically 1kb+ length) to leverage the contextual learning of transformer models.

## Build Reference Database

Examples:
- Minimal (init + embeddings + FAISS index if available):
  - `strainvector index --input examples/genomes --out-db refdb/`
- Init only (no embeddings):
  - `strainvector index --input examples/genomes --out-db refdb/ --no-embed`
- With metadata and custom chunking:
  - `strainvector index --input examples/genomes --out-db refdb/ --metadata examples/metadata.csv --window 5000 --stride 2500`

### Common Flags (index)
- `--input`: Path to FASTA file or directory (supports .fa, .fna, .fasta and .gz).
- `--out-db`: Output directory for the reference DB.
- `--name`: Reference DB name (default: `strainvector-refdb`).
- `--metadata`: Optional CSV of genome metadata.
- `--k`: k-mer size parameter for documentation; embedding uses the model’s tokenizer (default: 6).
- `--window`: Window length in bp for chunking (default: 5000).
- `--stride`: Stride between windows in bp (default: 5000).
- `--min-contig-len`: Skip contigs shorter than this length (default: 1000).
- `--max-ns-frac`: Max fraction of Ns allowed in a window (default: 0.1).
- `--model`: HF model name (default: `zhihan1996/DNABERT-2-117M`). Note: DNABERT-2 now works on macOS via a Triton compatibility workaround with MPS acceleration support.
- `--tokenizer`: HF tokenizer override (default: same as model).
- `--pooling`: `mean` or `cls` pooling (default: `mean`).
- `--no-normalize`: Disable L2 normalization of embeddings.
- `--precision`: `fp32`, `fp16`, or `bf16` (default: `fp32`).
- `--batch-size`: Sequences per embed batch (default: 16).
- `--num-workers`: Parallelize per-file embedding with processes (default: 4). Use 1 to disable.
- `--device`: `cpu|cuda|auto` (default: `auto`).
- `--seed`: Random seed (default: 42).
- `--shard-size`: Vectors per shard file (default: 50000).
- `--index-backend`: `faiss|pgvector|milvus|flat` (default: `faiss`).
- `--metric`: `cosine|ip|l2` (default: `cosine`).
- `--faiss-index`: FAISS type (e.g., `Flat`, `IVF1024`) (default: `Flat`).
- `--shards`: Reserved for future sharding (default: 1).
- `--force`: Overwrite an existing out-db directory.
- `--no-embed`: Skip embedding generation (init only).
- `--no-index`: Skip index construction.
- `--log-file`: Write index logs to this path (default: `refdb/logs/index.log`).

### Outputs
- `refdb/config.json`: Captures build parameters and inputs.
- `refdb/vectors/part_XXXXX.npy`: Sharded embeddings (float32 arrays, N x D).
- `refdb/vectors/part_XXXXX.jsonl`: One JSON metadata object per vector.
- `refdb/index/faiss.index`: FAISS index (if built).
- `refdb/metadata/`: Notes about metadata sources.
- `refdb/logs/`: Logs and stats files.
- `refdb/logs/index_stats.json`: Embedding stats (vectors, windows, shards, dim).
- `refdb/logs/index.log`: Log lines for the indexing run.
- `refdb/logs/faiss_stats.json`: Index stats if FAISS was built.

### Vector Metadata Schema (per line in JSONL)
- `record_id`: Monotonically increasing integer ID per window.
- `source`: Absolute or relative path to the source FASTA file.
- `source_basename`: The FASTA filename.
- `contig`: Contig/header token from FASTA.
- `start`: Start coordinate (0-based, inclusive) of the window.
- `end`: End coordinate (0-based, exclusive) of the window.
- `length`: Length of the window in bp.
- `k`: k-mer parameter used for documentation.
- `window`: Window size used for chunking.
- `stride`: Stride used for chunking.
- `model`: Model name used to compute embeddings.

### Metadata CSV (optional)
Metadata is optional and currently recorded for provenance only (not used in embedding or indexing). Recommended fields and examples are documented in `docs/METADATA.md`.

An example CSV is provided at `examples/metadata.csv`.

## Profile a Sample
The `profile` subcommand embeds contigs from a sample, retrieves nearest neighbors from the reference DB (FAISS if present, otherwise brute force), and aggregates hits to report likely strains.

Example usage:
```
strainvector profile --sample examples/genomes --db refdb/ --out results.json --window 60 --stride 60 --min-contig-len 1
```

Common flags (profile)
- `--sample`: Path to sample FASTA or directory.
- `--db`: Reference DB directory (from `strainvector index`).
- `--out`: Output JSON summary path (neighbors go to `neighbors.jsonl` beside it).
- `--top-k`: Number of neighbors per query chunk (default: 50).
- `--window`, `--stride`, `--min-contig-len`, `--max-ns-frac`: Same semantics as `index`.
- `--model`, `--tokenizer`, `--pooling`, `--no-normalize`, `--precision`, `--batch-size`, `--device`, `--metric`: Same semantics as `index`.
- `--similarity-threshold`: Only count neighbors with similarity >= this value (default: 0.85).
- `--aggregate-mode`: `top1` (count only the best passing neighbor per chunk) or `sum` (sum all passing neighbors). Default: `top1`.
- `--weighting-mode`: Aggregation weighting strategy (default: `uniform`):
  - `uniform`: All windows weighted equally (standard arithmetic mean)
  - `inverse`: Linear emphasis on differences (weight = 1 - similarity), hypothetically emphasizing variable/accessory regions
  - `quadratic`: Strong emphasis on differences (weight = (1 - similarity)²), for highly conserved genomes
  - **Note**: Empirical testing shows minimal impact with pretrained models due to narrow similarity distributions. Most useful for evaluating future fine-tuned embeddings.
- `--downsample-rate`: Process 1 out of every N windows for faster profiling (default: 1, meaning no downsampling). Example: `--downsample-rate 5` processes every 5th window.
- `--max-windows`: Cap the total number of windows processed (default: no cap). Example: `--max-windows 10000` stops after 10,000 windows.
- `--sample-fraction`: Randomly sample this fraction of windows (0-1 range, default: no sampling). Example: `--sample-fraction 0.1` processes 10% of windows randomly.
- `--log-file`: Write profile logs to this path (default: `results.log` next to `results.json`).

Outputs
- `results.json`: Summary with candidate references (`source_basename`) and statistics per candidate:
  - `count`: Number of query windows matching this reference
  - `mean_similarity`: Average similarity across all matching windows (arithmetic mean)
  - `weighted_mean_similarity`: Weighted average similarity using the specified weighting mode
  - `max_similarity`: Highest similarity score observed
  - `hybrid_score`: Abundance-weighted similarity score. Uses `count × weighted_mean_similarity` when `--weighting-mode` is not `uniform`, otherwise `count × mean_similarity`. Candidates are sorted by this metric to prioritize abundant and similar strains
- `neighbors.jsonl`: Per-query-chunk neighbors with similarity scores and reference metadata markers.
- `profile_stats.json`: Small summary of counts; a `results.log` is also written alongside `results.json`.

Bootstrap script
- `bash scripts/bootstrap_profile.sh` runs an end-to-end profile against the example DB, auto-tuning small windows when the sample is under `examples/`.

## Compare Samples
Performs pairwise comparison of genome samples using one of two modes.

### Comparison Modes

**Centroid Mode** (`--comparison-mode centroid`)
- Computes a single mean vector per sample by averaging all window embeddings
- Fast, suitable for high-level taxonomic comparison
- Output: Simple similarity matrix

**Window Mode** (`--comparison-mode window`, default)
- Compares samples window-by-window, finding best matches
- Computes distribution statistics: mean, median, std, percentiles, % below thresholds
- Reveals strain-level variation when models have sufficient resolution
- Slower but more detailed than centroid mode

### Examples

Centroid mode (fast):
```
strainvector compare --samples genome1.fa genome2.fa \
  --comparison-mode centroid \
  --out compare.json
```

Window mode (detailed):
```
strainvector compare --samples genome1.fa genome2.fa \
  --comparison-mode window \
  --out compare.json --out-tsv compare.tsv
```

### Common flags (compare)
- `--samples`: One or more FASTA paths or directories (recursively scanned).
- `--out`: Output JSON containing `labels`, `metric`, `comparison_mode`, `matrix`, and per-sample `stats`.
- `--out-tsv`: Optional TSV matrix for quick viewing.
- `--comparison-mode`: `centroid` (fast, genome-level) or `window` (detailed, distribution stats). Default: `window`.
- `--downsample-rate`: Process 1 out of every N windows (default: 1, no downsampling). Example: `--downsample-rate 5`.
- `--max-windows`: Cap total windows processed per sample (default: no cap). Example: `--max-windows 10000`.
- `--sample-fraction`: Randomly sample this fraction of windows (0-1, default: no sampling). Example: `--sample-fraction 0.2`.
- `--window`, `--stride`, `--min-contig-len`, `--max-ns-frac`, `--model`, `--tokenizer`, `--pooling`, `--no-normalize`, `--precision`, `--batch-size`, `--device`, `--metric`, `--log-file`: Same semantics as other commands.

### Output Format

**Centroid mode** outputs:
```json
{
  "labels": ["sample1.fa", "sample2.fa"],
  "metric": "cosine",
  "comparison_mode": "centroid",
  "matrix": [[1.0, 0.95], [0.95, 1.0]],
  "stats": {...}
}
```

**Window mode** outputs include additional `distributions` field:
```json
{
  "labels": ["sample1.fa", "sample2.fa"],
  "metric": "cosine",
  "comparison_mode": "window",
  "matrix": [[1.0, 0.94], [0.94, 1.0]],
  "distributions": {
    "sample1.fa__vs__sample2.fa": {
      "mean": 0.94,
      "median": 0.95,
      "std": 0.08,
      "min": 0.65,
      "max": 0.99,
      "percentiles": {"10": 0.82, "25": 0.89, "50": 0.95, "75": 0.97, "90": 0.98},
      "below_threshold_pct": {"0.95": 42.3, "0.90": 18.5, "0.85": 8.2, "0.80": 3.1},
      "n_windows_i": 10000,
      "n_windows_j": 10000
    }
  },
  "stats": {...}
}
```

The `below_threshold_pct` field shows the percentage of windows with similarity below each threshold, useful for identifying samples with many poorly-matching regions (strain-specific genes, variable loci).
