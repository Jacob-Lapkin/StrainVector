# StrainVector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

An open-source Python pipeline for embedding-based microbial genome comparison using nucleotide language models such as DNABERT-2.

StrainVector provides a reproducible framework for converting **assembled genomic sequences** into fixed-length vector representations, storing them in FAISS-indexed vector databases, and performing alignment-free similarity searches for strain-level metagenomic analysis.

## Statement of Need

Traditional strain-level metagenomic analysis relies on computationally expensive alignment-based approaches like average nucleotide identity (ANI) calculation and read mapping. While effective, these methods require full read alignment against large reference databases and are sensitive to sequencing noise.

Recent advances in nucleotide transformers enable alignment-free sequence comparison using vector similarity metrics. However, no standardized pipeline exists for converting reference genomes into searchable embedding databases, performing chunked embedding for variable-length sequences, or cross-platform deployment of nucleotide transformers.

StrainVector addresses these gaps by providing modular, extensible infrastructure for embedding-based genomic analysis with support for macOS Apple Silicon (via MPS acceleration) and multiple embedding models.

## Key Features

- **Multi-model support**: DNABERT-2, DNABERT v1, Nucleotide Transformer via HuggingFace
- **Flexible chunking**: Configurable window size and stride for variable-length sequences
- **Vector search**: FAISS-based approximate nearest-neighbor search for large-scale databases
- **Cross-platform**: Native support for macOS Apple Silicon with MPS acceleration
- **Reproducible workflows**: Bootstrap scripts for reference indexing, profiling, and comparison
- **Extensible architecture**: Model-agnostic design for evaluating emerging nucleotide language models

## Installation

### Requirements
- Python 3.9 or higher
- 8GB+ RAM recommended for embedding large genomes
- GPU optional but recommended (CUDA or Metal/MPS)

### Install from source

```bash
git clone https://github.com/jacoblapkin/strainvector.git
cd strainvector
pip install -e .
```

### Dependencies

Core dependencies will be installed automatically:
- `torch>=2.0` - PyTorch for model inference
- `transformers>=4.30` - HuggingFace transformers
- `numpy>=1.20` - Array operations
- `faiss-cpu` or `faiss-gpu` - Vector search (CPU version installed by default)
- `tqdm` - Progress bars

For GPU support on NVIDIA hardware:
```bash
pip install faiss-gpu
```

For macOS with Apple Silicon, the pipeline automatically enables MPS acceleration.

## Input Requirements

**StrainVector requires assembled sequences in FASTA format:**
- ✅ **Supported**: Assembled genomes, contigs, or scaffolds (FASTA: `.fa`, `.fasta`, `.fna`, `.gz`)
- ❌ **Not supported**: Raw sequencing reads (FASTQ files)

**Preprocessing recommendations:**
- If you have raw metagenomic reads, assemble them first using tools like:
  - **metaSPAdes** or **MEGAHIT** for metagenomic assembly
  - **SPAdes** or **Unicycler** for isolate genomes
- Filter low-quality contigs (< 1kb recommended)
- The default window size is 3000-5000bp, so longer contigs provide better context for embedding

## Download Example Genomes

Use NCBI Entrez Direct to fetch assembled genomes for testing StrainVector with real strain-level data.

### 1. Install Entrez Direct

Entrez Direct is distributed through the Bioconda channel and cannot be installed via pip.

```bash
# Create a small environment for Entrez Direct
conda create -n edirect -c bioconda entrez-direct
conda activate edirect
```

### 2. Download Example Strain Genomes

**🧫 Example 1 — Bacteroides fragilis**

Fetch five assembled strain genomes and place them in `data/genomes/`:

```bash
mkdir -p data/genomes && cd data/genomes && \
esearch -db assembly -query "Bacteroides fragilis[organism]" \
| esummary \
| xtract -pattern DocumentSummary -element FtpPath_RefSeq \
| head -n 5 \
| while read -r line; do fname=$(basename "$line"); wget "$line/${fname}_genomic.fna.gz"; done && cd ../..
```

**🧬 Example 2 — Akkermansia muciniphila**

Add another five strains of a different species to the same folder:

```bash
cd data/genomes && \
esearch -db assembly -query "Akkermansia muciniphila[organism]" \
| esummary \
| xtract -pattern DocumentSummary -element FtpPath_RefSeq \
| head -n 5 \
| while read -r line; do fname=$(basename "$line"); wget "$line/${fname}_genomic.fna.gz"; done && cd ../..
```

After running both commands, verify your downloaded genomes:

```bash
ls data/genomes | grep .fna.gz
```

### 3. Build a Reference Database

Once downloaded, build your embedding reference index:

```bash
strainvector index --input data/genomes --out-db refdb/
```

**✅ Summary**
- **Tool**: `entrez-direct` (install via conda/bioconda)
- **Example species**: *Bacteroides fragilis* and *Akkermansia muciniphila*
- **Output directory**: `data/genomes/`
- **Ready for**: indexing with `strainvector index`

## Download an Assembled Metagenomic Sample

To demonstrate strain-level profiling, you can use a publicly available metagenomic assembly directly from the EMBL-EBI MGnify database. This avoids the need for SRA downloads or local assembly.

### Fetch a pre-assembled metagenome

The following command downloads a small gut metagenomic sample (MGYA00579260) and extracts it into `data/samples/`:

```bash
mkdir -p data/samples && \
curl -L -o data/samples/MGYA00579260_contigs.fasta.gz \
"https://www.ebi.ac.uk/metagenomics/api/v1/analyses/MGYA00579260/file/ERZ1643254_FASTA.fasta.gz" && \
gunzip -f data/samples/MGYA00579260_contigs.fasta.gz
```

After running this, you'll have:

```
data/samples/MGYA00579260_contigs.fasta
```

This FASTA file contains assembled contigs (e.g., `>NODE-1-length-996284-cov-12.35`) representing reconstructed genomic fragments from a real human gut microbiome.

## Quick Start

### Build a reference database

```bash
# Index reference genomes into vector database
strainvector index \
  --input reference_genomes/ \
  --out-db refdb/ \
  --model zhihan1996/DNABERT-2-117M \
  --window 3000 --stride 3000
```

This creates a searchable database in `refdb/` with embeddings, metadata, and FAISS index.

### Profile a metagenomic sample

```bash
# Profile sample against reference database
strainvector profile \
  --sample sample_contigs.fasta \
  --db refdb/ \
  --out results.json \
  --top-k 50 --similarity-threshold 0.85
```

Results include ranked candidate strains with similarity scores and abundance estimates.

### Compare genomes pairwise

```bash
# Compute pairwise similarity matrix with detailed window-level statistics
strainvector compare \
  --samples genome1.fa genome2.fa genome3.fa \
  --out compare.json \
  --out-tsv compare.tsv \
  --comparison-mode window

# Or use fast centroid mode for quick high-level comparison
strainvector compare \
  --samples genome1.fa genome2.fa genome3.fa \
  --out compare.json \
  --comparison-mode centroid
```

Outputs a similarity matrix for direct all-versus-all comparison. Window mode provides distribution statistics revealing strain-level variation.

## Documentation

- **Usage Guide**: See [`docs/USAGE.md`](docs/USAGE.md) for detailed CLI documentation
- **Bootstrap Scripts**: See [`scripts/`](scripts/) for automated workflow examples

## Example Workflow

```bash
# 1. Build reference database from bacterial genomes
bash scripts/bootstrap_index.sh

# 2. Profile a metagenomic sample
bash scripts/bootstrap_profile.sh

# 3. Compare multiple genomes
bash scripts/bootstrap_compare.sh
```

## Community Guidelines

### Contributing
We welcome contributions! See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for guidelines on:
- Development setup
- Code style
- Submitting pull requests

### Reporting Issues
Found a bug or have a feature request? Open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (OS, Python version)

### Getting Support
- **Usage questions**: Check [`docs/USAGE.md`](docs/USAGE.md) first
- **Need help?**: Open an issue with the "question" label
- **Discussions**: Use GitHub Issues for now; Discussions may be enabled later

## Citation

If you use StrainVector in your research, please cite:

```bibtex
@software{lapkin2025strainvector,
  author = {Lapkin, Jacob},
  title = {StrainVector: An Embedding-Based Framework for Strain-Level Metagenomic Analysis Using DNA Language Models},
  year = {2025},
  url = {https://github.com/jacoblapkin/strainvector}
}
```

A JOSS paper is in preparation. Once published, please cite the paper instead.

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## License

StrainVector is released under the [MIT License](LICENSE).

Copyright (c) 2025 Jacob Lapkin

## Acknowledgments

This project builds on:
- [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2) for nucleotide language models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for model inference

---

**Status**: Early development (v0.0.1). Contributions and feedback welcome!
