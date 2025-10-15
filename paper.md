---
title: 'StrainVector: An Embedding-Based Framework for Strain-Level Metagenomic Analysis Using DNA Language Models'
tags:
  - Python
  - metagenomics
  - embeddings
  - DNA language models
  - bioinformatics
  - DNABERT
  - vector search
authors:
  - name: Jacob Lapkin
    orcid: 0009-0002-6903-5925
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 13 January 2025
bibliography: paper.bib
---

# Summary

StrainVector is an open-source Python pipeline for embedding, indexing, and comparing microbial genomes using nucleotide language models such as DNABERT-2 [@zhou2023dnabert2]. It provides a reproducible framework for converting genomic sequences into fixed-length vector representations, storing them in FAISS-indexed [@johnson2019billion] vector databases, and performing alignment-free similarity searches. The pipeline includes bootstrap scripts for reference database construction, sample profiling against reference genomes, and pairwise genome comparison, with built-in support for macOS Apple Silicon via a novel Triton compatibility workaround.

# Statement of Need

Traditional strain-level metagenomic analysis relies on alignment-based approaches such as average nucleotide identity (ANI) calculation [@jain2018high], read mapping [@olm2021instrain], and reference-guided assembly. While effective, these methods are computationally expensive, requiring full read alignment against large reference databases, and are sensitive to sequencing noise and reference genome completeness [@pasolli2019extensive].

Recent advances in biological sequence modeling demonstrate that pretrained nucleotide transformers [@dalla2023nucleotide; @zhou2023dnabert2] can capture meaningful genomic context through learned embeddings. These models enable alignment-free sequence comparison using vector similarity metrics (cosine similarity, inner product, or L2 distance), with potential applications in rapid metagenomic profiling, novelty detection, and large-scale comparative genomics.

However, no standardized open-source pipeline exists for:

1. Converting reference genomes into searchable embedding databases
2. Performing chunked embedding and pooling for variable-length sequences
3. Cross-platform deployment of nucleotide transformers (particularly on macOS with MPS acceleration)
4. Reproducible profiling workflows integrating FAISS vector search

StrainVector addresses these gaps by providing modular, extensible infrastructure for embedding-based genomic analysis. While designed with strain-level comparison as a motivating goal, the framework is model-agnostic and serves as a foundation for evaluating emerging nucleotide language models on real-world metagenomic tasks. The tool is designed for researchers in computational biology, microbiome science, and metagenomics who need faster alternatives to alignment-based methods or wish to explore the application of modern language models to genomic data.

# Implementation

StrainVector is implemented in Python 3.9+ and consists of four main modules:

**Embeddings Module** (`strainvector/embeddings/`): Provides a factory interface for nucleotide language models via HuggingFace Transformers [@wolf2020transformers]. Currently supports DNABERT-2 [@zhou2023dnabert2], DNABERT v1, and Nucleotide Transformer [@dalla2023nucleotide] with configurable pooling strategies (mean, CLS token, max pooling) and precision modes (FP32, FP16, BF16). Includes platform detection logic that applies a fake Triton module shim for macOS compatibility, enabling MPS (Metal Performance Shaders) acceleration on Apple Silicon without requiring CUDA-specific dependencies.

**Indexing Module** (`strainvector/indexing/`): Handles reference database construction from FASTA input. Sequences are chunked into overlapping windows (configurable size and stride), embedded in batches, and stored as sharded NumPy arrays with aligned JSONL metadata. Optional FAISS index construction [@johnson2019billion] enables fast approximate nearest-neighbor search for large-scale databases.

**Profiling Module** (`strainvector/profiling/`): Embeds query sequences (metagenomic contigs or assembled genomes), retrieves top-k nearest neighbors from the reference database, and aggregates results by reference genome. Outputs include ranked candidate lists with similarity scores and per-query neighbor relationships in JSONL format. Supports downsampling strategies (rate-based, random fraction, or max window cap) for rapid exploratory analysis. Implements configurable variance-weighted aggregation modes (uniform, inverse, quadratic) to test whether emphasizing poorly-matching windows improves strain resolution—though empirical testing shows minimal impact with pretrained models due to narrow similarity distributions.

**Comparison Module** (`strainvector/compare/`): Supports two comparison modes for pairwise genome analysis without reference database construction. **Centroid mode** computes genome-level mean embeddings and calculates pairwise similarity matrices—fast and suitable for high-level taxonomic comparison. **Window mode** performs window-by-window comparison, computing similarity distributions (mean, median, standard deviation, percentiles) to reveal strain-level variation when models have sufficient resolution. Window mode identifies poorly-matching genomic regions corresponding to accessory genes or variable loci, though current pretrained models show limited strain discrimination (as with profile aggregation). Results are exported as JSON summaries with optional tab-separated matrices.

The pipeline includes three bootstrap scripts (`scripts/bootstrap_*.sh`) that automate environment setup, dependency installation, and end-to-end execution for indexing, profiling, and comparison workflows. All scripts support macOS-specific OpenMP and MPS compatibility settings.

# Example Usage

**Input Requirements**: StrainVector operates on assembled genomic sequences in FASTA format (genomes, contigs, or scaffolds). Raw sequencing reads (FASTQ) are not supported; users should assemble metagenomic samples using tools like metaSPAdes or MEGAHIT before profiling.

Reference database construction and sample profiling can be performed via command-line interface:

```bash
# Build reference database from FASTA genomes
strainvector index \
  --input genomes/ \
  --out-db refdb/ \
  --model zhihan1996/DNABERT-2-117M \
  --window 3000 --stride 3000

# Profile metagenomic sample against reference
strainvector profile \
  --sample contigs.fasta \
  --db refdb/ \
  --out results.json \
  --top-k 50 --similarity-threshold 0.85
```

Example pairwise comparison of eleven reference genomes (two bacterial: *E. coli*, *B. theta*; nine eukaryotic: primates and horse) demonstrates domain-level separation: bacterial genomes show 0.99 similarity to each other and ~0.74-0.75 to eukaryotes, while eukaryotic genomes cluster tightly at 0.999+ similarity. This illustrates that DNABERT-2 embeddings capture broad phylogenetic signal but collapse at finer taxonomic scales, motivating future work on fine-tuned or contrastively trained models.

# Current Limitations and Future Directions

At present, mean-pooled genome embeddings from pretrained DNABERT-2 show limited strain-level discrimination, as evidenced by near-identical similarity scores (0.999+) for closely related genomes. This reflects the pretraining objectives of existing nucleotide language models—which prioritize masked token prediction across diverse genomic contexts—rather than limitations of the pipeline architecture itself.

To investigate whether aggregation strategies could improve strain discrimination without model fine-tuning, we implemented variance-weighted pooling modes (inverse and quadratic weighting) that emphasize poorly-matching windows—hypothetically corresponding to strain-specific accessory genes or variable regions. Testing on *Bacteroides fragilis* reference strains (n=10) and a gut metagenomic sample revealed minimal practical impact: weighted mean similarity (0.882) differed by only ~0.4% from arithmetic mean (0.886). This narrow effect arises from three factors: (1) the similarity threshold (0.85) filters out highly variable regions before aggregation, leaving only conserved windows; (2) the remaining similarity distribution (0.85-0.96) provides insufficient variance for weighting to differentiate strains; and (3) closely related bacterial strains share 95-99% sequence identity genome-wide, with strain-specific differences concentrated in small accessory elements that contribute minimally to whole-genome averages. This empirical result reinforces that pretrained embeddings lack the resolution for strain-level tasks, independent of aggregation method, and validates the need for task-specific fine-tuning or contrastive learning objectives.

StrainVector's modular design positions it as a platform for evaluating future fine-tuned embeddings. Planned extensions include:

- Integration of contrastive learning approaches [@chen2020simple] with strain-labeled pairs
- Support for hybrid nucleotide-protein embeddings (e.g., combining DNABERT with ESM-3 [@hayes2024simulating])
- Benchmarking against alignment-based metrics (ANI, inStrain [@olm2021instrain]) on simulated and real metagenomic datasets

The framework's abstraction of model inference, chunking, and vector storage enables drop-in replacement of embedding models without modifying downstream indexing or profiling logic.

# References
