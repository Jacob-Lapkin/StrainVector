# Testing StrainVector

This directory contains automated tests and manual testing procedures for verifying StrainVector functionality.

## Automated Tests

### Running Basic Tests

Basic structural tests verify that modules can be imported and core components are accessible:

```bash
cd tests
python test_basic.py
```

These tests check:
- Module imports (embeddings, indexing, profiling, compare, io)
- Embeddings factory structure
- Indexing module functions
- I/O module functions

### Future Test Coverage

Planned additional test areas:
- Embedding computation accuracy
- FASTA parsing and chunking
- Vector database I/O
- FAISS index construction
- Profiling aggregation logic
- CLI argument parsing

## Manual Testing Procedures

For comprehensive validation, run the bootstrap scripts with example data:

### 1. Test Reference Database Construction

```bash
cd /path/to/strainvector
bash scripts/bootstrap_index.sh
```

**Expected outcome:**
- Creates `refdb/` directory
- Generates vector embeddings in `refdb/vectors/`
- Builds FAISS index in `refdb/index/`
- Produces `refdb/config.json` and `refdb/logs/index_stats.json`

**Verify:**
```bash
ls -lh refdb/vectors/  # Should contain .npy and .jsonl files
ls -lh refdb/index/    # Should contain faiss.index
cat refdb/logs/index_stats.json  # Check vector count and dimensions
```

### 2. Test Sample Profiling

```bash
bash scripts/bootstrap_profile.sh
```

**Expected outcome:**
- Creates `profiles/` directory with results
- Generates `results.json` with ranked candidate strains
- Produces `neighbors.jsonl` with per-query matches
- Creates `profile_stats.json` with summary statistics

**Verify:**
```bash
cat profiles/*/results.json | head -20  # Check candidate rankings
grep -c "source_basename" profiles/*/neighbors.jsonl  # Count neighbors found
```

### 3. Test Pairwise Comparison

```bash
bash scripts/bootstrap_compare.sh
```

**Expected outcome:**
- Generates similarity matrix in JSON format
- Creates TSV matrix file for inspection
- Shows domain-level separation (bacterial vs. eukaryotic genomes)

**Verify:**
```bash
cat compare.json  # Check similarity scores
cat compare.tsv   # View matrix in tabular format
```

### 4. Test Cross-Platform Compatibility

**macOS with Apple Silicon:**
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
# Should print: MPS available: True
```

**Linux with CUDA:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True (if GPU present)
```

### 5. Test CLI Commands

**Test help output:**
```bash
strainvector --help
strainvector index --help
strainvector profile --help
strainvector compare --help
```

**Test with minimal data:**
```bash
# Create a tiny test genome
echo ">test_genome" > test.fa
echo "ATCGATCGATCGATCGATCGATCG" >> test.fa

# Index it
strainvector index --input test.fa --out-db test_db/ --window 10 --stride 10

# Verify output
ls -lh test_db/
cat test_db/config.json
```

## Reporting Test Failures

If any tests fail, please [open an issue](https://github.com/jacoblapkin/strainvector/issues) with:
- Test command that failed
- Error message or unexpected output
- System information (OS, Python version, dependencies)
- Output of `pip list | grep -E "(torch|transformers|faiss)"`

