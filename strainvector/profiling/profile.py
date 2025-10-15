from __future__ import annotations

import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..embeddings import get_embedder
from ..io.fasta import fasta_iter, chunk_sequence, ns_fraction
from .search import load_db_vectors, brute_force_topk, faiss_topk
from ..util.logger import logger_for_profile, Logger


def _clear_memory_caches():
    """Clear PyTorch device caches to free memory (MPS, CUDA, CPU)."""
    try:
        import torch
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def load_optional_metadata(db_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load optional metadata CSV referenced in DB metadata folder.

    Returns a mapping from source_basename -> row dict (best effort).
    """
    src_note = db_dir / "metadata" / "METADATA_SOURCE.txt"
    if not src_note.exists():
        return {}
    line = src_note.read_text(encoding="utf-8").strip()
    # Format: "Original metadata CSV: <path>"
    parts = line.split(":", 1)
    if len(parts) != 2:
        return {}
    csv_path = Path(parts[1].strip())
    if not csv_path.exists():
        return {}

    result: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row.get("file") or ""
            basename = Path(file_path).name if file_path else None
            if basename:
                result[basename] = row
    return result


def run_profile(
    sample_path: Path,
    db_dir: Path,
    *,
    top_k: int = 50,
    window: int = 5000,
    stride: int = 5000,
    min_contig_len: int = 1000,
    max_ns_frac: float = 0.1,
    model_name: str = "zhihan1996/DNABERT-2-117M",
    tokenizer_name: Optional[str] = None,
    pooling: str = "mean",
    normalize: bool = True,
    precision: str = "fp32",
    batch_size: int = 16,
    device: str = "auto",
    metric: str = "cosine",
    similarity_threshold: float = 0.85,
    aggregate_mode: str = "top1",
    weighting_mode: str = "uniform",
    downsample_rate: int = 1,
    max_windows: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    out_json: Path = Path("results.json"),
    out_neighbors: Optional[Path] = None,
    log_file: Optional[Path] = None,
) -> Path:
    logger = Logger(log_file) if log_file else logger_for_profile(out_json)
    # 1) Embed query windows
    embedder = get_embedder(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        pooling=pooling,
        normalize=normalize,
        precision=precision,
        device=device,
    )

    # Collect sample files (file or directory)
    allowed_ext = [".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz"]
    sample_files: List[Path] = []
    if sample_path.is_file():
        sample_files = [sample_path]
    else:
        for ext in allowed_ext:
            sample_files.extend(sorted(sample_path.rglob(f"*{ext}")))
        if not sample_files:
            raise FileNotFoundError(f"No FASTA files found under directory: {sample_path}")

    # chunk and embed in batches
    batch_seqs: List[str] = []
    batch_meta: List[Dict] = []
    query_vecs: List = []
    query_info: List[Dict] = []

    # Downsampling counters
    total_windows_seen = 0
    windows_processed = 0
    windows_skipped = 0

    # Random sampling setup
    import random
    if sample_fraction is not None:
        random.seed(42)  # reproducible sampling

    def flush_batch():
        nonlocal batch_seqs, batch_meta
        if not batch_seqs:
            return
        vecs = embedder.embed(batch_seqs)
        query_vecs.append(vecs)
        query_info.extend(batch_meta)
        batch_seqs = []
        batch_meta = []
        # Clear memory after each batch to prevent accumulation
        _clear_memory_caches()

    for inp in sample_files:
        for header, seq in fasta_iter(inp):
            if len(seq) < min_contig_len:
                continue
            contig_id = header.split()[0]
            for start, end, s in chunk_sequence(seq, window=window, stride=stride):
                if ns_fraction(s) > max_ns_frac:
                    continue

                total_windows_seen += 1

                # Apply downsampling strategies
                # 1. Max windows cap (highest priority - hard stop)
                if max_windows is not None and windows_processed >= max_windows:
                    break

                # 2. Rate-based downsampling (skip N-1 out of N windows)
                if downsample_rate > 1 and (total_windows_seen % downsample_rate) != 0:
                    windows_skipped += 1
                    continue

                # 3. Fraction-based random sampling
                if sample_fraction is not None and random.random() > sample_fraction:
                    windows_skipped += 1
                    continue

                # Window passed all filters - process it
                windows_processed += 1
                batch_seqs.append(s)
                batch_meta.append({
                    "contig": contig_id,
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "source": str(inp),
                    "source_basename": inp.name,
                })
                if len(batch_seqs) >= batch_size:
                    flush_batch()

            # Break outer loop if max_windows reached
            if max_windows is not None and windows_processed >= max_windows:
                break
    flush_batch()

    # Log downsampling stats
    if windows_skipped > 0:
        logger.info(
            f"Downsampling: processed={windows_processed} skipped={windows_skipped} total_seen={total_windows_seen}"
        )

    if not query_vecs:
        raise ValueError("No query windows produced; adjust window/stride/min_contig_len.")

    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Profiling requires 'numpy'.") from e
    Q = np.concatenate(query_vecs, axis=0)

    # 2) Load DB vectors and metadata
    X, db_meta = load_db_vectors(db_dir / "vectors")
    # 3) Search top-k via FAISS (if available) or brute force
    index_path = db_dir / "index" / "faiss.index"
    if index_path.exists():
        idxs, scrs = faiss_topk(index_path, Q, top_k, metric=metric)
    else:
        idxs, scrs = brute_force_topk(Q, X, top_k, metric=metric)

    # Define weighting function based on mode
    def compute_weight(similarity: float, mode: str) -> float:
        """Compute aggregation weight for a similarity score.

        Args:
            similarity: Cosine similarity score (0-1)
            mode: Weighting strategy
                - 'uniform': All windows weighted equally (weight=1)
                - 'inverse': Emphasize differences (weight = 1 - similarity)
                - 'quadratic': Strongly emphasize differences (weight = (1 - similarity)^2)

        Returns:
            Weight value (higher = more important for aggregation)
        """
        if mode == "inverse":
            # Linear weighting: windows with lower similarity get higher weight
            # Intuition: variable/accessory regions match poorly, get emphasized
            return 1.0 - similarity
        elif mode == "quadratic":
            # Quadratic weighting: even stronger emphasis on differences
            # Useful for highly conserved genomes where small differences matter
            diff = 1.0 - similarity
            return diff * diff
        else:  # uniform (default)
            return 1.0

    # 4) Aggregate by source_basename and attach optional metadata
    meta_map = load_optional_metadata(db_dir)
    agg = defaultdict(lambda: {
        "count": 0,
        "sum_similarity": 0.0,
        "sum_weighted_similarity": 0.0,
        "sum_weights": 0.0,
        "max_similarity": 0.0,
    })

    neighbors_path = out_neighbors or out_json.with_name("neighbors.jsonl")
    with neighbors_path.open("w", encoding="utf-8") as neigh_f:
        for i, (nbr_idx, nbr_scr) in enumerate(zip(idxs, scrs)):
            qmeta = query_info[i]
            selected_neighbors = []
            mode = aggregate_mode if aggregate_mode in ("top1", "sum") else "top1"
            for j, s in zip(nbr_idx, nbr_scr):
                if float(s) < similarity_threshold:
                    continue
                m = db_meta[j]
                ref_id = m.get("source_basename", m.get("source", "unknown"))
                selected_neighbors.append({
                    "db_index": int(j),
                    "similarity": float(s),
                    "source_basename": ref_id,
                    "contig": m.get("contig"),
                    "start": m.get("start"),
                    "end": m.get("end"),
                    "length": m.get("length"),
                })
                # aggregate with weighting
                similarity = float(s)
                weight = compute_weight(similarity, weighting_mode)
                agg[ref_id]["count"] += 1
                agg[ref_id]["sum_similarity"] += similarity
                agg[ref_id]["sum_weighted_similarity"] += similarity * weight
                agg[ref_id]["sum_weights"] += weight
                if similarity > agg[ref_id]["max_similarity"]:
                    agg[ref_id]["max_similarity"] = similarity
                if mode == "top1":
                    break
            neigh_f.write(json.dumps({
                "query": qmeta,
                "neighbors": selected_neighbors,
            }) + "\n")

    # 5) Build summary JSON
    candidates = []
    for ref_id, stats in agg.items():
        count = int(stats["count"])
        mean_sim = (stats["sum_similarity"] / max(1, count)) if count else 0.0

        # Compute weighted mean similarity
        if stats["sum_weights"] > 0:
            weighted_mean_sim = stats["sum_weighted_similarity"] / stats["sum_weights"]
        else:
            weighted_mean_sim = mean_sim

        # Use weighted mean for hybrid score to emphasize variable regions
        hybrid_score = count * weighted_mean_sim if weighting_mode != "uniform" else count * mean_sim

        out_row = {
            "id": ref_id,
            "count": count,
            "mean_similarity": mean_sim,
            "weighted_mean_similarity": weighted_mean_sim,
            "max_similarity": float(stats["max_similarity"]),
            "hybrid_score": hybrid_score,
        }
        md = meta_map.get(ref_id)
        if md:
            out_row["metadata"] = md
        candidates.append(out_row)

    # sort by hybrid_score (count * mean_similarity) to prioritize abundant + similar strains
    candidates.sort(key=lambda r: r["hybrid_score"], reverse=True)

    summary = {
        "db_path": str(db_dir),
        "total_query_chunks": len(query_info),
        "total_windows_seen": total_windows_seen,
        "windows_processed": windows_processed,
        "windows_skipped": windows_skipped,
        "downsample_rate": downsample_rate if downsample_rate > 1 else None,
        "max_windows": max_windows,
        "sample_fraction": sample_fraction,
        "top_k": top_k,
        "metric": metric,
        "aggregate_mode": aggregate_mode,
        "weighting_mode": weighting_mode,
        "similarity_threshold": similarity_threshold,
        "candidates": candidates,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # Write stats log
    logger.info(
        f"Profile complete: query_chunks={len(query_info)} candidates={len(candidates)} neighbors_per_chunk={top_k} metric={metric}"
    )
    # Persist stats JSON
    stats = {
        "query_chunks": len(query_info),
        "neighbors_per_chunk": top_k,
        "metric": metric,
        "candidates": len(candidates),
        "db_path": str(db_dir),
    }
    with out_json.with_name("profile_stats.json").open("w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)
    return out_json
