from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..embeddings import get_embedder
from ..io.fasta import fasta_iter, chunk_sequence, ns_fraction
from ..util.logger import Logger


def _collect_sample_files(root: Path) -> List[Path]:
    allowed_ext = [".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz"]
    if root.is_file():
        return [root]
    files: List[Path] = []
    for ext in allowed_ext:
        files.extend(sorted(root.rglob(f"*{ext}")))
    if not files:
        raise FileNotFoundError(f"No FASTA files found under: {root}")
    return files


def _embed_sample(
    sample: Path,
    *,
    window: int,
    stride: int,
    min_contig_len: int,
    max_ns_frac: float,
    batch_size: int,
    embedder,
    logger,
    downsample_rate: int = 1,
    max_windows: Optional[int] = None,
    sample_fraction: Optional[float] = None,
) -> Tuple["np.ndarray", Dict]:
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise ImportError("Compare requires 'numpy'.") from e

    seqs: List[str] = []
    contigs_processed = 0
    chunks_collected = 0

    # Downsampling counters
    total_windows_seen = 0
    windows_processed = 0
    windows_skipped = 0

    # Random sampling setup
    import random
    if sample_fraction is not None:
        random.seed(42)  # reproducible sampling

    for header, seq in fasta_iter(sample):
        if len(seq) < min_contig_len:
            continue
        contigs_processed += 1
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
            seqs.append(s)
            chunks_collected += 1
            # Log progress every 1000 chunks
            if chunks_collected % 1000 == 0:
                logger.info(
                    f"Sample {sample.name}: collected {chunks_collected} chunks from {contigs_processed} contigs..."
                )

        # Break outer loop if max_windows reached
        if max_windows is not None and windows_processed >= max_windows:
            break

    # Log downsampling stats
    if windows_skipped > 0:
        logger.info(
            f"Sample {sample.name}: downsampling - processed={windows_processed} skipped={windows_skipped} total_seen={total_windows_seen}"
        )

    if not seqs:
        return np.empty((0, embedder.dim), dtype=np.float32), {"chunks": 0}

    vecs: List["np.ndarray"] = []
    total_batches = (len(seqs) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(seqs), batch_size), start=1):
        batch = seqs[i : i + batch_size]
        vecs.append(embedder.embed(batch))
        # Log every 100 batches or on the last batch
        if batch_idx % 100 == 0 or batch_idx == total_batches:
            vectors_so_far = sum(v.shape[0] for v in vecs)
            logger.info(
                f"Sample {sample.name}: embedded batch {batch_idx}/{total_batches} ({vectors_so_far} vectors)"
            )
    X = np.concatenate(vecs, axis=0)
    return X, {
        "chunks": X.shape[0],
        "dim": X.shape[1] if X.size else embedder.dim,
        "batches": total_batches,
        "total_windows_seen": total_windows_seen,
        "windows_processed": windows_processed,
        "windows_skipped": windows_skipped,
    }


def run_compare(
    samples: List[Path],
    *,
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
    comparison_mode: str = "centroid",
    downsample_rate: int = 1,
    max_windows: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    out_json: Path = Path("compare.json"),
    out_tsv: Optional[Path] = None,
    log_file: Optional[Path] = None,
) -> Path:
    logger = Logger(log_file) if log_file else Logger(out_json.with_suffix(".log"))
    logger.info(
        f"Compare start: n_inputs={len(samples)} mode={comparison_mode} metric={metric} window={window} stride={stride}"
    )

    embedder = get_embedder(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        pooling=pooling,
        normalize=normalize,
        precision=precision,
        device=device,
    )

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise ImportError("Compare requires 'numpy'.") from e

    sample_files: List[Path] = []
    for s in samples:
        sample_files.extend(_collect_sample_files(s))

    labels: List[str] = []
    embeddings: List["np.ndarray"] = []  # Store full embeddings for window mode
    stats: Dict[str, Dict] = {}
    for idx, f in enumerate(sample_files, start=1):
        import time
        start_time = time.time()

        logger.info(f"Processing sample {idx}/{len(sample_files)}: {f.name}")

        X, st = _embed_sample(
            f,
            window=window,
            stride=stride,
            min_contig_len=min_contig_len,
            max_ns_frac=max_ns_frac,
            batch_size=batch_size,
            embedder=embedder,
            logger=logger,
            downsample_rate=downsample_rate,
            max_windows=max_windows,
            sample_fraction=sample_fraction,
        )

        elapsed = time.time() - start_time
        labels.append(f.name)
        stats[f.name] = st
        embeddings.append(X)

        batches = st.get('batches', 0)
        logger.info(
            f"Embedded {f.name} -> chunks={st['chunks']}, batches={batches}, time={elapsed:.1f}s"
        )

    # Compute similarity matrix based on comparison mode
    if comparison_mode == "centroid":
        # Original centroid-based approach
        centroids: List["np.ndarray"] = []
        for X in embeddings:
            if X.shape[0] == 0:
                centroids.append(np.zeros((embedder.dim,), dtype=np.float32))
            else:
                c = X.mean(axis=0)
                if metric == "cosine":
                    c = c / (np.linalg.norm(c) + 1e-12)
                centroids.append(c.astype(np.float32))

        C = np.stack(centroids, axis=0)
        if metric == "cosine" or metric == "ip":
            S = C @ C.T
        else:
            # negative squared distance as score
            q2 = (C ** 2).sum(axis=1, keepdims=True)
            b2 = q2.T
            S = -(q2 + b2 - 2.0 * (C @ C.T))

        summary = {
            "labels": labels,
            "metric": metric,
            "comparison_mode": comparison_mode,
            "matrix": S.tolist(),
            "stats": stats,
        }

    elif comparison_mode == "window":
        # Window-by-window comparison with distribution statistics
        logger.info("Computing window-by-window similarity distributions...")
        n_samples = len(embeddings)
        S = np.zeros((n_samples, n_samples), dtype=np.float32)
        distributions = {}

        # For cosine similarity, ensure embeddings are normalized
        if metric == "cosine":
            normalized_embeddings = []
            for idx, X in enumerate(embeddings):
                if X.shape[0] > 0:
                    norms = np.linalg.norm(X, axis=1, keepdims=True)
                    X_norm = X / (norms + 1e-12)
                    normalized_embeddings.append(X_norm.astype(np.float32))
                else:
                    normalized_embeddings.append(X)
            embeddings = normalized_embeddings

        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    # Self-similarity is 1.0
                    S[i, j] = 1.0
                    distributions[f"{labels[i]}__vs__{labels[j]}"] = {
                        "mean": 1.0,
                        "median": 1.0,
                        "std": 0.0,
                        "min": 1.0,
                        "max": 1.0,
                        "percentiles": {"10": 1.0, "25": 1.0, "50": 1.0, "75": 1.0, "90": 1.0},
                        "below_threshold": {"0.95": 0.0, "0.90": 0.0, "0.85": 0.0, "0.80": 0.0},
                    }
                else:
                    X_i, X_j = embeddings[i], embeddings[j]
                    if X_i.shape[0] == 0 or X_j.shape[0] == 0:
                        S[i, j] = S[j, i] = 0.0
                        continue

                    # Compute pairwise similarity matrix between all windows
                    # For memory efficiency, compute in chunks if needed
                    logger.info(f"  Comparing {labels[i]} ({X_i.shape[0]} windows) vs {labels[j]} ({X_j.shape[0]} windows)")

                    if metric == "cosine" or metric == "ip":
                        # Similarity matrix: shape [n_windows_i, n_windows_j]
                        # Note: embeddings are pre-normalized for cosine above
                        sim_matrix = X_i @ X_j.T
                    else:  # L2
                        # Compute negative squared L2 distance
                        q2 = (X_i ** 2).sum(axis=1, keepdims=True)
                        b2 = (X_j ** 2).sum(axis=1, keepdims=True).T
                        sim_matrix = -(q2 + b2 - 2.0 * (X_i @ X_j.T))

                    # For each window in sample i, find best match in sample j
                    best_matches_i = sim_matrix.max(axis=1)
                    # For each window in sample j, find best match in sample i
                    best_matches_j = sim_matrix.max(axis=0)
                    # Combine both directions for a symmetric measure
                    all_best_matches = np.concatenate([best_matches_i, best_matches_j])

                    # Compute statistics
                    mean_sim = float(all_best_matches.mean())
                    median_sim = float(np.median(all_best_matches))
                    std_sim = float(all_best_matches.std())
                    min_sim = float(all_best_matches.min())
                    max_sim = float(all_best_matches.max())

                    percentiles = np.percentile(all_best_matches, [10, 25, 50, 75, 90])
                    below_thresholds = {
                        "0.95": float((all_best_matches < 0.95).sum() / len(all_best_matches) * 100),
                        "0.90": float((all_best_matches < 0.90).sum() / len(all_best_matches) * 100),
                        "0.85": float((all_best_matches < 0.85).sum() / len(all_best_matches) * 100),
                        "0.80": float((all_best_matches < 0.80).sum() / len(all_best_matches) * 100),
                    }

                    # Store mean similarity in matrix
                    S[i, j] = S[j, i] = mean_sim

                    # Store detailed distribution
                    distributions[f"{labels[i]}__vs__{labels[j]}"] = {
                        "mean": mean_sim,
                        "median": median_sim,
                        "std": std_sim,
                        "min": min_sim,
                        "max": max_sim,
                        "percentiles": {
                            "10": float(percentiles[0]),
                            "25": float(percentiles[1]),
                            "50": float(percentiles[2]),
                            "75": float(percentiles[3]),
                            "90": float(percentiles[4]),
                        },
                        "below_threshold_pct": below_thresholds,
                        "n_windows_i": int(X_i.shape[0]),
                        "n_windows_j": int(X_j.shape[0]),
                    }

                    logger.info(
                        f"    Mean similarity: {mean_sim:.4f} | Median: {median_sim:.4f} | "
                        f"Std: {std_sim:.4f} | <0.90: {below_thresholds['0.90']:.1f}%"
                    )

        summary = {
            "labels": labels,
            "metric": metric,
            "comparison_mode": comparison_mode,
            "matrix": S.tolist(),
            "distributions": distributions,
            "stats": stats,
        }
    else:
        raise ValueError(f"Unknown comparison_mode: {comparison_mode}. Use 'centroid' or 'window'.")

    # Write JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write TSV if requested
    if out_tsv:
        with out_tsv.open("w", encoding="utf-8") as tf:
            tf.write("sample\t" + "\t".join(labels) + "\n")
            for i, lab in enumerate(labels):
                row = "\t".join(f"{S[i,j]:.6f}" for j in range(len(labels)))
                tf.write(f"{lab}\t{row}\n")

    logger.info(f"Compare done: n_samples={len(labels)} | wrote {out_json}")
    return out_json

