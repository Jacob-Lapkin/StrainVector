"""Reference DB build scaffolding.

Creates directory structure and writes a config manifest without
performing model inference or index construction. This keeps dependencies
light and allows users to customize parameters up-front.
"""

from __future__ import annotations

import gc
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Dict

from ..embeddings import get_embedder
from ..io.fasta import fasta_iter, chunk_sequence, ns_fraction
from .vector_store import VectorWriter
from ..util.logger import Logger, logger_for_refdb
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def _gather_inputs(root: Path, exts: Iterable[str]) -> List[Path]:
    if root.is_file():
        return [root]
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted({p.resolve() for p in files})


def init_refdb(
    out_dir: Path,
    input_path: Path,
    *,
    name: str,
    metadata_csv: Optional[Path],
    k: int,
    window: int,
    stride: int,
    min_contig_len: int,
    max_ns_frac: float,
    model_name: str,
    tokenizer_name: str | None,
    pooling: str,
    normalize: bool,
    precision: str,
    batch_size: int,
    num_workers: int,
    device: str,
    seed: int,
    index_backend: str,
    metric: str,
    faiss_index: str,
    shards: int,
    force: bool = False,
    log_file: Optional[Path] = None,
) -> Path:
    """Create a new reference DB directory and write a config manifest.

    Returns the path to the created reference DB directory.
    """
    if out_dir.exists():
        if any(out_dir.iterdir()) and not force:
            raise FileExistsError(
                f"Output directory {out_dir} exists and is not empty. Use --force to overwrite."
            )
        if force:
            shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Directory layout
    paths = {
        "vectors": out_dir / "vectors",
        "index": out_dir / "index",
        "metadata": out_dir / "metadata",
        "logs": out_dir / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # Gather inputs (record only; no reading yet)
    allowed_ext = [
        ".fa",
        ".fna",
        ".fasta",
        ".fa.gz",
        ".fna.gz",
        ".fasta.gz",
    ]
    inputs = [str(p) for p in _gather_inputs(input_path, allowed_ext)]

    # Write config manifest (JSON)
    config = {
        "name": name,
        "schema_version": "0.1",
        "input_paths": inputs,
        "metadata_csv": str(metadata_csv) if metadata_csv else None,
        "k": k,
        "window": window,
        "stride": stride,
        "min_contig_len": min_contig_len,
        "max_ns_frac": max_ns_frac,
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "pooling": pooling,
        "normalize": normalize,
        "precision": precision,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device,
        "seed": seed,
        "index_backend": index_backend,
        "metric": metric,
        "faiss_index": faiss_index,
        "shards": shards,
    }

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    # Write a minimal README describing contents and next steps
    readme = out_dir / "README.md"
    readme.write_text(
        (
            "# StrainVector Reference DB\n\n"
            "This directory was initialized by StrainVector.\n\n"
            "Contents:\n"
            "- config.json: Build configuration and parameters\n"
            "- vectors/: Placeholder for embedding vectors (to be generated)\n"
            "- index/: Placeholder for vector index (e.g., FAISS)\n"
            "- metadata/: Optional metadata files and manifests\n"
            "- logs/: Build logs\n\n"
            "Next steps: implement embedding generation and index construction.\n"
        ),
        encoding="utf-8",
    )

    # If metadata CSV provided, record a copy path (do not copy to avoid surprises)
    if metadata_csv:
        (paths["metadata"] / "METADATA_SOURCE.txt").write_text(
            f"Original metadata CSV: {metadata_csv}\n", encoding="utf-8"
        )

    # Log initialization
    try:
        logger = Logger(log_file) if log_file else logger_for_refdb(out_dir)
        logger.info(
            f"Initialized reference DB: name={name} inputs={len(inputs)} model={model_name}"
        )
    except Exception:
        pass

    return out_dir


def build_embeddings(
    out_dir: Path,
    *,
    input_paths: List[Path],
    k: int,
    window: int,
    stride: int,
    min_contig_len: int,
    max_ns_frac: float,
    model_name: str,
    tokenizer_name: Optional[str],
    pooling: str,
    normalize: bool,
    precision: str,
    batch_size: int,
    num_workers: int,
    device: str,
    shard_size: int = 50000,
    log_file: Optional[Path] = None,
) -> Dict[str, int]:
    logger = Logger(log_file) if log_file else logger_for_refdb(out_dir)
    vectors_dir = out_dir / "vectors"
    # Parallel per-file embedding when multiple workers/files
    if num_workers and num_workers > 1 and len(input_paths) > 1:
        total_enqueued = 0
        total_embedded = 0
        dim_any = 0
        args_list = []
        for i, fp in enumerate(input_paths):
            args_list.append((
                str(fp),
                str(vectors_dir),
                k,
                window,
                stride,
                min_contig_len,
                max_ns_frac,
                model_name,
                tokenizer_name,
                pooling,
                normalize,
                precision,
                batch_size,
                device,
                shard_size,
                f"part_w{i}",
            ))
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_embed_file_worker, a) for a in args_list]
            for fut in as_completed(futures):
                st = fut.result()
                total_enqueued += st.get("enqueued_windows", 0)
                total_embedded += st.get("embedded_vectors", 0)
                dim_any = dim_any or st.get("dim", 0)
        shard_count = len(list((out_dir / "vectors").glob("part_*.npy")))
        stats = {
            "enqueued_windows": int(total_enqueued),
            "embedded_vectors": int(total_embedded),
            "shards": int(shard_count),
            "dim": int(dim_any),
        }
        logger.info(
            f"Embedding complete (parallel): {stats['embedded_vectors']} vectors from {stats['enqueued_windows']} windows; dim={stats['dim']} shards={stats['shards']}"
        )
        logs_dir = out_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        with (logs_dir / "index_stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        return stats
    embedder = get_embedder(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        pooling=pooling,
        normalize=normalize,
        precision=precision,
        device=device,
    )

    vectors_dir = out_dir / "vectors"
    writer = VectorWriter(vectors_dir=vectors_dir, shard_prefix="part", shard_size=shard_size)

    batch_seqs: List[str] = []
    batch_meta: List[Dict] = []
    total_embedded = 0
    total_enqueued = 0

    def flush_batch():
        nonlocal batch_seqs, batch_meta
        if not batch_seqs:
            return
        vecs = embedder.embed(batch_seqs)
        writer.add(vecs, batch_meta)
        nonlocal total_embedded
        total_embedded += vecs.shape[0]
        batch_seqs = []
        batch_meta = []
        # Clear memory after each batch to prevent accumulation
        _clear_memory_caches()

    record_id = 0
    for inp in input_paths:
        for header, seq in fasta_iter(inp):
            if len(seq) < min_contig_len:
                continue
            # simple contig name parse
            contig_id = header.split()[0]
            for start, end, s in chunk_sequence(seq, window=window, stride=stride):
                if len(s) < window:
                    # allow last partial window
                    pass
                if ns_fraction(s) > max_ns_frac:
                    continue
                batch_seqs.append(s)
                batch_meta.append({
                    "record_id": record_id,
                    "contig": contig_id,
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "source": str(inp),
                    "source_basename": inp.name,
                    "k": k,
                    "window": window,
                    "stride": stride,
                    "model": model_name,
                })
                record_id += 1
                total_enqueued += 1
                if len(batch_seqs) >= batch_size:
                    flush_batch()

    flush_batch()
    writer.flush()
    shard_count = len(list((out_dir / "vectors").glob("part_*.npy")))
    stats = {
        "enqueued_windows": int(total_enqueued),
        "embedded_vectors": int(total_embedded),
        "shards": int(shard_count),
        "dim": int(getattr(embedder, "dim", 0)),
    }
    logger.info(
        f"Embedding complete: {stats['embedded_vectors']} vectors from {stats['enqueued_windows']} windows; "
        f"dim={stats['dim']} shards={stats['shards']}"
    )
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "index_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def _embed_file_worker(args) -> Dict[str, int]:
    (
        file_path,
        vectors_dir,
        k,
        window,
        stride,
        min_contig_len,
        max_ns_frac,
        model_name,
        tokenizer_name,
        pooling,
        normalize,
        precision,
        batch_size,
        device,
        shard_size,
        shard_prefix,
    ) = args
    from pathlib import Path as _Path

    file_path = _Path(file_path)
    vectors_dir = _Path(vectors_dir)
    local_embedder = get_embedder(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        pooling=pooling,
        normalize=normalize,
        precision=precision,
        device=device,
    )
    writer = VectorWriter(
        vectors_dir=vectors_dir, shard_prefix=shard_prefix, shard_size=shard_size
    )
    total_emb = 0
    total_enq = 0
    record_id = 0
    batch_seqs: List[str] = []
    batch_meta: List[Dict] = []

    def _flush():
        nonlocal batch_seqs, batch_meta, total_emb
        if not batch_seqs:
            return
        vecs = local_embedder.embed(batch_seqs)
        writer.add(vecs, batch_meta)
        total_emb += vecs.shape[0]
        batch_seqs = []
        batch_meta = []
        # Clear memory after each batch to prevent accumulation
        _clear_memory_caches()

    for header, seq in fasta_iter(file_path):
        if len(seq) < min_contig_len:
            continue
        contig_id = header.split()[0]
        for start, end, s in chunk_sequence(seq, window=window, stride=stride):
            if len(s) < window:
                pass
            if ns_fraction(s) > max_ns_frac:
                continue
            batch_seqs.append(s)
            batch_meta.append({
                "record_id": record_id,
                "contig": contig_id,
                "start": start,
                "end": end,
                "length": end - start,
                "source": str(file_path),
                "source_basename": file_path.name,
                "k": k,
                "window": window,
                "stride": stride,
                "model": model_name,
            })
            record_id += 1
            total_enq += 1
            if len(batch_seqs) >= batch_size:
                _flush()
    _flush()
    writer.flush()
    return {
        "enqueued_windows": total_enq,
        "embedded_vectors": total_emb,
        "shards": len(list(vectors_dir.glob(f"{shard_prefix}_*.npy"))),
        "dim": int(getattr(local_embedder, "dim", 0)),
    }
