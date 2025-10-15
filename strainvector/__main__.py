"""CLI entry point (scaffold).

Uses stdlib argparse to avoid early dependency commitments.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional
from pathlib import Path

from .indexing.build import init_refdb, build_embeddings
from .indexing.faiss_index import build_faiss_index
from .profiling.profile import run_profile
from .compare.compare import run_compare


def cmd_index(args: argparse.Namespace) -> int:
    out_dir = args.out_db
    out_dir = out_dir if isinstance(out_dir, Path) else Path(out_dir)

    try:
        out_created = init_refdb(
            out_dir=out_dir,
            input_path=args.input,
            name=args.name,
            metadata_csv=args.metadata,
            k=args.k,
            window=args.window,
            stride=args.stride,
            min_contig_len=args.min_contig_len,
            max_ns_frac=args.max_ns_frac,
            model_name=args.model,
            tokenizer_name=args.tokenizer,
            pooling=args.pooling,
            normalize=not args.no_normalize,
            precision=args.precision,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            index_backend=args.index_backend,
            metric=args.metric,
            faiss_index=args.faiss_index,
            shards=args.shards,
            force=args.force,
            log_file=args.log_file,
        )
        print(f"Initialized reference DB at: {out_dir}")
        if not args.no_embed:
            print("Building embeddings...")
            # Build list of inputs from the recorded config (re-scan input path)
            allowed_ext = [
                ".fa",
                ".fna",
                ".fasta",
                ".fa.gz",
                ".fna.gz",
                ".fasta.gz",
            ]
            input_paths = []
            if args.input.is_file():
                input_paths = [args.input]
            else:
                for ext in allowed_ext:
                    input_paths.extend(sorted(args.input.rglob(f"*{ext}")))
            stats = build_embeddings(
                out_dir=out_dir,
                input_paths=input_paths,
                k=args.k,
                window=args.window,
                stride=args.stride,
                min_contig_len=args.min_contig_len,
                max_ns_frac=args.max_ns_frac,
                model_name=args.model,
                tokenizer_name=args.tokenizer,
                pooling=args.pooling,
                normalize=not args.no_normalize,
                precision=args.precision,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                shard_size=args.shard_size,
                log_file=args.log_file,
            )
            print(
                f"Embeddings: {stats['embedded_vectors']} vectors from {stats['enqueued_windows']} windows; dim={stats['dim']} shards={stats['shards']}"
            )
        if not args.no_index and args.index_backend == "faiss":
            vectors_dir = out_dir / "vectors"
            has_shards = any(vectors_dir.glob("part_*.npy"))
            if not has_shards:
                print("No vector shards found; skipping FAISS index.")
            else:
                print("Building FAISS index...")
                idx_path, nvecs, dim = build_faiss_index(
                    vectors_dir=vectors_dir,
                    index_dir=out_dir / "index",
                    metric=args.metric,
                    index_type=args.faiss_index,
                )
                print(f"FAISS index written: {idx_path} (nvecs={nvecs}, dim={dim})")
                # Persist FAISS stats
                try:
                    import json
                    logs_dir = (out_dir / "logs")
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    with (logs_dir / "faiss_stats.json").open("w", encoding="utf-8") as f:
                        json.dump({"index_path": str(idx_path), "nvecs": int(nvecs), "dim": int(dim)}, f, indent=2)
                except Exception:
                    pass
        elif not args.no_index:
            print(f"Index backend '{args.index_backend}' not implemented yet; skipping.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_profile(args: argparse.Namespace) -> int:
    try:
        # macOS memory optimization: reduce batch size for memory efficiency
        import platform
        batch_size = args.batch_size
        if platform.system() == "Darwin" and batch_size == 16:
            batch_size = 4
            print(f"[StrainVector] macOS detected; reducing batch_size to {batch_size} for memory efficiency.")

        out_json = run_profile(
            sample_path=args.sample,
            db_dir=args.db,
            top_k=args.top_k,
            window=args.window,
            stride=args.stride,
            min_contig_len=args.min_contig_len,
            max_ns_frac=args.max_ns_frac,
            model_name=args.model,
            tokenizer_name=args.tokenizer,
            pooling=args.pooling,
            normalize=not args.no_normalize,
            precision=args.precision,
            batch_size=batch_size,
            device=args.device,
            metric=args.metric,
            out_json=args.out,
            similarity_threshold=args.similarity_threshold,
            aggregate_mode=args.aggregate_mode,
            weighting_mode=args.weighting_mode,
            downsample_rate=args.downsample_rate,
            max_windows=args.max_windows,
            sample_fraction=args.sample_fraction,
            log_file=args.log_file,
        )
        print(f"Wrote profile summary: {out_json}")
        print(f"Neighbors written to: {out_json.with_name('neighbors.jsonl')}")
        try:
            import json
            stats_path = out_json.with_name("profile_stats.json")
            if stats_path.exists():
                with stats_path.open("r", encoding="utf-8") as f:
                    s = json.load(f)
                print(
                    f"Profile stats: query_chunks={s.get('query_chunks')} candidates={s.get('candidates')} metric={s.get('metric')}"
                )
        except Exception:
            pass
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strainvector",
        description=(
            "Embedding-based framework for strain-level metagenomic comparison and profiling"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # index subcommand
    p_index = subparsers.add_parser(
        "index", help="Build a reference embedding database from genomes"
    )
    p_index.add_argument("--input", required=True, type=Path, help="Input FASTA/dir")
    p_index.add_argument("--out-db", required=True, type=Path, help="Output DB dir")
    p_index.add_argument("--name", type=str, default="strainvector-refdb", help="Reference DB name")
    p_index.add_argument("--metadata", type=Path, help="Optional CSV with genome metadata")
    p_index.add_argument("--k", type=int, default=6, help="k-mer size (default: 6)")
    p_index.add_argument("--window", type=int, default=5000, help="Chunk/window size in bp")
    p_index.add_argument("--stride", type=int, default=5000, help="Stride between windows (bp)")
    p_index.add_argument("--min-contig-len", type=int, default=1000, help="Minimum contig length to include")
    p_index.add_argument("--max-ns-frac", type=float, default=0.1, help="Maximum fraction of Ns allowed per window")

    p_index.add_argument("--model", type=str, default="zhihan1996/DNABERT-2-117M", help="HF model name for embeddings")
    p_index.add_argument("--tokenizer", type=str, help="HF tokenizer name (optional)")
    p_index.add_argument("--pooling", type=str, choices=["mean", "cls", "max"], default="mean", help="Pooling strategy")
    p_index.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization")
    p_index.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Compute precision")

    p_index.add_argument("--batch-size", type=int, default=16)
    p_index.add_argument("--num-workers", type=int, default=4)
    p_index.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    p_index.add_argument("--seed", type=int, default=42)
    p_index.add_argument(
        "--shard-size",
        type=int,
        default=50000,
        help="Vectors per shard (approx memory ~= shard_size * dim * 4 bytes)",
    )

    p_index.add_argument("--index-backend", type=str, choices=["faiss", "pgvector", "milvus", "flat"], default="faiss")
    p_index.add_argument("--metric", type=str, choices=["cosine", "ip", "l2"], default="cosine")
    p_index.add_argument("--faiss-index", type=str, default="Flat", help="FAISS index type (e.g., Flat, IVF1024,IVF-PQ)")
    p_index.add_argument("--shards", type=int, default=1, help="Number of index shards")

    p_index.add_argument("--force", action="store_true", help="Overwrite existing out-db directory if present")
    p_index.add_argument("--no-embed", action="store_true", help="Skip embedding generation (init only)")
    p_index.add_argument("--no-index", action="store_true", help="Skip index construction")
    p_index.add_argument("--log-file", type=Path, help="Write index logs to this file instead of default location")
    p_index.set_defaults(func=cmd_index)

    # profile subcommand
    p_profile = subparsers.add_parser(
        "profile", help="Profile a sample against the reference DB"
    )
    p_profile.add_argument("--sample", required=True, type=Path, help="Sample FASTA or directory")
    p_profile.add_argument("--db", required=True, type=Path, help="Reference DB dir")
    p_profile.add_argument("--out", required=True, type=Path, help="Output JSON path for summary")
    p_profile.add_argument("--top-k", type=int, default=50, help="Top-k neighbors per chunk")

    p_profile.add_argument("--window", type=int, default=5000, help="Chunk/window size in bp")
    p_profile.add_argument("--stride", type=int, default=5000, help="Stride between windows (bp)")
    p_profile.add_argument("--min-contig-len", type=int, default=1000, help="Minimum contig length to include")
    p_profile.add_argument("--max-ns-frac", type=float, default=0.1, help="Maximum fraction of Ns allowed per window")

    p_profile.add_argument("--model", type=str, default="zhihan1996/DNABERT-2-117M", help="HF model name for embeddings")
    p_profile.add_argument("--tokenizer", type=str, help="HF tokenizer name (optional)")
    p_profile.add_argument("--pooling", type=str, choices=["mean", "cls", "max"], default="mean", help="Pooling strategy")
    p_profile.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization")
    p_profile.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Compute precision")
    p_profile.add_argument("--batch-size", type=int, default=16)
    p_profile.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    p_profile.add_argument("--metric", type=str, choices=["cosine", "ip", "l2"], default="cosine")
    p_profile.add_argument("--similarity-threshold", type=float, default=0.85, help="Only count neighbors with similarity >= this value")
    p_profile.add_argument("--aggregate-mode", type=str, choices=["top1", "sum"], default="top1", help="Aggregate per-chunk: only top1 match or sum all passing threshold")
    p_profile.add_argument("--weighting-mode", type=str, choices=["uniform", "inverse", "quadratic"], default="uniform",
                          help="Weighting strategy for aggregation: uniform (default, equal weights), inverse (weight=1-similarity, emphasizes differences), quadratic (weight=(1-similarity)^2, strongly emphasizes differences)")

    # Downsampling options for speed optimization
    p_profile.add_argument("--downsample-rate", type=int, default=1, help="Process 1 out of every N windows (default: 1, no downsampling)")
    p_profile.add_argument("--max-windows", type=int, help="Maximum number of windows to process (cap total)")
    p_profile.add_argument("--sample-fraction", type=float, help="Randomly sample this fraction of windows (e.g., 0.2 for 20%%)")

    p_profile.add_argument("--log-file", type=Path, help="Write profile logs to this file instead of default location")
    p_profile.set_defaults(func=cmd_profile)

    # compare subcommand
    p_compare = subparsers.add_parser(
        "compare", help="Compare multiple samples via centroid embeddings"
    )
    p_compare.add_argument(
        "--samples", nargs="+", type=Path, required=True, help="Sample FASTA files or directories"
    )
    p_compare.add_argument("--out", required=True, type=Path, help="Output JSON matrix path")
    p_compare.add_argument("--out-tsv", type=Path, help="Optional TSV matrix path")
    p_compare.add_argument("--window", type=int, default=5000, help="Chunk/window size in bp")
    p_compare.add_argument("--stride", type=int, default=5000, help="Stride between windows (bp)")
    p_compare.add_argument("--min-contig-len", type=int, default=1000, help="Minimum contig length to include")
    p_compare.add_argument("--max-ns-frac", type=float, default=0.1, help="Maximum fraction of Ns allowed per window")
    p_compare.add_argument("--model", type=str, default="zhihan1996/DNABERT-2-117M", help="HF model name for embeddings")
    p_compare.add_argument("--tokenizer", type=str, help="HF tokenizer name (optional)")
    p_compare.add_argument("--pooling", type=str, choices=["mean", "cls", "max"], default="mean", help="Pooling strategy")
    p_compare.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization")
    p_compare.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Compute precision")
    p_compare.add_argument("--batch-size", type=int, default=16)
    p_compare.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    p_compare.add_argument("--metric", type=str, choices=["cosine", "ip", "l2"], default="cosine")
    p_compare.add_argument("--comparison-mode", type=str, choices=["centroid", "window"], default="centroid",
                          help="Comparison mode: centroid (average all windows, fast) or window (window-by-window with distribution stats, slower but more detailed)")

    # Downsampling options for speed optimization
    p_compare.add_argument("--downsample-rate", type=int, default=1, help="Process 1 out of every N windows (default: 1, no downsampling)")
    p_compare.add_argument("--max-windows", type=int, help="Maximum number of windows to process per sample (cap total)")
    p_compare.add_argument("--sample-fraction", type=float, help="Randomly sample this fraction of windows (e.g., 0.2 for 20%%)")

    p_compare.add_argument("--log-file", type=Path, help="Write compare logs to this file instead of default location")

    def cmd_compare(args: argparse.Namespace) -> int:
        try:
            # macOS memory optimization: reduce batch size for memory efficiency
            import platform
            batch_size = args.batch_size
            if platform.system() == "Darwin" and batch_size == 16:
                batch_size = 4
                print(f"[StrainVector] macOS detected; reducing batch_size to {batch_size} for memory efficiency.")

            out_json = run_compare(
                samples=args.samples,
                window=args.window,
                stride=args.stride,
                min_contig_len=args.min_contig_len,
                max_ns_frac=args.max_ns_frac,
                model_name=args.model,
                tokenizer_name=args.tokenizer,
                pooling=args.pooling,
                normalize=not args.no_normalize,
                precision=args.precision,
                batch_size=batch_size,
                device=args.device,
                metric=args.metric,
                comparison_mode=args.comparison_mode,
                downsample_rate=args.downsample_rate,
                max_windows=args.max_windows,
                sample_fraction=args.sample_fraction,
                out_json=args.out,
                out_tsv=args.out_tsv,
                log_file=args.log_file,
            )
            print(f"Wrote compare matrix: {out_json}")
            if args.out_tsv:
                print(f"TSV matrix: {args.out_tsv}")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
