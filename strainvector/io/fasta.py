from __future__ import annotations

import gzip
from pathlib import Path
from typing import Generator, Iterable, Tuple


def open_maybe_gzip(path: Path, mode: str = "rt"):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, mode)
    return open(p, mode)


def fasta_iter(path: Path) -> Generator[Tuple[str, str], None, None]:
    """Stream FASTA records as (header, sequence) tuples.

    Minimal parser to avoid external dependencies.
    """
    header = None
    seq_chunks = []
    with open_maybe_gzip(path, "rt") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            yield header, "".join(seq_chunks)


def ns_fraction(seq: str) -> float:
    if not seq:
        return 1.0
    n = sum(1 for c in seq if c in ("N", "n"))
    return n / len(seq)


def chunk_sequence(seq: str, window: int, stride: int) -> Iterable[Tuple[int, int, str]]:
    if stride <= 0:
        stride = window
    L = len(seq)
    if L == 0:
        return []
    for start in range(0, max(1, L - window + 1), stride):
        end = min(L, start + window)
        if end - start <= 0:
            continue
        yield start, end, seq[start:end]

