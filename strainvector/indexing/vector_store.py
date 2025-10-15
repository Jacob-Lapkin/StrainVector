from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class VectorWriter:
    """Writes embeddings and a manifest into sharded files.

    This implementation saves per-batch .npy files in vectors/ and parallel
    JSONL manifests with metadata. Requires numpy at runtime.
    """

    def __init__(self, vectors_dir: Path, shard_prefix: str = "part", shard_size: int = 50000):
        self.vectors_dir = vectors_dir
        self.shard_prefix = shard_prefix
        self.shard_size = shard_size
        self._np = None
        self._buffer: List = []
        self._meta: List[Dict] = []
        self._shard_idx = 0
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_np(self):
        if self._np is None:
            try:
                import numpy as np  # type: ignore
            except Exception as e:  # pragma: no cover
                raise ImportError("VectorWriter requires 'numpy'.") from e
            self._np = np

    def add(self, vectors, metadata: List[Dict]):
        self._ensure_np()
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        self._buffer.append(self._np.asarray(vectors))
        self._meta.extend(metadata)
        if sum(b.shape[0] for b in self._buffer) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        self._ensure_np()
        np = self._np
        shard_vectors = np.concatenate(self._buffer, axis=0)
        shard_path = self.vectors_dir / f"{self.shard_prefix}_{self._shard_idx:05d}.npy"
        meta_path = self.vectors_dir / f"{self.shard_prefix}_{self._shard_idx:05d}.jsonl"
        np.save(shard_path, shard_vectors)
        with meta_path.open("w", encoding="utf-8") as f:
            for m in self._meta:
                f.write(json.dumps(m) + "\n")
        self._buffer = []
        self._meta = []
        self._shard_idx += 1
