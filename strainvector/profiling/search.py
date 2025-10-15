from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_db_vectors(vectors_dir: Path):
    """Load concatenated DB vectors and aligned metadata list.

    Returns (X, meta) where X is (N, D) float32 and meta is a list of dicts.
    Requires numpy.
    """
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Loading vectors requires 'numpy'.") from e

    arrs = []
    metas: List[Dict] = []
    shards = sorted(vectors_dir.glob("part_*.npy"))
    for npy in shards:
        jsonl = npy.with_suffix("")
        jsonl = jsonl.with_suffix(".jsonl")
        if not jsonl.exists():
            raise FileNotFoundError(f"Missing metadata JSONL for shard: {jsonl}")
        arrs.append(np.load(npy))
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                metas.append(json.loads(line))
    if not arrs:
        raise FileNotFoundError("No vector shards found in vectors_dir")
    X = np.concatenate(arrs, axis=0).astype("float32", copy=False)
    if X.shape[0] != len(metas):
        raise ValueError("Vectors and metadata count mismatch after loading shards")
    return X, metas


def brute_force_topk(queries, base, k: int, metric: str = "cosine") -> Tuple[List[List[int]], List[List[float]]]:
    """Compute top-k neighbors via numpy (small to medium datasets).

    Returns (indices, scores) where each is a list of lists per query.
    """
    import numpy as np  # type: ignore

    Q = queries.astype("float32", copy=False)
    B = base.astype("float32", copy=False)
    if metric == "cosine":
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        sims = Q @ B.T
        scores = sims
    elif metric == "ip":
        sims = Q @ B.T
        scores = sims
    else:  # l2 smaller is better; convert to negative distances
        # ||q - b||^2 = ||q||^2 + ||b||^2 - 2 q.b
        q2 = (Q ** 2).sum(axis=1, keepdims=True)
        b2 = (B ** 2).sum(axis=1, keepdims=True).T
        d2 = q2 + b2 - 2.0 * (Q @ B.T)
        scores = -d2

    idxs_list: List[List[int]] = []
    scrs_list: List[List[float]] = []
    for i in range(scores.shape[0]):
        row = scores[i]
        if k >= row.shape[0]:
            order = np.argsort(-row)
        else:
            part = np.argpartition(-row, k)[:k]
            order = part[np.argsort(-row[part])]
        idxs = order.tolist()
        scrs = row[order].tolist()
        idxs_list.append(idxs)
        scrs_list.append(scrs)
    return idxs_list, scrs_list


def faiss_topk(index_path: Path, queries, k: int, metric: str = "cosine") -> Tuple[List[List[int]], List[List[float]]]:
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("FAISS search requires 'faiss' and 'numpy'.") from e

    q = queries.astype("float32", copy=False)
    if metric == "cosine":
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    index = faiss.read_index(str(index_path))
    D, I = index.search(q, k)
    idxs_list = [row.tolist() for row in I]
    scrs_list = [row.tolist() for row in D]
    return idxs_list, scrs_list

