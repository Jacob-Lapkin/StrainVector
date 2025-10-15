from __future__ import annotations

from pathlib import Path


def _load_all_vectors(vectors_dir: Path):
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Building an index requires 'numpy'.") from e
    arrs = []
    for npy in sorted(vectors_dir.glob("part_*.npy")):
        arrs.append(np.load(npy))
    if not arrs:
        raise FileNotFoundError("No vector shards found in vectors_dir")
    return np.concatenate(arrs, axis=0)


def build_faiss_index(vectors_dir: Path, index_dir: Path, metric: str = "cosine", index_type: str = "Flat") -> tuple:
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("FAISS index build requires 'faiss' and 'numpy'.") from e

    index_dir.mkdir(parents=True, exist_ok=True)
    X = _load_all_vectors(vectors_dir).astype("float32")

    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "ip":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    else:
        faiss_metric = faiss.METRIC_L2

    d = X.shape[1]
    if index_type == "Flat":
        index = faiss.IndexFlat(d, faiss_metric)
    else:
        if index_type.startswith("IVF"):
            nlist = int(index_type.replace("IVF", ""))
            quantizer = faiss.IndexFlat(d, faiss_metric)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss_metric)
            index.train(X)
        else:
            index = faiss.IndexFlat(d, faiss_metric)

    index.add(X)
    out_path = index_dir / "faiss.index"
    faiss.write_index(index, str(out_path))
    nvecs = X.shape[0]
    dim = X.shape[1]
    return out_path, int(nvecs), int(dim)
