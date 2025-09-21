from __future__ import annotations
import numpy as np
from typing import Tuple
from pathlib import Path

try:
    import faiss
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False

def build_faiss(embeddings: np.ndarray, index_path: str, metric: str = "ip") -> None:
    if not _HAVE_FAISS:
        raise RuntimeError("FAISS not installed. Install faiss-cpu or faiss-gpu.")
    xb = embeddings.astype("float32", copy=False)
    dim = xb.shape[1]
    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError("metric must be 'ip' or 'l2'")
    
    index.add(xb)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)

def build_faiss_stream(embeddings_path: str, index_path: str, metric: str = "ip", add_chunk_rows: int = 50000) -> None:
    """Build FAISS index by streaming from a memmap file without loading all rows."""
    if not _HAVE_FAISS:
        raise RuntimeError("FAISS not installed.")
    mm = np.memmap(embeddings_path, mode="r")
    # We need shape: memmap created with known shape; reopen with correct shape by reading np.load? Use .shape via np.memmap? We'll infer using ._mmap shape stored in .shape attribute on creation.
    # If shape is missing (old numpy), force-load header via np.load and then memmap: safer path below.
    
    try:
        X = np.load(embeddings_path, mmap_mode="r")
        n, d = X.shape
    except Exception:
        # Fallback (should not happen if saved via np.memmap or np.save)
        raise RuntimeError("Cannot infer shape from embeddings file; save with np.save or np.memmap.")

    if metric == "ip":
        index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError("metric must be 'ip' or 'l2'")

    start = 0
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    while start < n:
        end = min(start + add_chunk_rows, n)
        xb = X[start:end].astype("float32", copy=False)
        index.add(xb)
        start = end
    faiss.write_index(index, index_path)

def search_faiss(index_path: str, queries: np.ndarray, top_k: int = 10, metric: str = "ip") -> Tuple[np.ndarray, np.ndarray]:
    if not _HAVE_FAISS:
        raise RuntimeError("FAISS not installed.")
    index = faiss.read_index(index_path)
    q = queries.astype("float32", copy=False)
    scores, ids = index.search(q, top_k)
    return scores, ids