from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd

from app.indexing.faiss_index import search_faiss
from app.indexing.elastic import ElasticClient
from app.retrieval.elastic_retriever import retrieve_bm25, retrieve_dense
from app.retrieval.hybrid import fuse_hybrid

class QueryEncoder:
    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 128,
                 normalize: bool = True, max_seq_length: Optional[int] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        if max_seq_length:
            self.model.max_seq_length = int(max_seq_length)
        self.batch_size = batch_size
        self.normalize = normalize

    def encode(self, texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            X = self.model.encode(
                texts, batch_size=self.batch_size, convert_to_numpy=True,
                normalize_embeddings=self.normalize, show_progress_bar=False
            )
        return X

class Retriever:
    """
    Unifies FAISS / ES-dense / BM25 / Hybrid for queries.
    Needs meta table (id map) for FAISS to map row indices -> string doc_ids.
    """
    def __init__(self, meta_path: str, faiss_index_path: Optional[str] = None,
                 es: Optional[ElasticClient] = None, es_index_name: Optional[str] = None):
        self.meta = pd.read_parquet(meta_path)
        self.row2id = self.meta["id"].astype(str).tolist()
        self.faiss_index_path = faiss_index_path
        self.es = es
        self.es_index = es_index_name

    def _faiss(self, Q: np.ndarray, top_k: int) -> List[List[Dict]]:
        scores, idxs = search_faiss(self.faiss_index_path, Q, top_k=top_k)
        out = []
        for i in range(Q.shape[0]):
            cur = []
            for j in range(top_k):
                row = int(idxs[i, j])
                doc_id = self.row2id[row] if 0 <= row < len(self.row2id) else str(row)
                cur.append({"doc_id": doc_id, "score": float(scores[i, j])})
            out.append(cur)
        return out

    def _es_dense(self, qvec: List[float], top_k: int) -> List[Dict]:
        return retrieve_dense(self.es, self.es_index, qvec, top_k=top_k)

    def _es_bm25(self, query_text: str, top_k: int) -> List[Dict]:
        return retrieve_bm25(self.es, self.es_index, query_text, top_k=top_k)

    def run(
        self,
        method: str, 
        Q_vecs: Optional[np.ndarray] = None,
        Q_texts: Optional[List[str]] = None,
        top_k: int = 10,
        hybrid_alpha: float = 0.5
    ) -> List[List[Dict]]:
        """
        Returns per-query list of {"doc_id","score"} sorted desc.
        """
        method = method.lower()
        if method == "faiss":
            assert Q_vecs is not None and self.faiss_index_path, "FAISS requires encoded queries + index."
            return self._faiss(Q_vecs, top_k=top_k)

        if method in ("elastic_dense", "elastic_bm25", "hybrid"):
            assert self.es is not None and self.es_index, "ES client/index required."
            assert Q_texts is not None, "Provide raw query texts for ES."

            results = []
            for i, qt in enumerate(Q_texts):
                dense_res, bm25_res = [], []
                if method in ("elastic_dense", "hybrid"):
                    assert Q_vecs is not None, "Hybrid/dense needs query vectors."
                    dense_res = self._es_dense(Q_vecs[i].tolist(), top_k=top_k)
                if method in ("elastic_bm25", "hybrid"):
                    bm25_res = self._es_bm25(qt, top_k=top_k)

                if method == "hybrid":
                    fused = fuse_hybrid(bm25=bm25_res, dense=dense_res, alpha=hybrid_alpha, top_k=top_k)
                    results.append(fused)
                else:
                    results.append(dense_res if method == "elastic_dense" else bm25_res)
            return results

        raise ValueError("Unknown method: {}".format(method))
