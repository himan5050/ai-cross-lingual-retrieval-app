from __future__ import annotations
from typing import Any, Dict, List, Optional

# Uses the high-level wrapper you already have
from ..indexing.elastic import ElasticClient

Hit = Dict[str, Any]


def _parse_hits(resp: Dict[str, Any], include_source: bool = False) -> List[Hit]:
    """
    Parse ES search/knn response into a list of {"doc_id","score", ...}.
    """
    out: List[Hit] = []
    try:
        hits = resp.get("hits", {}).get("hits", [])
        for h in hits:
            item: Hit = {
                "doc_id": str(h.get("_id")),
                "score": float(h.get("_score", 0.0)),
            }
            if include_source and "_source" in h:
                item["source"] = h["_source"]
            out.append(item)
    except Exception:
        # Return what we have (empty on total failure)
        return out
    return out


def retrieve_bm25(
    es: ElasticClient,
    index: str,
    query_text: str,
    top_k: int = 10,
    include_source: bool = False,
) -> List[Hit]:
    """
    BM25 retrieval from Elasticsearch.
    Returns: [{"doc_id": str, "score": float, ("source": {...})}, ...]
    """
    resp = es.search_bm25(index=index, query=query_text, top_k=top_k)
    return _parse_hits(resp, include_source=include_source)


def retrieve_dense(
    es: ElasticClient,
    index: str,
    query_vec: List[float],
    top_k: int = 10,
    include_source: bool = False,
) -> List[Hit]:
    """
    Vector (kNN) retrieval from Elasticsearch.
    Returns: [{"doc_id": str, "score": float, ("source": {...})}, ...]
    """
    resp = es.search_dense(index=index, query_vec=query_vec, top_k=top_k)
    return _parse_hits(resp, include_source=include_source)
