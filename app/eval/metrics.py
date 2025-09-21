from __future__ import annotations
from typing import List, Dict, Tuple
import math
import numpy as np

def precision_at_k(gains: List[float], k: int) -> float:
    k = min(k, len(gains))
    if k == 0: return 0.0
    return float(np.sum(np.array(gains[:k]) > 0)) / k

def recall_at_k(gains: List[float], total_relevant: int, k: int) -> float:
    if total_relevant == 0: 
        return 0.0
    k = min(k, len(gains))
    return float(np.sum(np.array(gains[:k]) > 0)) / total_relevant

def average_precision(gains: List[float], k: int | None = None) -> float:
    """AP: mean precision at ranks of relevant docs."""
    n = len(gains) if k is None else min(k, len(gains))
    hits, s = 0, 0.0
    for i in range(n):
        if gains[i] > 0:
            hits += 1
            s += hits / (i + 1.0)
    return s / hits if hits > 0 else 0.0

def mrr_at_k(gains: List[float], k: int) -> float:
    n = min(k, len(gains))
    for i in range(n):
        if gains[i] > 0:
            return 1.0 / (i + 1.0)
    return 0.0

def dcg_at_k(gains: List[float], k: int, use_graded: bool = True) -> float:
    n = min(k, len(gains))
    dcg = 0.0
    for i in range(n):
        rel = gains[i] if use_graded else (1.0 if gains[i] > 0 else 0.0)
        denom = math.log2(i + 2.0)  # positions 1..k -> log2(i+2)
        dcg += (rel) / denom
    return dcg

def ndcg_at_k(gains: List[float], ideal_gains: List[float], k: int, use_graded: bool = True) -> float:
    ideal_sorted = sorted(ideal_gains, reverse=True)
    idcg = dcg_at_k(ideal_sorted, k, use_graded=use_graded)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(gains, k, use_graded=use_graded) / idcg

def compute_per_query_metrics(
    ranked_doc_ids: List[str],
    qrels: Dict[str, float],
    ks: List[int],
    use_graded_ndcg: bool = True,
) -> Dict[str, float]:
    """
    qrels: {doc_id: relevance} for this query (0 if missing)
    ranked_doc_ids: predicted order
    """
    gains = [qrels.get(d, 0.0) for d in ranked_doc_ids]
    ideal_gains = list(qrels.values())
    total_rel = int(np.sum(np.array(ideal_gains) > 0))

    out: Dict[str, float] = {}
    for k in ks:
        out[f"P@{k}"] = precision_at_k(gains, k)
        out[f"R@{k}"] = recall_at_k(gains, total_rel, k)
        out[f"nDCG@{k}"] = ndcg_at_k(gains, ideal_gains, k, use_graded=use_graded_ndcg)
    out["AP"] = average_precision(gains)
    out["MRR@{}".format(max(ks))] = mrr_at_k(gains, max(ks))
    return out

def reduce_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query: 
        return {}
    keys = per_query[0].keys()
    return {k: float(np.mean([r[k] for r in per_query])) for k in keys}
