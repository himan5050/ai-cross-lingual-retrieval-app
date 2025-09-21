from __future__ import annotations
from typing import Dict, List, Tuple

Hit = Dict[str, float]  # expects keys: "doc_id", "score"


def _min_max_norm(hits: List[Hit]) -> Dict[str, float]:
    """
    Normalize scores into [0,1] per list. If all equal/empty -> zeros.
    Returns mapping: doc_id -> norm_score
    """
    if not hits:
        return {}
    scores = [h["score"] for h in hits]
    s_min, s_max = min(scores), max(scores)
    if s_max <= s_min:
        # all equal -> return zeros (keeps neutrality in fusion)
        return {h["doc_id"]: 0.0 for h in hits}
    rng = (s_max - s_min)
    return {h["doc_id"]: (h["score"] - s_min) / rng for h in hits}


def fuse_hybrid(
    bm25: List[Hit],
    dense: List[Hit],
    alpha: float = 0.5,
    top_k: int = 10,
) -> List[Hit]:
    """
    Weighted score fusion (min–max normalized), then sort desc by fused score.

    Args:
        bm25:  list of {"doc_id","score"} from BM25
        dense: list of {"doc_id","score"} from dense kNN
        alpha: weight for DENSE (0..1). 1.0 => only dense, 0.0 => only BM25
        top_k: number of results to return

    Returns:
        Ranked list of {"doc_id","score"} with `score` = fused_score
    """
    # Normalize each list independently
    n_bm25 = _min_max_norm(bm25)
    n_dense = _min_max_norm(dense)

    # Union doc_ids
    all_ids = set(n_bm25.keys()) | set(n_dense.keys())
    if not all_ids:
        return []

    fused: List[Tuple[str, float, float, float]] = []  # (doc_id, fused, n_dense, n_bm25)
    for doc_id in all_ids:
        sd = n_dense.get(doc_id, 0.0)
        sb = n_bm25.get(doc_id, 0.0)
        fused_score = alpha * sd + (1.0 - alpha) * sb
        fused.append((doc_id, fused_score, sd, sb))

    # Sort by fused desc, then by better of components to break ties
    fused.sort(key=lambda x: (x[1], max(x[2], x[3])), reverse=True)

    return [{"doc_id": d, "score": float(s)} for d, s, _, _ in fused[:top_k]]


def rrf_fuse(
    lists: List[List[Hit]],
    k: int = 60,
    top_k: int = 10,
) -> List[Hit]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.
    Score(d) = sum_i 1 / (k + rank_i(d))
    """
    if not lists:
        return []
    # Build rank positions (1-based)
    rank_maps: List[Dict[str, int]] = []
    for hits in lists:
        rm: Dict[str, int] = {}
        for i, h in enumerate(hits):
            rm[h["doc_id"]] = i + 1
        rank_maps.append(rm)

    # Union doc_ids
    all_ids = set().union(*[set(rm.keys()) for rm in rank_maps])
    scores: Dict[str, float] = {}
    for d in all_ids:
        s = 0.0
        for rm in rank_maps:
            if d in rm:
                s += 1.0 / (k + rm[d])
        scores[d] = s

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [{"doc_id": d, "score": float(s)} for d, s in ranked]
