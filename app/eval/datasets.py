from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
import json
from pathlib import Path

def load_qrels_jsonl(path: str) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            r = json.loads(line)
            qid = str(r["qid"]); doc = str(r["doc_id"]); rel = float(r.get("rel", 1))
            qrels.setdefault(qid, {})[doc] = rel
    return qrels


def load_qrels(path: str) -> Dict[str, Dict[str, float]]:
    """
    Supports:
      - TREC qrels (4 cols): qid 0 docid rel  (space/tsv)
      - Generic CSV/TSV with columns: qid,doc_id,rel
    Returns: {qid: {doc_id: rel, ...}, ...}
    """
    p = Path(path)
    if p.suffix.lower() in [".csv", ".tsv"]:
        sep = "," if p.suffix.lower()==".csv" else "\t"
        df = pd.read_csv(p, sep=sep)
        # flexible col names
        col_q = next(c for c in df.columns if c.lower() in ("qid","query_id","q"))
        col_d = next(c for c in df.columns if c.lower() in ("doc_id","docid","did","pid"))
        col_r = next(c for c in df.columns if c.lower() in ("rel","relevance","label"))
        qrels = {}
        for _, r in df.iterrows():
            q = str(r[col_q]); d = str(r[col_d]); rel = float(r[col_r])
            qrels.setdefault(q, {})[d] = rel
        return qrels
    # TREC-style space-separated
    qrels = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split()
            if len(parts) < 4: 
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            qrels.setdefault(str(qid), {})[str(docid)] = float(rel)
    return qrels
