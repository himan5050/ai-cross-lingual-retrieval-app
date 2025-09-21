from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..clir_config import Config
from ..clir_query import QueryEncoder, Retriever
from .datasets import load_queries_jsonl, load_qrels
from .metrics import compute_per_query_metrics, reduce_metrics
from ..indexing.elastic import ElasticClient

@dataclass
class EvalCfg:
    queries_path: str
    qrels_path: str
    top_k: List[int]
    use_graded_ndcg: bool = True
    query_batch_size: int = 128
    max_seq_length: Optional[int] = None
    cache_query_emb: Optional[str] = None  # npy path to cache/reuse query embeddings
    # Optional filter
    max_queries: Optional[int] = None

class Evaluator:
    def __init__(self, cfg: Config, ecfg: EvalCfg):
        self.cfg = cfg
        self.ecfg = ecfg
        self.Q = load_queries_jsonl(ecfg.queries_path)
        if ecfg.max_queries:
            self.Q = self.Q[: int(ecfg.max_queries)]
        self.qrels = load_qrels(ecfg.qrels_path)

        # init encoder
        self.encoder = QueryEncoder(
            model_name=cfg.model_name(),
            device=cfg.embed.device,
            batch_size=ecfg.query_batch_size,
            normalize=cfg.embed.normalize,
            max_seq_length=ecfg.max_seq_length,
        )

        # init retriever
        es_client = None
        if cfg.elastic.enabled:
            es_client = ElasticClient(
                host=cfg.elastic.host,
                username=cfg.elastic.username,
                password=cfg.elastic.password,
                api_key=cfg.elastic.api_key,
                verify_certs=cfg.elastic.verify_certs,
            )
        self.retriever = Retriever(
            meta_path=cfg.paths.meta_table,
            faiss_index_path=cfg.paths.faiss_index if cfg.faiss.enabled else None,
            es=es_client,
            es_index_name=cfg.paths.es_index_name if cfg.elastic.enabled else None,
        )

    def _encode_queries(self, texts: List[str]) -> np.ndarray:
        if self.ecfg.cache_query_emb and Path(self.ecfg.cache_query_emb).exists():
            return np.load(self.ecfg.cache_query_emb, mmap_mode="r")
        Xs = []
        # batch encode
        for i in tqdm(range(0, len(texts), self.ecfg.query_batch_size), desc="Encode queries", unit="q"):
            batch = texts[i : i + self.ecfg.query_batch_size]
            Xs.append(self.encoder.encode(batch))
        Q = np.vstack(Xs) if Xs else np.zeros((0, 0), dtype=np.float32)
        if self.ecfg.cache_query_emb:
            Path(self.ecfg.cache_query_emb).parent.mkdir(parents=True, exist_ok=True)
            np.save(self.ecfg.cache_query_emb, Q)
        return Q

    def evaluate(self) -> Dict[str, float]:
        qids = [q["qid"] for q in self.Q]
        texts = [q["text"] for q in self.Q]
        Q_vec = None

        method = self.cfg.retrieval.method
        need_vecs = method in ("faiss", "elastic_dense", "hybrid")
        if need_vecs:
            Q_vec = self._encode_queries(texts)

        per_query_results = []
        runs = []  # for optional TREC run output

        # retrieve per query (FAISS can batch; ES we loop)
        if method == "faiss":
            results = self.retriever.run(method="faiss", Q_vecs=Q_vec, top_k=max(self.ecfg.top_k))
            for qi, qid in enumerate(qids):
                ranked = results[qi]
                doc_ids = [r["doc_id"] for r in ranked]
                per_query_results.append(
                    compute_per_query_metrics(doc_ids, self.qrels.get(qid, {}), self.ecfg.top_k)
                )
                runs.extend([(qid, d, ranked[i]["score"]) for i, d in enumerate(doc_ids)])
        else:
            for i, q in enumerate(tqdm(self.Q, desc="Retrieve", unit="q")):
                ranked = self.retriever.run(
                    method=method,
                    Q_vecs=(None if Q_vec is None else Q_vec[i : i + 1]),
                    Q_texts=[q["text"]],
                    top_k=max(self.ecfg.top_k),
                    hybrid_alpha=self.cfg.retrieval.hybrid_alpha,
                )[0]
                doc_ids = [r["doc_id"] for r in ranked]
                per_query_results.append(
                    compute_per_query_metrics(doc_ids, self.qrels.get(q["qid"], {}), self.ecfg.top_k)
                )
                runs.extend([(q["qid"], d, ranked[j]["score"]) for j, d in enumerate(doc_ids)])

        agg = reduce_metrics(per_query_results)
        self._save_report(per_query_results, agg, runs)
        return agg

    def _save_report(self, per_query, aggregate, runs):
        out_dir = Path("artifacts/eval")
        out_dir.mkdir(parents=True, exist_ok=True)

        # per-query metrics
        pd.DataFrame(per_query).to_csv(out_dir / "per_query_metrics.csv", index=False)
        # aggregate
        with open(out_dir / "aggregate.json", "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)
        # TREC run file (qid Q0 docid rank score tag)
        with open(out_dir / "run.trec", "w", encoding="utf-8") as f:
            tag = f"{self.cfg.experiment.name}-{self.cfg.retrieval.method}"
            for qid, doc_id, score in runs:
                # rank is implicit by order
                f.write(f"{qid} Q0 {doc_id} 0 {score:.6f} {tag}\n")
