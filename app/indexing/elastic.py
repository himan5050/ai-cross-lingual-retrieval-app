from __future__ import annotations
from typing import List, Dict, Any, Optional
import math

try:
    from elasticsearch import Elasticsearch
except Exception as e:
    Elasticsearch = None


class ElasticClient:
    def __init__(self, host: str, username: Optional[str] = None,
                password: Optional[str] = None, api_key: Optional[str] = None,
                verify_certs: bool = True):
        if Elasticsearch is None:
            raise RuntimeError("elasticsearch client not installed. pip install elasticsearch")
        if api_key:
            self.es = Elasticsearch(hosts=[host], api_key=api_key, verify_certs=verify_certs)
        elif username and password:
            self.es = Elasticsearch(hosts=[host], basic_auth=(username, password), verify_certs=verify_certs)
        else:
            self.es = Elasticsearch(hosts=[host], verify_certs=verify_certs)


    def create_index(self, index: str, dims: int, backend: str = "elasticsearch",
                    similarity: str = "cosine", m: int = 16, ef_construction: int = 100) -> None:
        # ES >= 8: dense_vector with HNSW indexing
        if backend == "elasticsearch":
            body = {
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "text": {"type": "text"},
                        "lang": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": dims,
                            "index": True,
                            "similarity": similarity,
                            "index_options": {"type": "hnsw", "m": m, "ef_construction": ef_construction}
                        }
                    }
                }
            }
        else: # OpenSearch
            body = {
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "text": {"type": "text"},
                        "lang": {"type": "keyword"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": dims,
                            "method": {"name": "hnsw", "space_type": similarity, "engine": "nmslib"}
                        }
                    }
                }
            }
            self.es.indices.create(index=index, body=body, ignore=400)


    def bulk_index(self, index: str, docs: List[Dict[str, Any]], refresh: bool = False):
        actions = []
        for d in docs:
            actions.append({"index": {"_index": index, "_id": d["id"]}})
            actions.append(d)
        self.es.bulk(index=index, operations=actions, refresh=refresh)


    def search_bm25(self, index: str, query: str, top_k: int = 10) -> Dict[str, Any]:
        body = {"size": top_k, "query": {"multi_match": {"query": query, "fields": ["title^2", "text"]}}}
        return self.es.search(index=index, body=body)


    def search_dense(self, index: str, query_vec: list[float], top_k: int = 10) -> Dict[str, Any]:
        body = {"size": top_k, "knn": {"field": "embedding", "query_vector": query_vec, "k": top_k, "num_candidates": max(100, top_k*10)}}
        return self.es.search(index=index, body=body)