from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
import yaml, os

class Paths(BaseModel):
    raw_data_filepath: str
    processed_jsonl: str
    embeddings_npy: str
    metadata_table: str
    faiss_index: Optional[str] = None
    es_index_name: Optional[str] = None

class IngestCfg(BaseModel):
    text_col: Optional[str] = None
    title_col: Optional[str] = None
    lang_col: Optional[str] = None
    chunksize: int = 10000
    encoding: str = "utf-8"
    min_chars: int = 100
    max_chars: int = 200000
    dedup: bool = True

class PreprocessCfg(BaseModel):
    lowercase: bool = True
    strip_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    keep_punct: bool = True
    keep_digits: bool = True
    collapse_whitespace: bool = True
    unicode_form: str = "NFKC"
    detect_language: bool = False
    use_stopwords: bool = True
    stopwords_fallback: str = "en"
    min_tokens: int = 3
    max_tokens: int = 4096
    lang_detector: str = "lid176"
    lid_model_path: str = "data/model/lid.176.bin"
    lid_min_chars: int = 20

class EmbedCfg(BaseModel):
    model: Literal["labse", "st-base", "st-paraphrase", "xlm-r", "custom"] = "st-base"
    custom_model_name: Optional[str] = None
    device: Optional[Literal["cpu", "cuda"]] = None
    batch_size: int = 128 # model batch size
    fp16: bool = True # compute/store FP16 when CUDA
    normalize: bool = True
    write_dtype: Literal["float32", "float16"] = "float16" # on-disk dtype
    stream_batch_docs: int = 1024 # docs per streaming chunk
    max_docs: Optional[int] = None # cap for quick experiments

class FAISSCfg(BaseModel):
    enabled: bool = True
    metric: Literal["ip", "l2"] = "ip"
    build_in_chunks: bool = True
    add_chunk_rows: int = 50000 # rows to add per chunk

class ESCfg(BaseModel):
    enabled: bool = False
    host: str = "http://localhost:9200"
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    verify_certs: bool = True
    backend: Literal["elasticsearch", "opensearch"] = "elasticsearch"
    dims: int = 768
    similarity: Literal["cosine", "dot_product", "l2_norm"] = "cosine"
    knn_hnsw_m: int = 16
    knn_ef_construction: int = 100
    refresh: bool = False
    bulk_size: int = 1000

class RetrievalCfg(BaseModel):
    method: Literal["faiss", "elastic_dense", "elastic_bm25", "hybrid"] = "hybrid"
    top_k: int = 10
    hybrid_alpha: float = 0.5 # weight for dense in [0,1]
    query_language: Optional[str] = None

class ExperimentCfg(BaseModel):
    name: str = "default-exp"
    seed: int = 42

class Config(BaseModel):
    experiment: ExperimentCfg
    paths: Paths
    ingest: IngestCfg
    preprocess: PreprocessCfg
    embed: EmbedCfg
    faiss: FAISSCfg
    elastic: ESCfg
    retrieval: RetrievalCfg


    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Config(**data)


    def model_name(self) -> str:
        if self.embed.model == "custom":
            assert self.embed.custom_model_name, "custom_model_name must be set"
            return self.embed.custom_model_name
        return {
            "labse": "sentence-transformers/LaBSE",
            "st-base": "sentence-transformers/distiluse-base-multilingual-cased-v2",
            "st-paraphrase": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "xlm-r": "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        }[self.embed.model]
