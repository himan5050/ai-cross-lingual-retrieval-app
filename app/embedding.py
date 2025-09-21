from __future__ import annotations
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import json

def _dtype(write_dtype: str):
    return np.float16 if write_dtype == "float16" else np.float32

class Embedder:
    def __init__(self, model_name: str, device: str | None = None,
                batch_size: int = 128, fp16: bool = True, normalize: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        self.fp16 = fp16
        self.normalize = normalize


    def encode(self, texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            embs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=True,
            )

        if self.fp16 and self.device == "cuda":
            embs = embs.astype(np.float16)
        
        return embs


    @staticmethod
    def save(path: str, arr: np.ndarray) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

def count_lines(path: str, max_docs: Optional[int] = None) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
            if max_docs and n >= max_docs:
                break
    return n

def encode_jsonl_stream_to_memmap(
    input_jsonl: str,
    out_embeddings: str,
    out_meta: str,
    model_name: str,
    device: Optional[str] = None,
    batch_size: int = 128,
    fp16_store: bool = True,
    normalize: bool = True,
    stream_batch_docs: int = 1024,
    max_docs: Optional[int] = None,
) -> Tuple[int, int]:
    """Stream JSONL -> memmap embeddings without loading all texts in RAM.
    Returns (num_docs, dim).
    """
    
    import pandas as pd
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)


    total = count_lines(input_jsonl, max_docs)
    mm_dtype = _dtype("float16" if (fp16_store and device == "cuda") else "float32")


    ids, titles, langs = [], [], []
    dim = None
    mm = None
    wrote = 0


    Path(out_embeddings).parent.mkdir(parents=True, exist_ok=True)

    def yield_batches():
        batch_txt, batch_meta = [], []
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_docs and i >= max_docs:
                    break
                if not line.strip():
                    continue
                rec = json.loads(line)
                batch_txt.append(rec.get("text", ""))
                batch_meta.append((rec.get("id"), rec.get("title"), rec.get("lang")))
                if len(batch_txt) >= stream_batch_docs:
                    yield batch_txt, batch_meta
                    batch_txt, batch_meta = [], []
            if batch_txt:
                yield batch_txt, batch_meta
    
    with torch.inference_mode():
        for texts, meta in yield_batches():
            X = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                            normalize_embeddings=normalize, show_progress_bar=False)
            if fp16_store and device == "cuda":
                X = X.astype(np.float16)
            if dim is None:
                dim = X.shape[1]
                mm = np.memmap(out_embeddings, mode="w+", dtype=mm_dtype, shape=(total, dim))
                # write slice
                mm[wrote:wrote + X.shape[0], :] = X
                mm.flush()
                # collect meta
                for (idv, title, lang) in meta:
                    ids.append(idv); titles.append(title); langs.append(lang)
                wrote += X.shape[0]
    
    # write meta at end
    import pandas as pd
    meta_df = pd.DataFrame({"id": ids, "title": titles, "lang": langs})
    Path(out_meta).parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(out_meta, index=False)

    return wrote, dim or 0