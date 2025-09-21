# Cross-lingual information retrieval executor file.

# Load required files.
from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable, Tuple, Dict
from tqdm import tqdm
from numpy.lib.format import open_memmap

import argparse
import json
import logging
import os
import time
import numpy as np
import pandas as pd
import hashlib

# Import application modules.
import app.utilities.clir_utilities as Utilities
from app.clir_config import Config
from app.utilities.clir_logging_utils import init_logger
from app.clir_preprocess import PreprocessParams, normalize, simple_tokenize, remove_stopwords, detect_lang
from app.clir_ingest import IngestParams, Ingestor, _find_columns_case_insensitive

def _infer_dim_from_index(cfg):
    try:
        import faiss  # if this crashes on mac, skip and let size inference handle it
        return int(faiss.read_index(cfg.paths.faiss_index).d)
    except Exception:
        return None
    
# ────────────────────────────────────────────────────────────────────────────────
# Stage 1: Ingest + Preprocess with per-file progress when input is a directory
# ────────────────────────────────────────────────────────────────────────────────

def _clir_ingest(config: Config, logger: logging.Logger) -> None:

    """
    Processes multiple CSV files containing text data, applies preprocessing and deduplication,
    and writes the cleaned results to a JSONL file with per-file progress tracking.

    This function operates in "directory-of-CSVs" mode:
      - Uses an outer progress bar for files and an inner progress bar for rows within each file.
      - Streams data in chunks for memory efficiency.
      - Deduplicates documents across all files using a stable hash.
      - Applies configurable preprocessing steps (e.g., lowercasing, HTML stripping, URL/email removal, punctuation/digit handling, whitespace collapsing, Unicode normalization, language detection, stopword removal, and token count thresholds).
      - Only documents meeting minimum/maximum character and token thresholds are included.
      - Each output document is assigned a stable ID in the format "<file-stem>:<row-idx>".

    Args:
        config (Config): Configuration object containing ingest, preprocess, and output parameters.
        logger (logging.Logger): Logger for status and progress messages.

    Side Effects:
        - Writes cleaned and deduplicated documents to a JSONL file at the path specified in config.
        - Logs progress and summary information.

    Raises:
        - May raise exceptions from file I/O, pandas, or preprocessing steps if errors occur.
    """

    Utilities._stage_header(logger, "Stage 1/3: Ingest + Preprocess → JSONL (per-file progress)")

    # Pull ingest params.
    recursive = True
    pattern = "*.csv"
    chunksize = int(getattr(config.ingest, "chunksize", 10_000))
    encoding = config.ingest.encoding
    min_chars = config.ingest.min_chars
    max_chars = config.ingest.max_chars
    dedup = config.ingest.dedup

    # Preprocess params
    pp = PreprocessParams(
        lowercase=config.preprocess.lowercase,
        strip_html=config.preprocess.strip_html,
        remove_urls=config.preprocess.remove_urls,
        remove_emails=config.preprocess.remove_emails,
        keep_punct=config.preprocess.keep_punct,
        keep_digits=config.preprocess.keep_digits,
        collapse_whitespace=config.preprocess.collapse_whitespace,
        unicode_form=config.preprocess.unicode_form,
        detect_language=config.preprocess.detect_language,
        use_stopwords=config.preprocess.use_stopwords,
        stopwords_fallback=config.preprocess.stopwords_fallback,
        min_tokens=config.preprocess.min_tokens,
        max_tokens=config.preprocess.max_tokens,
        lang_detector=config.preprocess.lang_detector,
        lid_model_path=config.preprocess.lid_model_path,
        lid_min_chars=config.preprocess.lid_min_chars
    )

    out_path = config.paths.processed_jsonl
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def _valid_len(s: str) -> bool:
        n = len(s)
        return (n >= min_chars) and (n <= max_chars)

    def _stable_hash(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

    files = Utilities._list_csvs(config.paths.raw_data_filepath, recursive=recursive, pattern=pattern)
    seen: set[str] = set()
    wrote_docs = 0

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out, tqdm(
        files, desc="Files", unit="file", dynamic_ncols=True
    ) as file_bar:
        for csv_path in file_bar:
            approx_rows = Utilities._rough_row_count_csv(csv_path, encoding=encoding)
            file_bar.set_postfix_str(csv_path.name)

            # Per-file inner bar
            with tqdm(total=approx_rows or None, desc=f"{csv_path.name}", unit="row", leave=False, dynamic_ncols=True) as row_bar:
                row_idx = 0
                for chunk in pd.read_csv(csv_path, encoding=encoding, chunksize=chunksize):
                    colmap = _find_columns_case_insensitive(chunk.columns, ("title", "text"))
                    Tcol, tcol = colmap["title"], colmap["text"]

                    # Iterate rows in this chunk
                    for _, r in chunk.iterrows():
                        raw_text = str(r.get(tcol, "") or "").strip()
                        if not raw_text or not _valid_len(raw_text):
                            row_idx += 1
                            row_bar.update(1)
                            continue

                        title = r.get(Tcol)
                        title = str(title).strip() if pd.notna(title) else None

                        # Language
                        lang = detect_lang(text=raw_text, pp=pp, fallback=pp.stopwords_fallback) if (pp.detect_language or True) else None

                        # Normalize → tokenize → optional stopwords
                        norm = normalize(raw_text, pp)
                        toks = simple_tokenize(norm)
                        if pp.use_stopwords and lang:
                            toks = remove_stopwords(toks, lang, pp.stopwords_fallback)

                        # token thresholds
                        if not (pp.min_tokens <= len(toks) <= pp.max_tokens):
                            row_idx += 1
                            row_bar.update(1)
                            continue

                        cleaned = " ".join(toks)

                        # dedup
                        if dedup:
                            key = _stable_hash(cleaned)
                            if key in seen:
                                row_idx += 1
                                row_bar.update(1)
                                continue
                            seen.add(key)

                        # doc id stable across runs: "<file-stem>:<row-idx>"
                        doc_id = f"{csv_path.stem}:{row_idx}"
                        out.write(json.dumps({"id": doc_id, "title": title, "text": cleaned, "lang": lang}, ensure_ascii=False) + "\n")
                        wrote_docs += 1

                        row_idx += 1
                        row_bar.update(1)

                        # optional global cap to speed debug
                        if config.embed.max_docs and wrote_docs >= int(config.embed.max_docs):
                            break
                    if config.embed.max_docs and wrote_docs >= int(config.embed.max_docs):
                        break

    dt = time.time() - t0
    logging.getLogger(__name__).info("Wrote cleaned JSONL: %s (docs=%d, %.2fs)", out_path, wrote_docs, dt)

# ────────────────────────────────────────────────────────────────────────────────
# Stage 2: Embedding (python/notebook) — keeps progress bar from earlier version
# ────────────────────────────────────────────────────────────────────────────────

def _count_lines(path: str, limit: Optional[int] = None) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
            if limit and n >= limit:
                break
    return n


def _embed(config: Config, logger: logging.Logger) -> tuple[int, int]:
    from sentence_transformers import SentenceTransformer
    import torch
    import pandas as pd  # noqa: F401 (only to align with notebook outputs)

    Utilities._stage_header(logger, "Stage 2/3: Embeddings (python) → memmap + meta")

    input_jsonl = config.paths.processed_jsonl
    out_embeddings = config.paths.embeddings_npy
    out_meta = config.paths.metadata_table

    device = config.embed.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.model_name()

    logger.info("Loading model %s on %s ...", model_name, device)
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    logger.info("Model loaded in %.2fs", time.time() - t0)

    N = _count_lines(input_jsonl, config.embed.max_docs)
    logger.info("Docs to embed: %d", N)

    use_fp16 = (config.embed.write_dtype == "float16") and (device == "cuda")
    mm_dtype = np.float16 if use_fp16 else np.float32

    ids: list[str] = []
    titles: list[Optional[str]] = []
    langs: list[Optional[str]] = []

    Path(out_embeddings).parent.mkdir(parents=True, exist_ok=True)
    Path(out_meta).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    dim: Optional[int] = None
    mm = None

    encode_t = 0.0
    io_t = 0.0

    with torch.inference_mode(), tqdm(
        total=N, desc="Embedding", unit="doc", dynamic_ncols=True
    ) as pbar:
        batch_txt: list[str] = []
        batch_meta: list[tuple[str, Optional[str], Optional[str]]] = []

        with open(input_jsonl, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if config.embed.max_docs is not None and i >= int(config.embed.max_docs):
                    break
                if not line.strip():
                    continue
                rec = json.loads(line)
                batch_txt.append(rec.get("text", "") or "")
                batch_meta.append((str(rec.get("id", "")), rec.get("title"), rec.get("lang")))

                if len(batch_txt) >= int(config.embed.stream_batch_docs):
                    t_enc0 = time.time()
                    X = model.encode(
                        batch_txt,
                        batch_size=int(config.embed.batch_size),
                        convert_to_numpy=True,
                        normalize_embeddings=bool(config.embed.normalize),
                        show_progress_bar=False,
                    )
                    encode_t += time.time() - t_enc0

                    if use_fp16:
                        X = X.astype(np.float16)

                    if mm is None:
                        dim = int(X.shape[1])
                        # mm = np.memmap(out_embeddings, mode="w+", dtype=mm_dtype, shape=(N, dim))
                        mm = open_memmap(out_embeddings, mode="w+", dtype=mm_dtype, shape=(N, dim))
                        logger.info("Memmap created: %s shape=(%d,%d) dtype=%s", out_embeddings, N, dim, mm_dtype)

                    t_io0 = time.time()
                    end = written + X.shape[0]
                    mm[written:end, :] = X
                    mm.flush()
                    io_t += time.time() - t_io0

                    for (idv, title, lang) in batch_meta:
                        ids.append(idv); titles.append(title); langs.append(lang)

                    written = end
                    pbar.update(X.shape[0])
                    batch_txt.clear(); batch_meta.clear()

            # tail
            if batch_txt:
                t_enc0 = time.time()
                X = model.encode(
                    batch_txt,
                    batch_size=int(config.embed.batch_size),
                    convert_to_numpy=True,
                    normalize_embeddings=bool(config.embed.normalize),
                    show_progress_bar=False,
                )
                encode_t += time.time() - t_enc0

                if use_fp16:
                    X = X.astype(np.float16)

                if mm is None:
                    dim = int(X.shape[1])
                    # mm = np.memmap(out_embeddings, mode="w+", dtype=mm_dtype, shape=(N, dim))
                    mm = open_memmap(out_embeddings, mode="w+", dtype=mm_dtype, shape=(N, dim))
                    logger.info("Memmap created: %s shape=(%d,%d) dtype=%s", out_embeddings, N, dim, mm_dtype)

                t_io0 = time.time()
                end = written + X.shape[0]
                mm[written:end, :] = X
                mm.flush()
                io_t += time.time() - t_io0

                for (idv, title, lang) in batch_meta:
                    ids.append(idv); titles.append(title); langs.append(lang)
                pbar.update(X.shape[0])
                written = end

    # write meta
    import pandas as pd
    pd.DataFrame({"id": ids, "title": titles, "lang": langs}).to_parquet(out_meta, index=False)

    logger.info("Embeddings saved: %s (rows=%d, dim=%s); meta: %s", out_embeddings, wrote if (wrote := written) else written, dim, out_meta)
    logger.info("Timing — encode: %.2fs | I/O: %.2fs | total: %.2fs", encode_t, io_t, time.time() - t0)
    return written, dim or 0

# ────────────────────────────────────────────────────────────────────────────────
# Stage 3: Indexing (FAISS / Elasticsearch) — progress bars kept
# ────────────────────────────────────────────────────────────────────────────────

def _build_faiss_with_progress(config: Config, logger: logging.Logger) -> None:
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("FAISS not installed. Install faiss-cpu or faiss-gpu.") from e

    path = config.paths.embeddings_npy
    try:
        X = np.load(path, mmap_mode="r")
    except ValueError:
        n = len(pd.read_parquet(config.paths.metadata_table))
        dtype = np.float16 if config.embed.write_dtype == "float16" else np.float32
        d = config.elastic.dims          # or hardcode 768 if that’s your model’s dim
        X = np.memmap(path, mode="r", dtype=dtype, shape=(n, d))

    X = np.array(X, dtype="float32")
    
    # Normalize embeddings to unit length
    faiss.normalize_L2(X)

    n, d = X.shape
    metric = config.faiss.metric
    if metric == "ip":
        index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError("faiss.metric must be 'ip' or 'l2'")

    add_chunk_rows = int(config.faiss.add_chunk_rows)
    logger.info("Building FAISS (%s) with %d vectors in chunks of %d ...", metric, n, add_chunk_rows)

    with tqdm(total=n, desc="FAISS add", unit="vec", dynamic_ncols=True) as pbar:
        start = 0
        while start < n:
            end = min(start + add_chunk_rows, n)
            xb = np.asarray(X[start:end], dtype=np.float32, order="C")
            index.add(xb)
            pbar.update(end - start)
            start = end

    Path(config.paths.faiss_index).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, config.paths.faiss_index)
    logger.info("FAISS index written: %s", config.paths.faiss_index)


def _bulk_index_es_with_progress(config: Config, logger: logging.Logger) -> None:
    from .indexing.elastic import ElasticClient
    import pandas as pd

    ec = ElasticClient(
        host=config.elastic.host,
        username=config.elastic.username,
        password=config.elastic.password,
        api_key=config.elastic.api_key,
        verify_certs=config.elastic.verify_certs,
    )

    index = config.paths.es_index_name
    dims = config.elastic.dims
    ec.create_index(
        index=index,
        dims=dims,
        backend=config.elastic.backend,
        similarity=config.elastic.similarity,
        m=config.elastic.knn_hnsw_m,
        ef_construction=config.elastic.knn_ef_construction,
    )

    meta = pd.read_parquet(config.paths.metadata_table)

    n = len(pd.read_parquet(config.paths.metadata_table))
    dtype = np.float16  # or np.float32 to match your run
    d = _infer_dim_from_index(config) or (os.path.getsize(config.paths.embeddings_npy) // (n * np.dtype(dtype).itemsize))
    X = np.memmap(config.paths.embeddings_npy, mode="r", dtype=dtype, shape=(n, d))




    # X = np.load(config.paths.embeddings_npy, mmap_mode="r")
    assert len(meta) == X.shape[0], "meta and embeddings length mismatch"

    bulk_size = int(config.elastic.bulk_size)
    n = len(meta)
    logger.info("Bulk indexing %d docs into '%s' (chunk=%d)", n, index, bulk_size)

    with tqdm(total=n, desc="Elasticsearch bulk", unit="doc", dynamic_ncols=True) as pbar:
        for start in range(0, n, bulk_size):
            end = min(start + bulk_size, n)
            actions: list[dict] = []
            for i in range(start, end):
                _id = str(meta.iloc[i]["id"])
                doc = {
                    "id": _id,
                    "title": meta.iloc[i].get("title"),
                    "text": "",
                    "lang": meta.iloc[i].get("lang"),
                    "embedding": X[i].tolist(),
                }
                actions.append({"index": {"_index": index, "_id": _id}})
                actions.append(doc)
            ec.es.bulk(index=index, operations=actions, refresh=False)
            pbar.update(end - start)

    logger.info("Elasticsearch bulk indexing completed: %s", index)


def _index(config: Config, logger: logging.Logger) -> None:
    Utilities._stage_header(logger, "Stage 3/3: Indexing (FAISS / Elasticsearch)")

    if config.faiss.enabled and config.paths.faiss_index:
        _build_faiss_with_progress(config, logger)

    if config.elastic.enabled and config.paths.es_index_name:
        _bulk_index_es_with_progress(config, logger)


# ────────────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────────────
def execute(
    config_path: str,
    skip_ingest: bool = False,
    skip_embed: bool = False,
    skip_index: bool = False,
) -> None:
    
    """
    Executes the end-to-end cross-lingual information retrieval pipeline with progress logging for each stage.
    Args:
        config_path (str): Path to the configuration file.
        skip_ingest (bool, optional): If True, skips the document ingestion and cleaning stage. Defaults to False.
        skip_embed (bool, optional): If True, skips the embedding generation stage. Defaults to False.
        skip_index (bool, optional): If True, skips the indexing stage. Defaults to False.
    Raises:
        KeyboardInterrupt: If the execution is interrupted by the user.
    The pipeline consists of three main stages:
        1. Document ingestion and cleaning.
        2. Embedding generation.
        3. Indexing (e.g., FAISS/ES).
    Each stage can be optionally skipped using the corresponding flags. Progress and status are logged throughout execution.
    """

    # Initial the logger.
    logger = init_logger()
    config = Config.load(config_path)

    logger.info("Loaded config: %s", config_path)
    logger.info("Experiment: %s | Model: %s | Runner: Python",
                config.experiment.name, config.model_name())

    try:
        # Stage 1
        if not skip_ingest:
            if Utilities._is_dir(config.paths.raw_data_filepath):
                _clir_ingest(config, logger)
        else:
            logger.info("Skipping document ingest (using existing): %s", config.paths.processed_jsonl)

        # Stage 2
        if not skip_embed:
            _embed(config, logger)
        else:
            logger.info("Skipping embed (using existing): %s ; %s",
                        config.paths.embeddings_npy, config.paths.metadata_table)

        # Stage 3
        if not skip_index:
            _index(config, logger)
        else:
            logger.info("Skipping indexing (FAISS/ES).")

        logger.info("Pipeline completed successfully.")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Partial outputs may exist.")
        raise