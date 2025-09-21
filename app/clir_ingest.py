# clir/ingest.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator, Dict, Any, Iterable, Tuple
import hashlib
import os

import pandas as pd

from app.clir_preprocess import (
    PreprocessParams,
    normalize,
    simple_tokenize,
    remove_stopwords,
    detect_lang,
    split_sentences,
    chunk_sentences,
    clean_and_tokenize,
)


@dataclass
class IngestParams:
    # Directory of CSVs
    input_path: str
    input_type: str = "csv"
    encoding: str = "utf-8"
    recursive: bool = True
    pattern: str = "*.csv"
    chunksize: int = 10000
    min_chars: int = 50
    max_chars: int = 200_000
    dedup: bool = True

    # Legacy single-file columns (unused for directory mode but kept for compat)
    text_col: Optional[str] = None
    id_col: Optional[str] = None
    title_col: Optional[str] = None
    lang_col: Optional[str] = None


def _find_columns_case_insensitive(cols_or_df, required: Tuple[str, ...]) -> Dict[str, str]:
    if hasattr(cols_or_df, "columns"):
        columns = list(cols_or_df.columns)
    else:
        columns = list(cols_or_df)
    lc_map = {str(c).lower(): str(c) for c in columns}
    out: Dict[str, str] = {}
    for need in required:
        if need in lc_map:
            out[need] = lc_map[need]
        else:
            variants = (need, need.capitalize(), need.upper())
            found = next((lc_map[v.lower()] for v in variants if v.lower() in lc_map), None)
            if not found:
                raise ValueError(
                    f"Required column '{need}' not found. Expect: {', '.join(required)}. Got: {columns}"
                )
            out[need] = found
    return out


def _gen_csv_paths(base: str, pattern: str, recursive: bool) -> Iterable[Path]:
    root = Path(base)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {base}")
    globber = root.rglob if recursive else root.glob
    for p in globber(pattern):
        if p.is_file() and p.suffix.lower() == ".csv":
            yield p


def _valid_length(s: str, min_chars: int, max_chars: int) -> bool:
    n = len(s)
    return (n >= min_chars) and (n <= max_chars)


def _stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


class Ingestor:
    """
    Streams CLEANED **chunks** from a directory of CSV files.
    Each input row → multiple output records (4–5 sentence chunks):
        {"id","title","text","lang"} where id = "<file-stem>:<row>#c<chunk_ix>"
    """
    def __init__(self, ip: IngestParams, pp: PreprocessParams):
        self.ip = ip
        self.pp = pp
        self._seen: set[str] = set()  # dedup on cleaned chunk text

    def _from_csv_dir(self) -> Iterable[Dict[str, Any]]:
        for csv_path in _gen_csv_paths(self.ip.input_path, self.ip.pattern, self.ip.recursive):
            row_idx = 0
            for chunk in pd.read_csv(csv_path, encoding=self.ip.encoding, chunksize=self.ip.chunksize):
                colmap = _find_columns_case_insensitive(chunk, ("title", "text"))
                Tcol, tcol = colmap["title"], colmap["text"]

                for _, r in chunk.iterrows():
                    raw = str(r.get(tcol, "") or "").strip()
                    if not raw or not _valid_length(raw, self.ip.min_chars, self.ip.max_chars):
                        row_idx += 1
                        continue

                    title = r.get(Tcol)
                    title = str(title).strip() if pd.notna(title) else None

                    # Language (fastText lid.176 if configured)
                    lang = detect_lang(raw, fallback=self.pp.stopwords_fallback, pp=self.pp)

                    # Sentence split + chunk
                    sents = split_sentences(raw, lang)
                    win = chunk_sentences(
                        sents,
                        min_s=self.pp.chunk_min_sentences,
                        max_s=self.pp.chunk_max_sentences,
                        overlap=self.pp.chunk_overlap_sentences,
                    ) if self.pp.chunking_enabled else [sents]

                    # Emit each chunk as a separate record
                    for ci, sent_list in enumerate(win):
                        chunk_text = " ".join(sent_list).strip()
                        if not chunk_text:
                            continue

                        cleaned, toks = clean_and_tokenize(chunk_text, lang, self.pp)

                        # token thresholds
                        if not (self.pp.min_tokens <= len(toks) <= self.pp.max_tokens):
                            continue

                        if self.ip.dedup:
                            key = _stable_hash(cleaned)
                            if key in self._seen:
                                continue
                            self._seen.add(key)

                        yield {
                            "id": f"{csv_path.stem}:{row_idx}#c{ci}",
                            "title": title,
                            "text": cleaned,
                            "lang": lang,
                            # Optional: parent id for traceability (won't break other components)
                            "parent_id": f"{csv_path.stem}:{row_idx}",
                            "chunk_index": ci,
                        }

                    row_idx += 1

    def stream_clean(self) -> Generator[Dict[str, Any], None, None]:
        # Currently only directory-of-CSVs mode is used; keep single-file modes if needed.
        if os.path.isdir(self.ip.input_path):
            yield from self._from_csv_dir()
            return
        raise NotADirectoryError(
            f"Expected directory at ingest.input_path={self.ip.input_path}. "
            f"For single files, extend Ingestor to handle CSV/JSONL modes."
        )
