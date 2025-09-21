# clir/preprocess.py
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# Optional: stopword lists (simple fallback); replace with your own if needed
_STOPWORDS = {
    "en": set("the a an and or for of to in on with without as is are was were be been being this that these those".split()),
    "es": set("el la los las y o de del para en con sin como es son fue fueron ser estar este esta estos estas".split()),
    "fr": set("le la les et ou de des pour en avec sans comme est sont été être ce cette ces".split()),
    # leave "km" empty (Khmer) unless you have a curated list
    "km": set(),
}

_SENT_END = re.compile(
    r"""(
        [\.\!\?]+       # Latin enders
        (?=\s+|$)
      | [។៕៚៖]         # Khmer enders (។ ៕ ៚ ៖)
    )""",
    re.X,
)

_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]+", re.U)  # keep letters/digits/underscore/space


@dataclass
class PreprocessParams:
    # Normalization
    lowercase: bool = True
    strip_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    keep_punct: bool = True
    keep_digits: bool = True
    collapse_whitespace: bool = True
    unicode_form: str = "NFKC"

    # Language detection
    detect_language: bool = True
    use_stopwords: bool = True
    stopwords_fallback: str = "en"

    # Token limits
    min_tokens: int = 3
    max_tokens: int = 4096

    # NEW: language detector selection
    lang_detector: str = "lid176"   # "lid176" | "simple"
    lid_model_path: Optional[str] = "data/model/lid.176.bin"
    lid_min_chars: int = 20         # fall back if too short

    # NEW: sentence chunking
    chunking_enabled: bool = True
    chunk_min_sentences: int = 4    # "around 4–5 sentences"
    chunk_max_sentences: int = 5
    chunk_overlap_sentences: int = 0  # set 1 for overlapping windows


# ────────────────────────── Basic cleaners ──────────────────────────

def _strip_html(s: str) -> str:
    s = html.unescape(s)
    # minimal tag removal (keeps text)
    return re.sub(r"<[^>]+>", " ", s)

def _normalize_unicode(s: str, form: str) -> str:
    try:
        return unicodedata.normalize(form, s)
    except Exception:
        return s

def _remove_urls_emails(s: str, rm_urls: bool, rm_emails: bool) -> str:
    if rm_urls:
        s = _URL.sub(" ", s)
    if rm_emails:
        s = _EMAIL.sub(" ", s)
    return s

def _basic_clean(s: str, pp: PreprocessParams) -> str:
    if pp.strip_html:
        s = _strip_html(s)
    s = _remove_urls_emails(s, pp.remove_urls, pp.remove_emails)
    s = _normalize_unicode(s, pp.unicode_form)
    if pp.collapse_whitespace:
        s = _WS.sub(" ", s).strip()
    return s


# ───────────────────── Language detection (with LID) ─────────────────────

def detect_lang(text: str, pp: Optional[PreprocessParams] = None, fallback: str = "en") -> str:
    """
    Detect language; prefers fastText lid.176 if configured & available.
    """
    if not text:
        print('this is text condition')
        return fallback
    if pp is None or not pp.detect_language:
        print('this is text condition 2')
        return fallback

    txt = text.strip()
    if pp.lang_detector == "lid176" and pp.lid_model_path:
        # very short strings are unreliable; use fallback
        if len(txt) < pp.lid_min_chars:
            return fallback
        try:
            from app.lang.lid import predict_lang_lid
            lang, score = predict_lang_lid(txt, pp.lid_model_path)
            # you can tune the threshold; 0.5 is a conservative default
            return lang if score >= 0.5 else fallback
        except Exception:
            # missing model or library → fall back
            pass

    # simple heuristic fallback: look for non-latin ranges
    if re.search(r"[\u1780-\u17FF]", txt):  # Khmer block
        return "km"
    
    return fallback


# ───────────────────── Tokenization / stopwords ─────────────────────

_WORD = re.compile(r"\w+", re.U)

def simple_tokenize(s: str) -> List[str]:
    # note: for Khmer you'd want a proper segmenter; this keeps it simple
    return _WORD.findall(s.lower())

def remove_stopwords(toks: List[str], lang: Optional[str], fallback: str) -> List[str]:
    sw = _STOPWORDS.get((lang or fallback), _STOPWORDS.get(fallback, set()))
    return [t for t in toks if t not in sw]


# ───────────────────── Sentence split + chunking ─────────────────────

def split_sentences(text: str, lang: Optional[str]) -> List[str]:
    """
    Very light multilingual splitter:
    - Latin: ., !, ?
    - Khmer: ។ ៕ ៚ ៖
    Keeps end punctuation attached to sentence.
    """
    if not text:
        return []
    parts: List[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        sent = text[start:end].strip()
        if sent:
            parts.append(sent)
        start = end
    # tail
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def chunk_sentences(
    sents: List[str],
    min_s: int,
    max_s: int,
    overlap: int = 0,
) -> List[List[str]]:
    """
    Group sentences into windows of size ~max_s, with optional overlap.
    Ensures the final chunk has at least min_s sentences if possible.
    """
    if not sents:
        return []
    chunks: List[List[str]] = []
    i = 0
    step = max(1, max_s - overlap)
    n = len(sents)
    while i < n:
        j = min(i + max_s, n)
        chunk = sents[i:j]
        # If last chunk is too small and we already have one, merge into previous
        if j >= n and len(chunk) < min_s and chunks:
            chunks[-1].extend(chunk)
            break
        chunks.append(chunk)
        i += step
    return chunks


# ───────────────────── Public cleaning API used by ingest ─────────────────────

def normalize(s: str, pp: PreprocessParams) -> str:
    s = _basic_clean(s, pp)
    if pp.lowercase:
        s = s.lower()
    if not pp.keep_punct:
        s = _PUNCT.sub(" ", s)  # remove punctuation
    if not pp.keep_digits:
        s = re.sub(r"\d+", " ", s)
    if pp.collapse_whitespace:
        s = _WS.sub(" ", s).strip()
    return s


def clean_and_tokenize(text: str, lang: Optional[str], pp: PreprocessParams) -> Tuple[str, List[str]]:
    s = normalize(text, pp)
    toks = simple_tokenize(s)
    if pp.use_stopwords and (lang or pp.stopwords_fallback):
        toks = remove_stopwords(toks, lang, pp.stopwords_fallback)
    return " ".join(toks), toks
