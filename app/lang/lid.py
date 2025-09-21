# clir/lang/lid.py
from __future__ import annotations
from functools import lru_cache
from typing import Optional, Tuple

_LID = None  # lazily loaded fastText model


@lru_cache(maxsize=1)
def _load_fasttext(model_path: str):
    """Load fastText LID model once per process."""
    global _LID
    if _LID is not None:
        return _LID
    try:
        import fasttext
    except Exception as e:
        raise RuntimeError(
            "fastText is not installed. Please `pip install fasttext-wheel` (preferred) "
            "or `pip install fasttext`."
        ) from e
    _LID = fasttext.load_model(model_path)
    return _LID


def predict_lang_lid(text: str, model_path: str, k: int = 1) -> Tuple[str, float]:
    """
    Predict language using fastText LID (lid.176.bin).
    Returns (lang_code, score). lang_code like 'en','km','es','fr',...
    """
    if not text or not text.strip():
        return "und", 0.0
    lid = _load_fasttext(model_path)
    labels, scores = lid.predict(text.strip().replace("\n", " "), k=k)
    # labels like __label__en
    lab = labels[0].replace("__label__", "") if labels else "und"
    sc = float(scores[0]) if scores else 0.0
    return lab, sc
