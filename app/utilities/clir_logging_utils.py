from __future__ import annotations
import logging, sys

_DEF_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

def init_logger(name: str = "cross-lingual-retrieval-app", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(_DEF_FMT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    
    return logger
