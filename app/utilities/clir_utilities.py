
# ────────────────────────────────────────────────────────────────────────────────
# Define Utilities
# ────────────────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _stage_header(logger: logging.Logger, title: str) -> None:
    """
    Logs a formatted header with the specified title using the provided logger.

    The header consists of a bar of '=' characters above and below the title,
    with the bar length being at least 12 characters or the length of the title plus 2.

    Args:
        logger (logging.Logger): The logger instance to use for logging the header.
        title (str): The title to display in the header.

    Returns:
        None
    """
    bar = "=" * max(12, len(title) + 2)
    logger.info("\n%s\n %s \n%s", bar, title, bar)


def _is_dir(p: str) -> bool:
    """
    Check if the given path exists and is a directory.

    Args:
        p (str): The path to check.

    Returns:
        bool: True if the path exists and is a directory, False otherwise.
    """
    return Path(p).exists() and Path(p).is_dir()


def _list_csvs(base: str, recursive: bool = True, pattern: str = "*.csv") -> list[Path]:
    """
    List CSV files in a given directory.

    Args:
        base (str): The base directory to search for CSV files.
        recursive (bool, optional): Whether to search recursively in subdirectories. Defaults to True.
        pattern (str, optional): Glob pattern to match CSV files. Defaults to "*.csv".

    Returns:
        list[Path]: Sorted list of Path objects pointing to found CSV files.

    Raises:
        FileNotFoundError: If the input directory does not exist or no CSV files are found.
    """
    root = Path(base)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {base}")
    globber = root.rglob if recursive else root.glob
    files = [p for p in globber(pattern) if p.is_file() and p.suffix.lower() == ".csv"]
    if not files:
        raise FileNotFoundError(f"No CSV files found at: {base} (pattern={pattern}, recursive={recursive})")
    return sorted(files)


def _rough_row_count_csv(path: Path, encoding: str = "utf-8") -> int:
    """
    Estimates the number of data rows in a CSV file by counting lines and subtracting one for the header.

    Args:
        path (Path): Path to the CSV file.
        encoding (str, optional): File encoding. Defaults to "utf-8".

    Returns:
        int: Approximate number of data rows (excluding header). Returns 0 if file cannot be read.

    Note:
        This method may be inaccurate if the CSV contains embedded newlines within quoted fields.
    
    Fast approximate row count (total lines - 1 header). Accurate for most CSVs.
    If your CSV has embedded newlines in quoted fields, the estimate may be off,
    but the progress bar will still advance correctly.
    """
    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            # subtract header
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def _find_columns_case_insensitive(columns: Iterable[str], required: Tuple[str, ...]) -> Dict[str, str]:
    """
    Maps required column names to their actual names in a given list of columns, ignoring case.

    For each required column name, searches for a matching column in `columns` in a case-insensitive manner.
    If a match is found, returns a dictionary mapping the required name to the actual column name.
    If a required column is not found, raises a ValueError with details.

    Args:
        columns (Iterable[str]): The list of available column names.
        required (Tuple[str, ...]): The tuple of required column names.

    Returns:
        Dict[str, str]: A dictionary mapping each required column name to its actual name in `columns`.

    Raises:
        ValueError: If any required column is not found in `columns` (case-insensitive).
    """
    lc_map = {c.lower(): c for c in columns}
    out: Dict[str, str] = {}
    for need in required:
        if need in lc_map:
            out[need] = lc_map[need]
        else:
            variants = (need, need.capitalize(), need.upper())
            found = next((lc_map[v.lower()] for v in variants if v.lower() in lc_map), None)
            if not found:
                raise ValueError(
                    f"Required column '{need}' not found. "
                    f"CSV must include: {', '.join(required)} (case-insensitive). "
                    f"Got: {list(columns)}"
                )
            out[need] = found
    return out
