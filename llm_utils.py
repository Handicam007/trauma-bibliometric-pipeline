"""
Shared LLM Pipeline Utilities
===============================
Centralized cache management, DOI handling, and async processing
to eliminate code duplication across LLM task modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from config import LLM_BATCH_SIZE, LLM_CACHE_DIR

logger = logging.getLogger("llm_pipeline")

CACHE_DIR = Path(__file__).parent / LLM_CACHE_DIR


# ═══════════════════════════════════════════════════════════════════════
# SAFE DOI KEY — Prevents NaN collision bug
# ═══════════════════════════════════════════════════════════════════════

def prompt_hash(prompt_text: str) -> str:
    """Generate a short hash of a prompt for cache-key incorporation.

    When this hash changes (because the prompt was edited), cache keys
    using it will no longer match, forcing re-processing. This prevents
    stale cached results from an old prompt being mixed with new results.
    """
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:8]


def safe_doi_key(doi_value, title: str = "") -> str:
    """
    Generate a safe cache key from a DOI, falling back to a title hash.

    Fixes the FATAL NaN-DOI cache collision bug where all papers with
    NaN/empty DOIs would map to the same cache key "nan".

    Args:
        doi_value: DOI string (may be NaN, None, "nan", or empty)
        title: Paper title as fallback for hashing

    Returns:
        A unique string key for caching
    """
    doi_str = str(doi_value).strip() if doi_value is not None else ""

    # Check for valid DOI (not empty, not "nan", not just whitespace)
    if doi_str and doi_str.lower() != "nan" and doi_str != "None":
        return doi_str

    # Fallback: hash the title for a unique-enough key
    if title and str(title).strip() and str(title).lower() != "nan":
        title_clean = str(title).strip().lower()
        title_hash = hashlib.sha256(title_clean.encode()).hexdigest()[:16]
        return f"__title_hash__{title_hash}"

    # Last resort: truly unknown paper
    return f"__unknown__{hashlib.sha256(os.urandom(16)).hexdigest()[:12]}"


def is_valid_doi(doi_value) -> bool:
    """Check if a DOI value is valid (not NaN, None, or empty)."""
    if doi_value is None:
        return False
    doi_str = str(doi_value).strip()
    return bool(doi_str) and doi_str.lower() != "nan" and doi_str != "None"


# ═══════════════════════════════════════════════════════════════════════
# ATOMIC CACHE OPERATIONS — Prevents corruption on interrupt
# ═══════════════════════════════════════════════════════════════════════

def load_cache(cache_file: Path) -> dict:
    """
    Load cache from disk safely.

    Returns empty dict if file doesn't exist or is corrupted.
    """
    if cache_file.exists():
        try:
            content = cache_file.read_text()
            if content.strip():
                return json.loads(content)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cache file corrupted ({cache_file}): {e}. Starting fresh.")
    return {}


def save_cache(cache: dict, cache_file: Path):
    """
    Save cache to disk atomically.

    Writes to a temp file first, then renames. This prevents corruption
    if the process is killed mid-write (Ctrl+C, OOM, etc.).
    """
    cache_dir = cache_file.parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write to temp file in same directory (for same-filesystem rename)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(cache_dir),
            prefix=".cache_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            # Atomic rename (same filesystem)
            os.replace(tmp_path, str(cache_file))
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.warning(f"Failed to save cache ({cache_file}): {e}")


# ═══════════════════════════════════════════════════════════════════════
# PROGRESS REPORTING
# ═══════════════════════════════════════════════════════════════════════

def should_report_progress(index: int, total: int, interval: int = None) -> bool:
    """Check if we should print a progress update."""
    if interval is None:
        interval = LLM_BATCH_SIZE
    return (index + 1) % interval == 0 or (index + 1) == total


# ═══════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════

def setup_logging(level: int = logging.INFO):
    """Configure logging for the LLM pipeline."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("llm_pipeline")
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)
    return root
