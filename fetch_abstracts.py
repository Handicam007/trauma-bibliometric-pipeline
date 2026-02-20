#!/usr/bin/env python3
"""
Abstract Retrieval Module
=========================
Fills missing abstracts using:
  1. Scopus Abstract Retrieval API (primary — uses existing API key)
  2. PubMed E-utilities (fallback — free, no key needed)

Resumable: saves progress to llm_cache/abstracts_cache.json.
On restart, already-fetched abstracts are skipped.

Usage:
    python fetch_abstracts.py [--input path] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────
INPUT = Path(__file__).parent / "results_refined" / "all_results.csv"
CACHE_DIR = Path(__file__).parent / "llm_cache"
CACHE_FILE = CACHE_DIR / "abstracts_cache.json"

# ── Rate limits ───────────────────────────────────────────────────────
SCOPUS_DELAY = 0.18       # ~5.5 req/sec (Scopus limit is ~6/sec)
PUBMED_DELAY = 0.35       # ~3 req/sec (PubMed limit without API key)
PUBMED_BATCH_SIZE = 200   # PubMed efetch supports batch retrieval

# ── Scopus API key (from bibtool auth cache) ──────────────────────────
def _load_scopus_key() -> Optional[str]:
    """Load Scopus API key from bibtool's auth cache or environment."""
    # Try environment variable first
    import os
    key = os.environ.get("SCOPUS_API_KEY", "")
    if key:
        return key

    # Try bibtool's auth cache
    auth_cache = Path.home() / "auth_cache.json"
    if auth_cache.exists():
        try:
            data = json.loads(auth_cache.read_text())
            return data.get("scopus", "")
        except (json.JSONDecodeError, KeyError):
            pass

    return None


def fetch_abstract_scopus(scopus_id: str, api_key: str) -> Optional[str]:
    """
    Fetch abstract from Scopus Abstract Retrieval API.

    Args:
        scopus_id: Scopus document ID (e.g., "SCOPUS_ID:85123456789")
        api_key: Elsevier API key

    Returns:
        Abstract text, or None if not found/error
    """
    # Clean the scopus_id
    clean_id = scopus_id.replace("SCOPUS_ID:", "").strip()
    if not clean_id:
        return None

    url = f"https://api.elsevier.com/content/abstract/scopus_id/{clean_id}"
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            coredata = data.get("abstracts-retrieval-response", {}).get("coredata", {})
            abstract = coredata.get("dc:description", "")
            if abstract and len(abstract.strip()) > 10:
                return abstract.strip()
        elif resp.status_code == 404:
            return None
        elif resp.status_code == 429:
            # Rate limited — wait and signal caller
            time.sleep(2)
            return None
        else:
            return None
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return None

    return None


def fetch_abstracts_pubmed_batch(pmids: list[str]) -> dict[str, str]:
    """
    Fetch abstracts from PubMed E-utilities in batch.

    Args:
        pmids: List of PubMed IDs

    Returns:
        Dict mapping PMID → abstract text
    """
    if not pmids:
        return {}

    # Use efetch with XML mode for structured parsing
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(str(p) for p in pmids),
        "rettype": "xml",
        "retmode": "xml",
    }

    try:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            return {}

        # Parse XML to extract abstracts
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)

        results = {}
        for article in root.findall(".//PubmedArticle"):
            # Get PMID
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None:
                continue
            pmid = pmid_elem.text

            # Get abstract
            abstract_elem = article.find(".//Abstract")
            if abstract_elem is None:
                continue

            # Concatenate all AbstractText elements
            parts = []
            for text_elem in abstract_elem.findall("AbstractText"):
                label = text_elem.get("Label", "")
                text = text_elem.text or ""
                if text_elem.tail:
                    text += text_elem.tail
                # Include child elements (e.g., <i>, <b>)
                for child in text_elem:
                    if child.text:
                        text += child.text
                    if child.tail:
                        text += child.tail
                if label:
                    parts.append(f"{label}: {text.strip()}")
                else:
                    parts.append(text.strip())

            abstract = " ".join(parts).strip()
            if len(abstract) > 10:
                results[pmid] = abstract

        return results

    except (requests.RequestException, ET.ParseError):
        return {}


def load_abs_cache() -> dict[str, str]:
    """Load abstract cache from disk (atomic-safe)."""
    from llm_utils import load_cache as _load
    return _load(CACHE_FILE)


def save_abs_cache(cache: dict[str, str]):
    """Save abstract cache to disk (atomic-safe)."""
    from llm_utils import save_cache as _save
    _save(cache, CACHE_FILE)


def main(input_path: Optional[str] = None, force: bool = False):
    """
    Main abstract retrieval pipeline.

    1. Load papers from CSV
    2. Load cache (skip already-fetched)
    3. Try Scopus Abstract Retrieval API
    4. Fall back to PubMed E-utilities for remaining
    5. Update CSV with filled abstracts
    """
    csv_path = Path(input_path) if input_path else INPUT

    if not csv_path.exists():
        print(f"ERROR: Input file not found: {csv_path}")
        sys.exit(1)

    print("=" * 70)
    print("ABSTRACT RETRIEVAL MODULE")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"  Loaded: {len(df):,} papers from {csv_path.name}")

    # Check current abstract status
    df["abstract"] = df["abstract"].fillna("")
    has_abstract = (df["abstract"].str.len() > 10).sum()
    missing = len(df) - has_abstract
    print(f"  Already have abstracts: {has_abstract:,}")
    print(f"  Missing abstracts: {missing:,}")

    if missing == 0 and not force:
        print("  All abstracts already present. Use --force to re-fetch.")
        return

    # Load cache
    cache = load_abs_cache()
    print(f"  Cache entries: {len(cache):,}")

    # Apply cached abstracts first
    cached_filled = 0
    for idx, row in df.iterrows():
        doi = str(row.get("doi", ""))
        if doi in cache and (not row["abstract"] or len(str(row["abstract"])) <= 10):
            df.at[idx, "abstract"] = cache[doi]
            cached_filled += 1

    if cached_filled > 0:
        print(f"  Filled from cache: {cached_filled:,}")

    # Identify remaining papers needing abstracts
    still_missing = df[df["abstract"].str.len() <= 10].copy()
    print(f"  Still need to fetch: {len(still_missing):,}")

    if len(still_missing) == 0:
        _save_and_report(df, csv_path, cache)
        return

    # ── Phase 1: Scopus Abstract Retrieval API ────────────────────────
    scopus_key = _load_scopus_key()
    scopus_fetched = 0

    if scopus_key:
        print(f"\n{'─' * 70}")
        print("Phase 1: Scopus Abstract Retrieval API")
        print(f"{'─' * 70}")

        scopus_papers = still_missing[
            still_missing["scopus_id"].notna()
            & (still_missing["scopus_id"].str.len() > 3)
        ]
        print(f"  Papers with Scopus IDs: {len(scopus_papers):,}")
        print(f"  Estimated time: ~{len(scopus_papers) * SCOPUS_DELAY / 60:.1f} min")

        for i, (idx, row) in enumerate(scopus_papers.iterrows()):
            doi = str(row.get("doi", ""))

            # Skip if already cached
            if doi in cache:
                df.at[idx, "abstract"] = cache[doi]
                scopus_fetched += 1
                continue

            abstract = fetch_abstract_scopus(str(row["scopus_id"]), scopus_key)
            if abstract:
                df.at[idx, "abstract"] = abstract
                cache[doi] = abstract
                scopus_fetched += 1

            # Progress
            if (i + 1) % 100 == 0 or (i + 1) == len(scopus_papers):
                print(f"  Scopus: {i+1:,}/{len(scopus_papers):,} "
                      f"(fetched: {scopus_fetched:,})")
                save_abs_cache(cache)  # Incremental save

            time.sleep(SCOPUS_DELAY)

        save_abs_cache(cache)
        print(f"  Scopus total fetched: {scopus_fetched:,}")
    else:
        print("\n  ⚠ No Scopus API key found. Skipping Scopus retrieval.")
        print("    Set SCOPUS_API_KEY env var or run bibtool scan first.")

    # ── Phase 2: PubMed E-utilities fallback ──────────────────────────
    print(f"\n{'─' * 70}")
    print("Phase 2: PubMed E-utilities Fallback")
    print(f"{'─' * 70}")

    # Re-check what's still missing
    still_missing2 = df[df["abstract"].str.len() <= 10].copy()
    pubmed_papers = still_missing2[
        still_missing2["pmid"].notna()
        & (still_missing2["pmid"].astype(str).str.len() > 1)
        & (still_missing2["pmid"].astype(str) != "nan")
    ]
    print(f"  Papers still missing with PMIDs: {len(pubmed_papers):,}")

    if len(pubmed_papers) > 0:
        # Batch fetch from PubMed
        pmid_to_idx = {}
        for idx, row in pubmed_papers.iterrows():
            pmid = str(row["pmid"]).split(".")[0].strip()  # Remove decimal
            if pmid.isdigit():
                pmid_to_idx[pmid] = idx

        all_pmids = list(pmid_to_idx.keys())
        pubmed_fetched = 0

        for batch_start in range(0, len(all_pmids), PUBMED_BATCH_SIZE):
            batch = all_pmids[batch_start:batch_start + PUBMED_BATCH_SIZE]
            results = fetch_abstracts_pubmed_batch(batch)

            for pmid, abstract in results.items():
                idx = pmid_to_idx.get(pmid)
                if idx is not None:
                    df.at[idx, "abstract"] = abstract
                    doi = str(df.at[idx, "doi"])
                    cache[doi] = abstract
                    pubmed_fetched += 1

            print(f"  PubMed batch: {min(batch_start + PUBMED_BATCH_SIZE, len(all_pmids)):,}/"
                  f"{len(all_pmids):,} (fetched: {pubmed_fetched:,})")
            save_abs_cache(cache)
            time.sleep(PUBMED_DELAY)

        print(f"  PubMed total fetched: {pubmed_fetched:,}")

    # ── Save results ──────────────────────────────────────────────────
    _save_and_report(df, csv_path, cache)


def _save_and_report(df: pd.DataFrame, csv_path: Path, cache: dict):
    """Save updated CSV and print summary."""
    save_abs_cache(cache)

    # Replace empty strings with NaN for clean CSV
    df.loc[df["abstract"].str.len() <= 10, "abstract"] = pd.NA

    df.to_csv(csv_path, index=False)

    # Final report
    has_abstract = df["abstract"].notna().sum()
    total = len(df)
    coverage = has_abstract / total * 100 if total > 0 else 0

    print(f"\n{'=' * 70}")
    print("ABSTRACT RETRIEVAL COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total papers:      {total:,}")
    print(f"  With abstracts:    {has_abstract:,} ({coverage:.1f}%)")
    print(f"  Still missing:     {total - has_abstract:,} ({100 - coverage:.1f}%)")
    print(f"  Cache file:        {CACHE_FILE}")
    print(f"  Updated CSV:       {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch missing abstracts for papers")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if abstracts exist")
    args = parser.parse_args()
    main(input_path=args.input, force=args.force)
