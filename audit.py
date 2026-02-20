#!/usr/bin/env python3
"""
FULL AUDIT — verify every number used in figures and narration.
All numbers are computed from data; nothing is hardcoded.
"""

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np

from config import (
    TREND_EARLY, TREND_LATE, YEAR_STATS_MAX,
    GEO_PRIMARY_REGEX, GEO_SECONDARY_REGEX,
    GEO_PRIMARY_LABEL, GEO_SECONDARY_LABEL,
)

RAW = Path(__file__).parent / "results_refined" / "all_results.csv"
FILTERED = Path(__file__).parent / "results_curated" / "all_filtered.csv"


def section(title):
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def main():
    # --- Check files exist ---
    for path, label in [(RAW, "Raw results"), (FILTERED, "Filtered results")]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            print("Run the search and filter scripts first.")
            sys.exit(1)

    errors = []

    # ── AUDIT 1: DATA PIPELINE NUMBERS ────────────────────────────
    section("AUDIT 1: DATA PIPELINE NUMBERS")

    df_raw = pd.read_csv(RAW)
    print(f"Raw results file rows: {len(df_raw)}")
    print(f"  Unique DOIs in raw: {df_raw['doi'].nunique()}")

    df = pd.read_csv(FILTERED)
    df["title"] = df["title"].fillna("")
    df["publication"] = df["publication"].fillna("")
    df["affiliation_country"] = df["affiliation_country"].fillna("")
    df["citations_count"] = pd.to_numeric(df["citations_count"], errors="coerce").fillna(0).astype(int)

    print(f"\nFiltered results file rows: {len(df)}")
    print(f"  Unique DOIs in filtered: {df['doi'].nunique()}")

    # Re-compute exclusion numbers from raw data
    from filter_results import EXCLUDE_TITLE_KEYWORDS, EXCLUDE_ABSTRACT_KEYWORDS, REQUIRE_ANY_KEYWORD
    df_raw_temp = pd.read_csv(RAW)
    df_raw_temp["title"] = df_raw_temp["title"].fillna("")
    df_raw_temp["abstract"] = df_raw_temp["abstract"].fillna("")
    title_lower = df_raw_temp["title"].str.lower()
    abstract_lower = df_raw_temp["abstract"].str.lower()
    exclude_mask = pd.Series(False, index=df_raw_temp.index)
    for kw in EXCLUDE_TITLE_KEYWORDS:
        exclude_mask |= title_lower.str.contains(kw, case=False, na=False)
    for kw in EXCLUDE_ABSTRACT_KEYWORDS:
        exclude_mask |= abstract_lower.str.contains(kw, case=False, na=False)
    n_excluded = exclude_mask.sum()
    n_post_exclusion = len(df_raw) - n_excluded

    print(f"\n  Pipeline breakdown:")
    print(f"    Raw:              {len(df_raw):,}")
    print(f"    Excluded:         {n_excluded:,}")
    print(f"    Post-exclusion:   {n_post_exclusion:,}")
    print(f"    Final (filtered): {len(df):,}")

    # ── AUDIT 2: YEAR DISTRIBUTION ────────────────────────────────
    section("AUDIT 2: YEAR DISTRIBUTION")
    year_counts = df["year"].value_counts().sort_index()
    for y, c in year_counts.items():
        print(f"  {y}: {c}")
    print(f"  Total: {year_counts.sum()}")
    print(f"  Year range: {df['year'].min()} - {df['year'].max()}")

    # ── AUDIT 3: GEOGRAPHIC NUMBERS ───────────────────────────────
    section("AUDIT 3: GEOGRAPHIC NUMBERS")
    primary_mask = df["affiliation_country"].str.contains(GEO_PRIMARY_REGEX, case=False, na=False)
    print(f"  {GEO_PRIMARY_LABEL}: {primary_mask.sum()} ({primary_mask.sum()/len(df)*100:.1f}%)")

    if GEO_SECONDARY_REGEX:
        secondary_mask = df["affiliation_country"].str.contains(GEO_SECONDARY_REGEX, case=False, na=False)
        print(f"  {GEO_SECONDARY_LABEL}: {secondary_mask.sum()} ({secondary_mask.sum()/len(df)*100:.1f}%)")

    # Country counts
    countries = df["affiliation_country"].fillna("").str.split("|")
    country_flat = [c.strip() for sublist in countries for c in sublist if c.strip()]
    country_counts = Counter(country_flat)
    print(f"\n  Top 5 countries:")
    for c, n in country_counts.most_common(5):
        print(f"    {c}: {n}")

    # ── AUDIT 4: TOP JOURNALS ─────────────────────────────────────
    section("AUDIT 4: TOP JOURNALS")
    for j, c in df["publication"].value_counts().head(10).items():
        print(f"  {c:>3} -- {j}")

    # ── AUDIT 5: CITATION STATS ───────────────────────────────────
    section("AUDIT 5: CITATION STATS")
    high_impact = (df["citations_count"] >= 20).sum()
    print(f"  Papers >=20 cites: {high_impact}")
    hot = ((df["year"] >= 2023) & (df["citations_count"] >= 5)).sum()
    print(f"  Hot topics (2023+, >=5 cites): {hot}")
    hot10 = ((df["year"] >= 2023) & (df["citations_count"] >= 10)).sum()
    print(f"  Hot topics (2023+, >=10 cites): {hot10}")
    cutting = (df["year"] >= 2025).sum()
    print(f"  Cutting edge (2025+): {cutting}")

    # ── AUDIT 6: CLINICAL CONCEPT COUNTS ──────────────────────────
    section("AUDIT 6: CLINICAL CONCEPT COUNTS (title-based)")

    from concept_definitions import CLINICAL_CONCEPTS
    titles_lower = df["title"].str.lower()
    concept_totals = {}
    for concept, pattern in CLINICAL_CONCEPTS.items():
        total = titles_lower.str.contains(pattern, regex=True, na=False).sum()
        concept_totals[concept] = total
        print(f"  {concept:>30}: {total}")

    # Coverage stat
    any_match = pd.Series(False, index=df.index)
    for pattern in CLINICAL_CONCEPTS.values():
        any_match |= titles_lower.str.contains(pattern, regex=True, na=False)
    coverage = any_match.sum() / len(df) * 100
    print(f"\n  Concept coverage: {any_match.sum()}/{len(df)} ({coverage:.1f}%)")

    # ── AUDIT 7: CONCEPT x YEAR MATRIX ────────────────────────────
    section("AUDIT 7: CONCEPT x YEAR MATRIX (top 6 concepts)")

    top6 = sorted(concept_totals, key=concept_totals.get, reverse=True)[:6]
    years = sorted(df["year"].unique())
    header = f"{'Concept':>25} | " + " | ".join(str(y) for y in years) + " | TOTAL"
    print(header)
    print("-" * len(header))
    for concept in top6:
        pattern = CLINICAL_CONCEPTS[concept]
        counts = []
        for y in years:
            c = df[df["year"] == y]["title"].str.lower().str.contains(pattern, regex=True, na=False).sum()
            counts.append(c)
        total = sum(counts)
        row = f"{concept:>25} | " + " | ".join(f"{c:>4}" for c in counts) + f" | {total:>5}"
        print(row)

    # ── AUDIT 8: TRENDING CONCEPTS ────────────────────────────────
    section(f"AUDIT 8: TRENDING CONCEPTS ({TREND_EARLY[0]}-{TREND_EARLY[1]} vs {TREND_LATE[0]}-{TREND_LATE[1]}, complete years only)")

    df_stats = df[df["year"] <= YEAR_STATS_MAX]
    early = df_stats[df_stats["year"].between(TREND_EARLY[0], TREND_EARLY[1])]
    late = df_stats[df_stats["year"].between(TREND_LATE[0], TREND_LATE[1])]
    n_early = len(early)
    n_late = len(late)
    print(f"  Early period ({TREND_EARLY[0]}-{TREND_EARLY[1]}): {n_early} papers")
    print(f"  Late period ({TREND_LATE[0]}-{TREND_LATE[1]}): {n_late} papers")

    # Show trending for top 8 concepts by total volume
    top8 = sorted(concept_totals, key=concept_totals.get, reverse=True)[:8]
    for concept in top8:
        pattern = CLINICAL_CONCEPTS[concept]
        early_count = early["title"].str.lower().str.contains(pattern, regex=True, na=False).sum()
        late_count = late["title"].str.lower().str.contains(pattern, regex=True, na=False).sum()
        early_pct = early_count / n_early * 100 if n_early > 0 else 0
        late_pct = late_count / n_late * 100 if n_late > 0 else 0
        if early_pct > 0:
            change = ((late_pct - early_pct) / early_pct) * 100
        else:
            change = float("inf")
        print(f"  {concept:>25}: early={early_count} ({early_pct:.2f}%), late={late_count} ({late_pct:.2f}%), change={change:+.1f}%")

    # ── AUDIT 9: QUERY CATEGORY DISTRIBUTION ──────────────────────
    section("AUDIT 9: QUERY CATEGORY DISTRIBUTION")
    for q, c in df["query"].value_counts().items():
        print(f"  {c:>4} -- {str(q)[:80]}")

    # ── AUDIT 10: DATA INTEGRITY ──────────────────────────────────
    section("AUDIT 10: DATA INTEGRITY")
    dup_dois = df["doi"].duplicated().sum()
    print(f"  Duplicate DOIs: {dup_dois}")
    if dup_dois > 0:
        errors.append(f"Found {dup_dois} duplicate DOIs!")

    null_titles = (df["title"].str.len() < 3).sum()
    print(f"  Empty/very short titles: {null_titles}")
    if null_titles > 0:
        errors.append(f"Found {null_titles} empty/short titles!")

    null_years = df["year"].isna().sum()
    print(f"  Missing years: {null_years}")
    if null_years > 0:
        errors.append(f"Found {null_years} missing years!")

    print(f"  Citation range: {df['citations_count'].min()} - {df['citations_count'].max()}")
    print(f"  Mean citations: {df['citations_count'].mean():.1f}")
    print(f"  Median citations: {df['citations_count'].median():.1f}")

    # ── FINAL VERDICT ─────────────────────────────────────────────
    section("AUDIT COMPLETE")
    if errors:
        print("  ISSUES FOUND:")
        for e in errors:
            print(f"    !! {e}")
    else:
        print("  All checks passed.")


if __name__ == "__main__":
    main()
