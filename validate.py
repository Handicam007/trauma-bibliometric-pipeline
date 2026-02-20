#!/usr/bin/env python3
"""
PRE-FLIGHT VALIDATION
======================
Run this BEFORE filter_results.py or analysis.py to catch mistakes early.
Checks regexes, keyword lists, concept names, and config consistency.

Usage:
    python validate.py              # run all checks
    python validate.py --dry-run    # also preview exclusion/inclusion impact
"""

import re
import sys
from pathlib import Path
from collections import Counter

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

errors = []
warnings = []


def check(label, passed, detail=""):
    if passed:
        print(f"  [{PASS}] {label}")
    else:
        msg = f"{label}: {detail}" if detail else label
        errors.append(msg)
        print(f"  [{FAIL}] {msg}")


def warn(label, detail=""):
    msg = f"{label}: {detail}" if detail else label
    warnings.append(msg)
    print(f"  [{WARN}] {msg}")


# ═══════════════════════════════════════════════════════════════════
# CHECK 1: REGEX PATTERNS IN concept_definitions.py
# ═══════════════════════════════════════════════════════════════════

def check_regexes():
    print("\n" + "=" * 60)
    print("CHECK 1: REGEX PATTERNS (concept_definitions.py)")
    print("=" * 60)

    from concept_definitions import CLINICAL_CONCEPTS, DOMAIN_GROUPS

    # 1a. Every regex must compile
    for concept, pattern in CLINICAL_CONCEPTS.items():
        try:
            re.compile(pattern, re.IGNORECASE)
            # Also check it's not empty
            check(f"Regex compiles: {concept}", len(pattern) > 0)
        except re.error as e:
            check(f"Regex compiles: {concept}", False,
                  f"INVALID REGEX -> {e}\n"
                  f"         Pattern: {pattern[:80]}...")

    # 1b. Every concept in DOMAIN_GROUPS must exist in CLINICAL_CONCEPTS
    print()
    all_grouped = set()
    for group_name, concept_list in DOMAIN_GROUPS.items():
        for concept in concept_list:
            all_grouped.add(concept)
            check(f"Domain group '{group_name}' -> '{concept}' exists",
                  concept in CLINICAL_CONCEPTS,
                  f"'{concept}' is NOT a key in CLINICAL_CONCEPTS (typo?)")

    # 1c. Warn about concepts NOT in any domain group
    ungrouped = set(CLINICAL_CONCEPTS.keys()) - all_grouped
    if ungrouped:
        for c in sorted(ungrouped):
            warn(f"Concept '{c}' is not in any DOMAIN_GROUP",
                 "It won't appear in the domain grid figures (fig08, fig18)")

    # 1d. Check for TOP_CONCEPTS_FOR_CHARTS consistency
    try:
        from concept_definitions import TOP_CONCEPTS_FOR_CHARTS
        print()
        for concept in TOP_CONCEPTS_FOR_CHARTS:
            check(f"TOP_CONCEPTS_FOR_CHARTS -> '{concept}' exists",
                  concept in CLINICAL_CONCEPTS,
                  f"'{concept}' is NOT a key in CLINICAL_CONCEPTS (typo?)")
    except ImportError:
        pass  # Optional list


# ═══════════════════════════════════════════════════════════════════
# CHECK 2: KEYWORD LISTS IN filter_results.py
# ═══════════════════════════════════════════════════════════════════

def check_keyword_lists():
    print("\n" + "=" * 60)
    print("CHECK 2: KEYWORD LISTS (filter_results.py)")
    print("=" * 60)

    from filter_results import (
        EXCLUDE_TITLE_KEYWORDS, EXCLUDE_ABSTRACT_KEYWORDS,
        REQUIRE_ANY_KEYWORD, TOP_TRAUMA_JOURNALS,
    )

    all_lists = {
        "EXCLUDE_TITLE_KEYWORDS": EXCLUDE_TITLE_KEYWORDS,
        "EXCLUDE_ABSTRACT_KEYWORDS": EXCLUDE_ABSTRACT_KEYWORDS,
        "REQUIRE_ANY_KEYWORD": REQUIRE_ANY_KEYWORD,
        "TOP_TRAUMA_JOURNALS": TOP_TRAUMA_JOURNALS,
    }

    for name, kw_list in all_lists.items():
        print(f"\n  --- {name} ({len(kw_list)} entries) ---")

        # 2a. Check for suspiciously long entries (likely missing comma = merged strings)
        for i, kw in enumerate(kw_list):
            if len(kw) > 60:
                warn(f"  Entry #{i+1} is {len(kw)} chars long (possible missing comma?)",
                     f'"{kw[:40]}...{kw[-20:]}"')

        # 2b. Check for duplicates
        counts = Counter(kw_list)
        dups = {k: v for k, v in counts.items() if v > 1}
        if dups:
            for kw, count in dups.items():
                warn(f"  Duplicate entry in {name}", f'"{kw}" appears {count} times')
        else:
            check(f"No duplicates in {name}", True)

        # 2c. Check for empty strings
        empties = [i for i, kw in enumerate(kw_list) if not kw.strip()]
        if empties:
            check(f"No empty entries in {name}", False,
                  f"Empty string at position(s): {empties}")
        else:
            check(f"No empty entries in {name}", True)

        # 2d. Check for entries with no spaces but very long (merged string trap)
        for i, kw in enumerate(kw_list):
            if len(kw) > 30 and " " not in kw and "-" not in kw:
                warn(f"  Entry #{i+1} in {name} has no spaces and is {len(kw)} chars",
                     f'"{kw}" -- likely two keywords merged by a missing comma')

    # 2e. Check for contradictions (same term in both exclude and include)
    exclude_set = set(k.lower() for k in EXCLUDE_TITLE_KEYWORDS)
    include_set = set(k.lower() for k in REQUIRE_ANY_KEYWORD)
    conflicts = exclude_set & include_set
    if conflicts:
        print()
        for c in sorted(conflicts):
            warn(f"Keyword '{c}' is in BOTH exclude and include lists",
                 "This is contradictory -- the exclusion runs first, so matching papers will be removed before inclusion is checked")
    else:
        print()
        check("No contradictions between exclude and include lists", True)


# ═══════════════════════════════════════════════════════════════════
# CHECK 3: config.py CONSISTENCY
# ═══════════════════════════════════════════════════════════════════

def check_config():
    print("\n" + "=" * 60)
    print("CHECK 3: CONFIG SETTINGS (config.py)")
    print("=" * 60)

    from config import (
        FIELD_NAME, FIELD_SHORT, YEAR_MIN, YEAR_MAX, BASE_YEAR,
        TREND_EARLY, TREND_LATE, GEO_FILTER_SCOPUS,
        GEO_HIGHLIGHT_COUNTRIES, GEO_LABEL,
        GEO_PRIMARY_REGEX, GEO_SECONDARY_REGEX,
        WORDCLOUD_STOPWORDS, JOURNAL_HIGHLIGHT_KEYWORDS,
        TRENDING_MIN_PAPERS, FIG_DIR_NAME,
        CITE_WEIGHT, RECENCY_WEIGHT, JOURNAL_BONUS,
    )

    check("FIELD_NAME is set", len(FIELD_NAME) > 0, "FIELD_NAME is empty")
    check("FIELD_SHORT is set", len(FIELD_SHORT) > 0, "FIELD_SHORT is empty")
    check(f"YEAR_MIN ({YEAR_MIN}) < YEAR_MAX ({YEAR_MAX})", YEAR_MIN < YEAR_MAX)
    check(f"BASE_YEAR ({BASE_YEAR}) = YEAR_MIN-1 ({YEAR_MIN-1})", BASE_YEAR == YEAR_MIN - 1,
          f"BASE_YEAR should be {YEAR_MIN - 1}")
    check(f"TREND_EARLY ({TREND_EARLY}) starts at YEAR_MIN", TREND_EARLY[0] == YEAR_MIN)
    check(f"TREND_LATE ({TREND_LATE}) ends at YEAR_MAX", TREND_LATE[1] == YEAR_MAX)
    check(f"Trend periods don't overlap",
          TREND_EARLY[1] < TREND_LATE[0],
          f"Early ends at {TREND_EARLY[1]}, Late starts at {TREND_LATE[0]}")
    check(f"No gap between trend periods",
          TREND_LATE[0] == TREND_EARLY[1] + 1,
          f"Gap between {TREND_EARLY[1]} and {TREND_LATE[0]}")

    if GEO_PRIMARY_REGEX:
        try:
            re.compile(GEO_PRIMARY_REGEX)
            check("GEO_PRIMARY_REGEX compiles", True)
        except re.error as e:
            check("GEO_PRIMARY_REGEX compiles", False, str(e))

    if GEO_SECONDARY_REGEX:
        try:
            re.compile(GEO_SECONDARY_REGEX)
            check("GEO_SECONDARY_REGEX compiles", True)
        except re.error as e:
            check("GEO_SECONDARY_REGEX compiles", False, str(e))

    check("WORDCLOUD_STOPWORDS has entries", len(WORDCLOUD_STOPWORDS) > 0,
          "Word cloud will be dominated by obvious field terms")
    check("JOURNAL_HIGHLIGHT_KEYWORDS has entries", len(JOURNAL_HIGHLIGHT_KEYWORDS) > 0,
          "No journals will be highlighted in the bar chart")
    check(f"TRENDING_MIN_PAPERS ({TRENDING_MIN_PAPERS}) is reasonable",
          2 <= TRENDING_MIN_PAPERS <= 50,
          f"Value {TRENDING_MIN_PAPERS} seems too {'low' if TRENDING_MIN_PAPERS < 2 else 'high'}")
    check("Scoring weights are positive",
          CITE_WEIGHT > 0 and RECENCY_WEIGHT > 0 and JOURNAL_BONUS > 0,
          "Negative weights would invert the ranking")


# ═══════════════════════════════════════════════════════════════════
# CHECK 4: DRY RUN — preview exclusion/inclusion impact
# ═══════════════════════════════════════════════════════════════════

def check_dry_run():
    print("\n" + "=" * 60)
    print("CHECK 4: DRY RUN — Exclusion/Inclusion Impact Preview")
    print("=" * 60)

    import pandas as pd
    raw_path = Path(__file__).parent / "results_refined" / "all_results.csv"
    if not raw_path.exists():
        warn("Cannot run dry-run preview", f"File not found: {raw_path}")
        print("  (Run the search script first to generate raw results)")
        return

    df = pd.read_csv(raw_path)
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    n_total = len(df)
    print(f"\n  Raw corpus: {n_total:,} papers")

    from filter_results import EXCLUDE_TITLE_KEYWORDS, EXCLUDE_ABSTRACT_KEYWORDS, REQUIRE_ANY_KEYWORD

    # --- Per-keyword exclusion impact ---
    print(f"\n  --- EXCLUSION IMPACT (per keyword) ---")
    title_lower = df["title"].str.lower()
    abstract_lower = df["abstract"].str.lower()

    exclusion_impacts = []
    for kw in EXCLUDE_TITLE_KEYWORDS:
        n_hit = title_lower.str.contains(kw, case=False, na=False).sum()
        if n_hit > 0:
            exclusion_impacts.append((kw, n_hit, "title"))

    for kw in EXCLUDE_ABSTRACT_KEYWORDS:
        n_hit = abstract_lower.str.contains(kw, case=False, na=False).sum()
        if n_hit > 0:
            exclusion_impacts.append((kw, n_hit, "abstract"))

    # Sort by impact descending
    exclusion_impacts.sort(key=lambda x: x[1], reverse=True)

    for kw, n_hit, source in exclusion_impacts:
        pct = n_hit / n_total * 100
        flag = ""
        if pct > 10:
            flag = f"  <-- !! REMOVES {pct:.1f}% OF CORPUS"
            warn(f"Exclusion keyword '{kw}' removes {pct:.1f}% of corpus ({n_hit:,} papers)")
        elif pct > 5:
            flag = f"  <-- WARNING: {pct:.1f}%"
            warn(f"Exclusion keyword '{kw}' removes {pct:.1f}% of corpus ({n_hit:,} papers)")
        print(f"    {n_hit:>5} papers ({pct:>5.1f}%) | [{source}] \"{kw}\"{flag}")

    if not exclusion_impacts:
        print("    (no exclusion keywords matched any papers)")

    # Total exclusion
    exclude_mask = pd.Series(False, index=df.index)
    for kw in EXCLUDE_TITLE_KEYWORDS:
        exclude_mask |= title_lower.str.contains(kw, case=False, na=False)
    for kw in EXCLUDE_ABSTRACT_KEYWORDS:
        exclude_mask |= abstract_lower.str.contains(kw, case=False, na=False)

    n_excluded = exclude_mask.sum()
    n_after_excl = n_total - n_excluded
    print(f"\n  Total excluded: {n_excluded:,} ({n_excluded/n_total*100:.1f}%)")
    print(f"  After exclusion: {n_after_excl:,}")

    # --- Inclusion impact ---
    print(f"\n  --- INCLUSION FILTER ---")
    df_post_excl = df[~exclude_mask].copy()
    combined = df_post_excl["title"].str.lower() + " " + df_post_excl["abstract"].str.lower()
    include_mask = pd.Series(False, index=df_post_excl.index)
    for kw in REQUIRE_ANY_KEYWORD:
        include_mask |= combined.str.contains(kw, case=False, na=False)

    n_included = include_mask.sum()
    n_dropped = len(df_post_excl) - n_included
    survival_rate = n_included / len(df_post_excl) * 100

    print(f"  Papers matching inclusion keywords: {n_included:,} ({survival_rate:.1f}%)")
    print(f"  Papers dropped (no keyword match): {n_dropped:,}")
    print(f"  Final corpus size: {n_included:,}")

    if survival_rate < 50:
        warn("Low inclusion survival rate",
             f"Only {survival_rate:.1f}% of papers passed the inclusion filter. "
             "Consider adding more inclusion keywords.")
    elif survival_rate > 95:
        check(f"Inclusion filter survival rate: {survival_rate:.1f}%", True)
    else:
        check(f"Inclusion filter survival rate: {survival_rate:.1f}%", True)

    # --- Concept coverage preview ---
    print(f"\n  --- CONCEPT COVERAGE PREVIEW ---")
    from concept_definitions import CLINICAL_CONCEPTS

    titles_lower = df_post_excl[include_mask]["title"].str.lower()
    n_corpus = len(titles_lower)
    zero_concepts = []

    for concept, pattern in CLINICAL_CONCEPTS.items():
        try:
            n_match = titles_lower.str.contains(pattern, regex=True, na=False).sum()
        except re.error as e:
            check(f"Concept '{concept}' regex", False, str(e))
            continue

        if n_match == 0:
            zero_concepts.append(concept)
            warn(f"Concept '{concept}' matches 0 papers",
                 "Check if the regex is too restrictive for your corpus")
        elif n_match < 3:
            warn(f"Concept '{concept}' matches only {n_match} papers",
                 "May not be meaningful for trending analysis")

    if zero_concepts:
        print(f"\n  Zero-match concepts ({len(zero_concepts)}):")
        for c in zero_concepts:
            print(f"    - {c}")
    else:
        check("All concepts match at least 1 paper", True)

    # ── Concept spot-check: sample matched titles for human validation ──
    print(f"\n  --- CONCEPT SPOT-CHECK (sample matched titles) ---")
    print(f"  Review these samples to verify regex precision.")
    print(f"  If a concept is matching irrelevant titles, tighten the regex.\n")

    all_titles = df_post_excl[include_mask]["title"].values
    titles_lower_arr = titles_lower.values
    n_samples = 3  # titles per concept

    suspect_count = 0
    for concept, pattern in CLINICAL_CONCEPTS.items():
        try:
            matches = [all_titles[i] for i, t in enumerate(titles_lower_arr)
                       if re.search(pattern, t)]
        except re.error:
            continue

        if len(matches) == 0:
            continue

        # Sample up to n_samples titles
        import random
        random.seed(42)
        sample = random.sample(matches, min(n_samples, len(matches)))

        # Print header
        n_match = len(matches)
        print(f"  [{concept}] (n={n_match})")
        for t in sample:
            short = t[:90] + "..." if len(t) > 90 else t
            print(f"      → {short}")
        print()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PRE-FLIGHT VALIDATION")
    print("Checking config, regexes, keyword lists, and concept names...")
    print("=" * 60)

    check_regexes()
    check_keyword_lists()
    check_config()

    # Dry run only if --dry-run flag or raw data exists
    if "--dry-run" in sys.argv:
        check_dry_run()
    else:
        raw_path = Path(__file__).parent / "results_refined" / "all_results.csv"
        if raw_path.exists():
            check_dry_run()
        else:
            print(f"\n  (Skipping dry-run preview -- no raw data at {raw_path})")
            print(f"  (Run 'python validate.py --dry-run' after search completes)")

    # --- Final verdict ---
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    if errors:
        print(f"\n  {FAIL} {len(errors)} ERROR(S) FOUND — fix before running pipeline:")
        for e in errors:
            print(f"    !! {e}")
    else:
        print(f"\n  {PASS} No errors found.")

    if warnings:
        print(f"\n  {WARN} {len(warnings)} WARNING(S) — review these:")
        for w in warnings:
            print(f"    >> {w}")
    else:
        print(f"  No warnings.")

    if errors:
        print(f"\n  Run 'python validate.py' again after fixing errors.")
        sys.exit(1)
    else:
        print(f"\n  Pipeline is ready to run.")
        sys.exit(0)


if __name__ == "__main__":
    main()
