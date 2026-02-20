#!/usr/bin/env python3
"""
Refined Trauma Acute Care Literature Search
=============================================
Targeted queries for a Level 1 trauma centre journal club on:
  - Innovations (new tools, techniques, technologies)
  - Hot topics & future directions in research
  - Unmet needs & gaps
  - What a trauma surgeon in Montreal needs to know for the future

Uses Scopus advanced query syntax for precision.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "bibtool" / "src"))

from bibtool.search_papers import LiteratureSearch, SearchQuery

OUTPUT_DIR = Path(__file__).parent / "results_refined"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# SCOPUS ADVANCED QUERIES
# ============================================================
# Scopus supports: TITLE-ABS-KEY(), AND, OR, W/ (proximity),
# SRCTITLE() for journal filtering, DOCTYPE() for article types
#
# We restrict to high-impact trauma & surgery journals where possible,
# and use precise Boolean logic to cut noise.
# ============================================================

QUERIES = {
    # ----------------------------------------------------------
    # 1. REVIEW / FUTURE DIRECTIONS / STATE OF THE ART
    # ----------------------------------------------------------
    "future_directions_reviews": SearchQuery(
        'TITLE-ABS-KEY(trauma AND "acute care surgery") '
        'AND TITLE-ABS-KEY("future direction*" OR "state of the art" OR "current state" '
        'OR "emerging" OR "paradigm shift" OR "changing landscape" OR "evolution") '
        'AND DOCTYPE(ar OR re)',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 2. UNMET NEEDS / GAPS / CHALLENGES IN TRAUMA
    # ----------------------------------------------------------
    "unmet_needs_gaps": SearchQuery(
        'TITLE-ABS-KEY(trauma AND (surgery OR surgical OR "acute care")) '
        'AND TITLE-ABS-KEY("unmet need*" OR "knowledge gap*" OR "research gap*" '
        'OR "research priorit*" OR "challenges" OR "barriers" OR "disparit*") '
        'AND DOCTYPE(ar OR re)',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 3. DAMAGE CONTROL SURGERY & RESUSCITATION - INNOVATIONS
    # ----------------------------------------------------------
    "damage_control_innovations": SearchQuery(
        'TITLE-ABS-KEY("damage control" AND (resuscitation OR surgery) AND trauma) '
        'AND TITLE-ABS-KEY(novel OR innovation OR update OR "new approach" '
        'OR guideline OR "best practice" OR protocol)',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 4. REBOA / ENDOVASCULAR TRAUMA
    # ----------------------------------------------------------
    "reboa_endovascular": SearchQuery(
        'TITLE-ABS-KEY(REBOA OR "resuscitative endovascular balloon" '
        'OR "endovascular resuscitation" OR "aortic occlusion" OR "ER-REBOA") '
        'AND TITLE-ABS-KEY(trauma OR hemorrha*)',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 5. WHOLE BLOOD / MASSIVE TRANSFUSION / HEMOSTATIC RESUSCITATION
    # ----------------------------------------------------------
    "transfusion_hemostasis": SearchQuery(
        'TITLE-ABS-KEY(trauma AND ("whole blood" OR "massive transfusion protocol" '
        'OR "hemostatic resuscitation" OR "low titer O" OR "cold stored" '
        'OR "viscoelastic" OR "TEG" OR "ROTEM" OR "fibrinogen")) '
        'AND TITLE-ABS-KEY(innovat* OR outcome* OR trial OR guideline OR protocol)',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 6. AI / MACHINE LEARNING / CLINICAL DECISION SUPPORT IN TRAUMA
    # ----------------------------------------------------------
    "ai_trauma": SearchQuery(
        'TITLE-ABS-KEY(("artificial intelligence" OR "machine learning" OR "deep learning" '
        'OR "clinical decision support" OR "predictive model*" OR "natural language processing") '
        'AND trauma AND (surgery OR triage OR outcome* OR mortality))',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 7. POCUS / IMAGING INNOVATION IN TRAUMA
    # ----------------------------------------------------------
    "pocus_imaging": SearchQuery(
        'TITLE-ABS-KEY((POCUS OR "point-of-care ultrasound" OR eFAST OR "CT perfusion" '
        'OR "whole-body CT" OR "contrast-enhanced ultrasound") '
        'AND trauma AND (innovat* OR novel OR "new" OR advance* OR outcome*))',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 8. TRAUMA SYSTEMS / QUALITY IMPROVEMENT / REGIONALIZATION
    # ----------------------------------------------------------
    "trauma_systems_qi": SearchQuery(
        'TITLE-ABS-KEY("trauma system*" OR "trauma cent*" OR "trauma network") '
        'AND TITLE-ABS-KEY("quality improvement" OR "performance improvement" '
        'OR "benchmarking" OR "regionalization" OR "transfer" OR "undertriage" '
        'OR "overtriage" OR "workforce" OR "surgeon shortage")',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 9. NON-OPERATIVE / INTERVENTIONAL / HYBRID APPROACHES
    # ----------------------------------------------------------
    "non_operative_hybrid": SearchQuery(
        'TITLE-ABS-KEY(trauma AND ("non-operative management" OR "nonoperative management" '
        'OR "interventional radiology" OR "angioembolization" OR "hybrid operating" '
        'OR "endovascular" OR "minimally invasive")) '
        'AND TITLE-ABS-KEY(innovat* OR outcome* OR trend* OR evolv* OR advance*)',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 10. TBI & NEUROTRAUMA - WHAT'S NEW
    # ----------------------------------------------------------
    "tbi_neurotrauma": SearchQuery(
        'TITLE-ABS-KEY(("traumatic brain injury" OR "TBI" OR neurotrauma) '
        'AND ("decompressive craniectomy" OR "ICP monitoring" OR "neuroprotect*" '
        'OR "brain tissue oxygen" OR "multimodal monitoring" OR guideline OR "new")) '
        'AND TITLE-ABS-KEY(innovat* OR advance* OR update OR "state of the art" OR trial)',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 11. GERIATRIC TRAUMA (growing problem in Montreal)
    # ----------------------------------------------------------
    "geriatric_trauma": SearchQuery(
        'TITLE-ABS-KEY(("geriatric trauma" OR "elderly trauma" OR "older adult* trauma" '
        'OR "frailty" AND trauma) AND (management OR outcome* OR innovat* OR protocol '
        'OR "anticoagul*" OR "hip fracture" OR guideline))',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 12. TRAUMA-INDUCED COAGULOPATHY / TIC / ATC
    # ----------------------------------------------------------
    "coagulopathy": SearchQuery(
        'TITLE-ABS-KEY(("trauma-induced coagulopathy" OR "acute traumatic coagulopathy" '
        'OR "TIC" OR "acute coagulopathy of trauma") '
        'AND (mechanism OR pathophysiology OR treatment OR management OR "new" OR advance*))',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 13. CANADIAN TRAUMA / MONTREAL SPECIFIC
    # ----------------------------------------------------------
    "canadian_trauma": SearchQuery(
        'TITLE-ABS-KEY(trauma AND (surgery OR surgical OR "acute care")) '
        'AND AFFILCOUNTRY(canada) '
        'AND TITLE-ABS-KEY(innovat* OR outcome* OR system* OR guideline OR protocol '
        'OR quality OR "level 1")',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 14. SIMULATION / TRAINING / COMPETENCY IN TRAUMA SURGERY
    # ----------------------------------------------------------
    "training_simulation": SearchQuery(
        'TITLE-ABS-KEY(("trauma surgery" OR "acute care surgery") '
        'AND (simulation OR "virtual reality" OR "competency" OR "training" OR "curriculum" '
        'OR "resident education" OR "ATLS"))',
        min_year=2020, max_year=2026,
    ),

    # ----------------------------------------------------------
    # 15. REMOTE / TELEMEDICINE / AUSTERE ENVIRONMENT TRAUMA
    # ----------------------------------------------------------
    "telemedicine_remote": SearchQuery(
        'TITLE-ABS-KEY(trauma AND ("telemedicine" OR "teletrauma" OR "remote" OR "rural" '
        'OR "austere" OR "military" OR "tactical") '
        'AND (innovat* OR technolog* OR "new" OR advance* OR outcome*))',
        min_year=2020, max_year=2026,
    ),
}


def run_search():
    search = LiteratureSearch()

    for i, (name, query) in enumerate(QUERIES.items()):
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(QUERIES)}] {name}", flush=True)
        print(f"  Query: {query.query[:100]}...", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            search.scan(query, max_results_per_year=30, source="scopus")
            n = len(search.results) if search.results is not None else 0
            print(f"  -> Running total: {n} unique papers", flush=True)
        except Exception as e:
            print(f"  Error: {e}", flush=True)
            continue

    if search.results is None or search.results.empty:
        print("\nNo results found.")
        return

    df = search.results.reset_index()
    df["citations_count"] = df["citations_count"].astype(int)

    # --- Save all results ---
    csv_all = OUTPUT_DIR / "all_results.csv"
    search.export_csv(csv_all)
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total unique papers: {len(df)}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")

    # --- North American filter ---
    if "affiliation_country" in df.columns:
        na_mask = df["affiliation_country"].str.contains(
            "United States|Canada", case=False, na=False
        )
        df_na = df[na_mask].copy()
        df_na.to_csv(OUTPUT_DIR / "north_america.csv", index=False)
        print(f"North American papers: {len(df_na)}")

        # Canadian specifically
        ca_mask = df["affiliation_country"].str.contains("Canada", case=False, na=False)
        df_ca = df[ca_mask].copy()
        df_ca.to_csv(OUTPUT_DIR / "canadian.csv", index=False)
        print(f"Canadian papers: {len(df_ca)}")

    # --- High-impact papers (>20 citations) ---
    df_high = df[df["citations_count"] >= 20].sort_values("citations_count", ascending=False)
    df_high.to_csv(OUTPUT_DIR / "high_impact.csv", index=False)
    print(f"High-impact papers (>=20 cites): {len(df_high)}")

    # --- Recent & highly cited (2023+, >=10 cites) = HOT TOPICS ---
    df_hot = df[(df["year"] >= 2023) & (df["citations_count"] >= 10)].sort_values(
        "citations_count", ascending=False
    )
    df_hot.to_csv(OUTPUT_DIR / "hot_topics_recent.csv", index=False)
    print(f"Hot topics (2023+, >=10 cites): {len(df_hot)}")

    # --- Very recent (2025-2026) = CUTTING EDGE ---
    df_new = df[df["year"] >= 2025].sort_values("citations_count", ascending=False)
    df_new.to_csv(OUTPUT_DIR / "cutting_edge_2025.csv", index=False)
    print(f"Cutting edge (2025-2026): {len(df_new)}")

    # --- Print top papers by category ---
    print(f"\n{'='*60}")
    print("TOP PAPERS BY QUERY CATEGORY")
    print(f"{'='*60}")
    for name in QUERIES:
        cat_mask = df["query"].str.contains(
            QUERIES[name].query[:40].replace("(", "\\(").replace(")", "\\)"),
            case=False, na=False, regex=True
        )
        df_cat = df[cat_mask].nlargest(3, "citations_count")
        if df_cat.empty:
            continue
        print(f"\n--- {name} ---")
        for _, row in df_cat.iterrows():
            print(f"  [{row['citations_count']:>4} cites] {row['title'][:85]}")
            print(f"            {row.get('publication', '')} ({row['year']})")

    # --- Print absolute top 20 for journal club consideration ---
    print(f"\n{'='*60}")
    print("TOP 20 JOURNAL CLUB CANDIDATES (high cite, recent)")
    print(f"{'='*60}")
    # Weight: citations + recency bonus
    df["score"] = df["citations_count"] + (df["year"] - 2020) * 5
    top20 = df.nlargest(20, "score")
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        print(f"\n  {rank:>2}. {row['title'][:100]}")
        print(f"      {row.get('publication', '')} ({row['year']}) — {row['citations_count']} citations")
        if row.get("abstract"):
            abstract_preview = str(row["abstract"])[:150].replace("\n", " ")
            print(f"      {abstract_preview}...")

    print(f"\n\nAll CSVs saved to: {OUTPUT_DIR}/")
    print("Files:")
    print("  all_results.csv         — Full dataset")
    print("  north_america.csv       — US + Canada affiliations")
    print("  canadian.csv            — Canadian affiliations only")
    print("  high_impact.csv         — >=20 citations")
    print("  hot_topics_recent.csv   — 2023+ with >=10 citations")
    print("  cutting_edge_2025.csv   — 2025-2026 papers")


if __name__ == "__main__":
    run_search()
