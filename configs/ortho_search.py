#!/usr/bin/env python3
"""
Orthopedic Surgery Literature Search
===================================================
Three-layer search strategy for an orthopedic surgery bibliometric analysis.

Copy this to the project root (replacing search_trauma_v3_unbiased.py) when
switching to the Orthopedic Surgery field.

IMPORTANT: This is a single-database search (Scopus only). Cross-validation
against PubMed and Web of Science is recommended for formal publication.

  Layer 1: Journal-level sweep (minimally-biased foundation)
           Captures ALL content from core orthopedic journals — no topic preselection.

  Layer 2: Broad topic queries (catch ortho papers in any journal)
           Uses broad "orthop* + fracture + arthroplasty" terms across the entire
           Scopus database so papers in NEJM, Lancet, general surgery journals, etc.
           are not missed.

  Layer 3: Targeted niche supplements (cross-disciplinary topics)
           Adds emerging/niche topics that appear in non-ortho journals
           (AI in CS venues, biologics in regenerative medicine, etc.)

Deduplication by DOI is handled automatically by bibtool.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "bibtool" / "src"))

from bibtool.search_papers import LiteratureSearch, SearchQuery

OUTPUT_DIR = Path(__file__).parent / "results_refined"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# GEOGRAPHIC FILTER
# ============================================================
# Set to '' for global search (no filter).
# Uncomment one of these if you want geographic restriction:
#   GEO_FILTER = ' AND (AFFILCOUNTRY(United States) OR AFFILCOUNTRY(Canada))'
#   GEO_FILTER = ' AND AFFILCOUNTRY(United Kingdom)'
GEO_FILTER = ''  # Global — ortho is an international field

# ============================================================
# YEAR RANGE
# ============================================================
YEAR_MIN = 2018
YEAR_MAX = 2026

# ============================================================
# LAYER 1: JOURNAL-LEVEL SWEEP
# Unbiased foundation — whatever these journals publish, we capture.
# ============================================================
QUERIES = {
    # 1a. Core orthopaedic journals (100% ortho-relevant, no topic filter)
    "L1_core_ortho_journals": SearchQuery(
        '(SRCTITLE("Journal of Bone and Joint Surgery") '
        'OR SRCTITLE("The Bone and Joint Journal") '
        'OR SRCTITLE("Clinical Orthopaedics and Related Research") '
        'OR SRCTITLE("Journal of Orthopaedic Trauma") '
        'OR SRCTITLE("Journal of Orthopaedic Research") '
        'OR SRCTITLE("Injury"))'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # 1b. Arthroplasty & joint-focused journals
    "L1_arthroplasty_joints": SearchQuery(
        '(SRCTITLE("Journal of Arthroplasty") '
        'OR SRCTITLE("Arthroplasty Today") '
        'OR SRCTITLE("Knee Surgery Sports Traumatology Arthroscopy") '
        'OR SRCTITLE("Arthroscopy") '
        'OR SRCTITLE("Journal of Shoulder and Elbow Surgery") '
        'OR SRCTITLE("American Journal of Sports Medicine") '
        'OR SRCTITLE("Sports Health"))'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # 1c. Spine journals
    "L1_spine": SearchQuery(
        '(SRCTITLE("Spine") '
        'OR SRCTITLE("Spine Journal") '
        'OR SRCTITLE("European Spine Journal") '
        'OR SRCTITLE("Journal of Neurosurgery: Spine") '
        'OR SRCTITLE("Global Spine Journal") '
        'OR SRCTITLE("North American Spine Society"))'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # 1d. Hand, foot, and subspecialty journals
    "L1_subspecialty": SearchQuery(
        '(SRCTITLE("Journal of Hand Surgery") '
        'OR SRCTITLE("Hand") '
        'OR SRCTITLE("Foot and Ankle International") '
        'OR SRCTITLE("Foot and Ankle Surgery") '
        'OR SRCTITLE("Journal of Pediatric Orthopaedics") '
        'OR SRCTITLE("Musculoskeletal Surgery") '
        'OR SRCTITLE("Acta Orthopaedica") '
        'OR SRCTITLE("International Orthopaedics"))'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # 1e. General surgery / high-impact with ortho content
    "L1_general_surgery_ortho": SearchQuery(
        '(SRCTITLE("Annals of Surgery") '
        'OR SRCTITLE("JAMA Surgery") '
        'OR SRCTITLE("British Journal of Surgery") '
        'OR SRCTITLE("Journal of the American College of Surgeons")) '
        'AND TITLE-ABS-KEY(fracture OR arthroplasty OR orthop* OR musculoskeletal)'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # ============================================================
    # LAYER 2: BROAD TOPIC QUERIES
    # Catches ortho papers in ANY journal (NEJM, Lancet, etc.)
    # ============================================================

    # 2a. Broad orthopedic surgery terms
    "L2_ortho_surgery": SearchQuery(
        'TITLE-ABS-KEY(("orthopedic surgery" OR "orthopaedic surgery" '
        'OR "orthopedic surgeon" OR "orthopaedic surgeon" '
        'OR arthroplasty OR "joint replacement") '
        'AND (outcome* OR management OR guideline OR protocol '
        'OR trial OR "systematic review" OR "meta-analysis" OR complication*)) '
        'AND DOCTYPE(ar OR re)'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # 2b. Fracture management — broadest net
    "L2_fracture_mgmt": SearchQuery(
        'TITLE(fracture AND (surgery OR surgical OR fixation OR arthroplasty '
        'OR "open reduction" OR "intramedullary" OR "plate" OR outcome*)) '
        'AND DOCTYPE(ar OR re)'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # 2c. Sports medicine / ligament / arthroscopy
    "L2_sports_medicine": SearchQuery(
        'TITLE-ABS-KEY(("ACL reconstruction" OR "rotator cuff" OR "meniscus repair" '
        'OR "shoulder arthroplasty" OR "knee ligament" OR "labral repair") '
        'AND (outcome* OR technique OR "return to sport" OR complication*)) '
        'AND DOCTYPE(ar OR re)'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # ============================================================
    # LAYER 3: TARGETED NICHE SUPPLEMENTS
    # Cross-disciplinary topics in non-ortho journals.
    # ============================================================

    # 3a. Robotics in orthopaedic surgery
    "L3_robotics_ortho": SearchQuery(
        'TITLE-ABS-KEY(("robotic" OR "robot-assisted" OR "computer-assisted" '
        'OR "navigation") '
        'AND (arthroplasty OR "knee replacement" OR "hip replacement" '
        'OR "spine surgery" OR orthop*))'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # 3b. AI / Machine learning in orthopaedics
    "L3_ai_ortho": SearchQuery(
        'TITLE-ABS-KEY(("artificial intelligence" OR "machine learning" '
        'OR "deep learning" OR "neural network" OR "predictive model*") '
        'AND (fracture OR arthroplasty OR orthop* OR musculoskeletal '
        'OR "joint replacement" OR spine))'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # 3c. 3D printing / patient-specific instruments
    "L3_3d_printing": SearchQuery(
        'TITLE-ABS-KEY(("3D printing" OR "three-dimensional printing" '
        'OR "additive manufacturing" OR "patient-specific" OR "custom implant") '
        'AND (orthop* OR fracture OR arthroplasty OR bone))'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # 3d. Biologics / PRP / stem cells in ortho
    "L3_biologics": SearchQuery(
        'TITLE-ABS-KEY(("platelet-rich plasma" OR PRP OR "stem cell" '
        'OR "bone morphogenetic" OR "growth factor" OR "tissue engineering") '
        'AND (orthop* OR fracture OR cartilage OR tendon OR bone))'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),

    # 3e. Ortho in top-tier general medical journals
    "L3_high_impact": SearchQuery(
        '(SRCTITLE("New England Journal of Medicine") '
        'OR SRCTITLE("JAMA") '
        'OR SRCTITLE("The Lancet") '
        'OR SRCTITLE("BMJ")) '
        'AND TITLE-ABS-KEY(fracture OR arthroplasty OR orthop* '
        'OR "joint replacement" OR "hip fracture")'
        + GEO_FILTER,
        min_year=YEAR_MIN, max_year=YEAR_MAX,
    ),
}

# Per-query limits (papers per year)
QUERY_LIMITS = {
    # Layer 1: Journal sweeps (high limits — unbiased foundation)
    "L1_core_ortho_journals":   500,
    "L1_arthroplasty_joints":   400,
    "L1_spine":                 300,
    "L1_subspecialty":          200,
    "L1_general_surgery_ortho": 100,
    # Layer 2: Broad topic (high limits — wide net)
    "L2_ortho_surgery":         300,
    "L2_fracture_mgmt":         300,
    "L2_sports_medicine":       200,
    # Layer 3: Niche supplements (moderate limits)
    "L3_robotics_ortho":        100,
    "L3_ai_ortho":              100,
    "L3_3d_printing":            50,
    "L3_biologics":              50,
    "L3_high_impact":            50,
}


def run_search():
    search = LiteratureSearch()

    # ── Track search metadata for reproducibility ─────────────────────
    search_start = datetime.now(timezone.utc)
    query_metadata = []
    truncation_events = []
    query_errors = []

    for i, (name, query) in enumerate(QUERIES.items()):
        limit = QUERY_LIMITS[name]
        layer = name.split("_")[0]  # L1, L2, L3
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(QUERIES)}] {name}  (Layer {layer}, max {limit}/yr)", flush=True)
        print(f"  Query: {query.query[:120]}...", flush=True)
        print(f"{'='*60}", flush=True)

        n_before = len(search.results) if search.results is not None else 0
        try:
            search.scan(query, max_results_per_year=limit, source="scopus")
            n_after = len(search.results) if search.results is not None else 0
            n_new = n_after - n_before
            print(f"  -> Running total (deduplicated): {n_after} unique papers", flush=True)

            n_years = (query.max_year or YEAR_MAX) - (query.min_year or YEAR_MIN) + 1
            max_possible = limit * n_years
            if n_new >= max_possible * 0.95:
                warn_msg = (
                    f"Warning: Query '{name}' returned {n_new} papers, "
                    f"near cap of {max_possible} ({limit}/yr x {n_years} yrs). "
                    f"Results may be TRUNCATED."
                )
                print(f"  {warn_msg}", flush=True)
                truncation_events.append({
                    "query": name,
                    "papers_returned": n_new,
                    "cap": max_possible,
                    "limit_per_year": limit,
                })

            query_metadata.append({
                "query_name": name,
                "layer": layer,
                "limit_per_year": limit,
                "papers_added": n_new,
                "status": "success",
            })

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            query_errors.append({"query": name, "error": str(e)})
            query_metadata.append({
                "query_name": name,
                "layer": layer,
                "limit_per_year": limit,
                "papers_added": 0,
                "status": f"error: {e}",
            })
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

    # --- Geographic breakdown ---
    if "affiliation_country" in df.columns:
        us_mask = df["affiliation_country"].str.contains("United States", case=False, na=False)
        uk_mask = df["affiliation_country"].str.contains("United Kingdom", case=False, na=False)
        print(f"US-affiliated: {us_mask.sum()}")
        print(f"UK-affiliated: {uk_mask.sum()}")

    # --- High-impact papers (>= 20 citations) ---
    df_high = df[df["citations_count"] >= 20].sort_values("citations_count", ascending=False)
    df_high.to_csv(OUTPUT_DIR / "high_impact.csv", index=False)
    print(f"High-impact papers (>=20 cites): {len(df_high)}")

    # --- Hot topics (recent + cited) ---
    df_hot = df[(df["year"] >= 2023) & (df["citations_count"] >= 10)].sort_values(
        "citations_count", ascending=False
    )
    df_hot.to_csv(OUTPUT_DIR / "hot_topics_recent.csv", index=False)
    print(f"Hot topics (2023+, >=10 cites): {len(df_hot)}")

    # --- Cutting edge ---
    df_new = df[df["year"] >= 2025].sort_values("citations_count", ascending=False)
    df_new.to_csv(OUTPUT_DIR / "cutting_edge_2025.csv", index=False)
    print(f"Cutting edge (2025-2026): {len(df_new)}")

    # --- Year distribution ---
    print(f"\n{'='*60}")
    print("PAPERS PER YEAR")
    print(f"{'='*60}")
    for y, c in df["year"].value_counts().sort_index().items():
        print(f"  {y}: {c}")

    # --- Layer breakdown ---
    print(f"\n{'='*60}")
    print("PAPERS BY SEARCH LAYER")
    print(f"{'='*60}")
    for name in QUERIES:
        q_str = QUERIES[name].query[:50].replace("(", "\\(").replace(")", "\\)")
        try:
            cat_mask = df["query"].str.contains(q_str, case=False, na=False, regex=True)
            n_cat = cat_mask.sum()
        except Exception:
            n_cat = "?"
        print(f"  {name}: {n_cat} papers")

    # --- Top 20 candidates ---
    print(f"\n{'='*60}")
    print("TOP 20 MOST IMPACTFUL PAPERS")
    print(f"{'='*60}")
    import numpy as np
    df["score"] = (
        np.log1p(df["citations_count"]) * 10
        + (df["year"] - (YEAR_MIN - 1)) * 5
    )
    top20 = df.nlargest(20, "score")
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        print(f"\n  {rank:>2}. {row['title'][:100]}")
        print(f"      {row.get('publication', '')} ({row['year']}) — {row['citations_count']} citations")

    # ── Save search metadata ─────────────────────────────────────────
    search_end = datetime.now(timezone.utc)
    metadata = {
        "search_date_utc": search_start.isoformat(),
        "search_end_utc": search_end.isoformat(),
        "duration_seconds": round((search_end - search_start).total_seconds(), 1),
        "database": "Scopus",
        "api_interface": "Elsevier Scopus Search API (via bibtool)",
        "geographic_filter": GEO_FILTER.strip() if GEO_FILTER else "None (global)",
        "year_range": f"{YEAR_MIN}-{YEAR_MAX}",
        "total_unique_papers": len(df),
        "total_queries": len(QUERIES),
        "queries_successful": sum(1 for q in query_metadata if q["status"] == "success"),
        "queries_failed": len(query_errors),
        "truncation_warnings": len(truncation_events),
        "query_details": query_metadata,
        "truncation_events": truncation_events,
        "query_errors": query_errors,
        "note": (
            "This is a single-database search (Scopus only). "
            "Per-query annual caps may truncate results. "
            "Cross-validation against PubMed and Web of Science is recommended."
        ),
    }

    metadata_file = OUTPUT_DIR / "search_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\n  Search metadata saved to: {metadata_file}")

    if truncation_events:
        print(f"\n  {len(truncation_events)} queries may have been truncated:")
        for te in truncation_events:
            print(f"    {te['query']}: {te['papers_returned']}/{te['cap']} papers")

    if query_errors:
        print(f"\n  {len(query_errors)} queries failed:")
        for qe in query_errors:
            print(f"    {qe['query']}: {qe['error']}")

    print(f"\n\nAll CSVs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_search()
