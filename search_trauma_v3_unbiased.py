#!/usr/bin/env python3
"""
Trauma Acute Care Literature Search — v3
===================================================
Three-layer search strategy for a North American bibliometric analysis.

IMPORTANT: This is a single-database search (Scopus only). Results should
be interpreted with this limitation in mind. Cross-validation against
PubMed and Web of Science is recommended for formal publication.

  Layer 1: Journal-level sweep (minimally-biased foundation)
           Captures ALL content from core trauma, surgery, EM, and
           critical-care journals — no topic preselection.

  Layer 2: Broad topic queries (catch trauma papers in any journal)
           Uses broad "trauma + surgery" terms across the entire
           Scopus database so papers in NEJM, Lancet, specialty
           journals, etc. are not missed.

  Layer 3: Targeted niche supplements (cross-disciplinary topics)
           Adds emerging/niche topics that appear in non-trauma
           journals (AI in CS venues, ECMO in cardiac, etc.)

ALL queries are restricted to:
    AFFILCOUNTRY(United States) OR AFFILCOUNTRY(Canada)

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
# NORTH AMERICA FILTER (appended to every query)
# ============================================================
NA_FILTER = ' AND (AFFILCOUNTRY(United States) OR AFFILCOUNTRY(Canada))'

# ============================================================
# LAYER 1: JOURNAL-LEVEL SWEEP
# Unbiased foundation — whatever these journals publish, we capture.
# No topic preselection = the data decides what matters.
# ============================================================
QUERIES = {
    # 1a. Core trauma journals (100% trauma-relevant, no topic filter)
    "L1_trauma_journals": SearchQuery(
        '(SRCTITLE("Journal of Trauma and Acute Care Surgery") '
        'OR SRCTITLE("Injury") '
        'OR SRCTITLE("World Journal of Emergency Surgery") '
        'OR SRCTITLE("Trauma Surgery and Acute Care Open") '
        'OR SRCTITLE("European Journal of Trauma and Emergency Surgery") '
        'OR SRCTITLE("Scandinavian Journal of Trauma, Resuscitation and Emergency Medicine"))'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # 1b. General surgery journals (filtered to trauma/injury content)
    "L1_general_surgery": SearchQuery(
        '(SRCTITLE("Annals of Surgery") '
        'OR SRCTITLE("JAMA Surgery") '
        'OR SRCTITLE("British Journal of Surgery") '
        'OR SRCTITLE("Journal of the American College of Surgeons") '
        'OR SRCTITLE("American Journal of Surgery") '
        'OR SRCTITLE("Surgery") '
        'OR SRCTITLE("American Surgeon") '
        'OR SRCTITLE("Journal of Surgical Research") '
        'OR SRCTITLE("Journal of Gastrointestinal Surgery") '
        'OR SRCTITLE("Surgical Clinics of North America") '
        'OR SRCTITLE("Current Problems in Surgery") '
        'OR SRCTITLE("Canadian Journal of Surgery")) '
        'AND TITLE-ABS-KEY(trauma OR injur* OR hemorrha* '
        'OR "acute care surgery" OR "emergency surgery")'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # 1c. Emergency medicine journals
    "L1_emergency_med": SearchQuery(
        '(SRCTITLE("Annals of Emergency Medicine") '
        'OR SRCTITLE("Academic Emergency Medicine") '
        'OR SRCTITLE("Prehospital Emergency Care") '
        'OR SRCTITLE("Journal of Emergency Medicine") '
        'OR SRCTITLE("Western Journal of Emergency Medicine") '
        'OR SRCTITLE("Canadian Journal of Emergency Medicine") '
        'OR SRCTITLE("CJEM")) '
        'AND TITLE-ABS-KEY(trauma OR injur* OR hemorrha*)'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # 1d. Critical care & resuscitation journals
    "L1_critical_care": SearchQuery(
        '(SRCTITLE("Critical Care Medicine") '
        'OR SRCTITLE("Critical Care") '
        'OR SRCTITLE("Intensive Care Medicine") '
        'OR SRCTITLE("Shock") '
        'OR SRCTITLE("Resuscitation") '
        'OR SRCTITLE("Transfusion") '
        'OR SRCTITLE("Journal of Intensive Care Medicine")) '
        'AND TITLE-ABS-KEY(trauma OR injur* OR hemorrha* OR "damage control" '
        'OR "massive transfusion")'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # 1e. Neurotrauma, orthopaedic trauma & surgical education
    "L1_neurotrauma_ortho": SearchQuery(
        '(SRCTITLE("Journal of Neurotrauma") '
        'OR SRCTITLE("Neurosurgery") '
        'OR SRCTITLE("Journal of Orthopaedic Trauma") '
        'OR SRCTITLE("Journal of Surgical Education") '
        'OR SRCTITLE("Neurocritical Care")) '
        'AND TITLE-ABS-KEY(trauma OR injur*)'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # ============================================================
    # LAYER 2: BROAD TOPIC QUERIES
    # Catches trauma surgery papers in ANY journal (NEJM, Lancet,
    # specialty journals, etc.)
    # ============================================================

    # 2a. Broad trauma surgery / acute care surgery / emergency general surgery
    "L2_trauma_surgery": SearchQuery(
        'TITLE-ABS-KEY(("trauma surgery" OR "trauma surgeon" '
        'OR "acute care surgery" OR "emergency general surgery") '
        'AND (innovat* OR outcome* OR management OR guideline OR protocol '
        'OR trial OR "systematic review" OR "meta-analysis" OR mortalit*)) '
        'AND DOCTYPE(ar OR re)'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # 2b. Broadest net — TITLE-only search
    "L2_trauma_title": SearchQuery(
        'TITLE(trauma AND (surgery OR surgical OR resuscitation OR hemorrha* '
        'OR "damage control" OR "acute care" OR "emergency")) '
        'AND DOCTYPE(ar OR re)'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # ============================================================
    # LAYER 3: TARGETED NICHE SUPPLEMENTS
    # Cross-disciplinary topics that appear in non-trauma journals.
    # These ADD to the broad base, they don't define the corpus.
    # ============================================================

    # 3a. REBOA / endovascular resuscitation
    "L3_reboa": SearchQuery(
        'TITLE-ABS-KEY(REBOA OR "resuscitative endovascular balloon" '
        'OR "ER-REBOA" OR ("aortic occlusion" AND trauma))'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # 3b. AI / Machine learning in trauma
    "L3_ai_trauma": SearchQuery(
        'TITLE-ABS-KEY(("artificial intelligence" OR "machine learning" '
        'OR "deep learning" OR "natural language processing" '
        'OR "predictive model*" OR "clinical decision support") '
        'AND (trauma OR "emergency surgery" OR triage OR "injury severity"))'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # 3c. Novel blood products / resuscitation fluids
    "L3_blood_products": SearchQuery(
        'TITLE-ABS-KEY(("whole blood" OR "low titer O" OR "cold stored platelet*" '
        'OR "freeze-dried plasma" OR "lyophilized plasma" OR "fibrinogen concentrate") '
        'AND (trauma OR hemorrha* OR "massive transfusion"))'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # 3d. ECMO in trauma (emerging)
    "L3_ecmo_trauma": SearchQuery(
        'TITLE-ABS-KEY((ECMO OR ECLS OR "extracorporeal membrane" '
        'OR "extracorporeal life support") AND trauma)'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # 3e. Teletrauma / telemedicine in trauma
    "L3_teletrauma": SearchQuery(
        'TITLE-ABS-KEY((telemedicine OR teletrauma OR "tele-trauma" '
        'OR telementoring OR "remote surgery" OR "teleguided") '
        'AND (trauma OR "emergency surgery"))'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # 3f. Mass casualty / disaster trauma
    "L3_mass_casualty": SearchQuery(
        'TITLE-ABS-KEY(("mass casualty" OR "mass shooting" OR "blast injury" '
        'OR "disaster surgery" OR "active shooter" OR "bombing") '
        'AND (trauma OR triage OR surgery))'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),

    # 3g. Trauma in top-tier general medical journals
    "L3_high_impact": SearchQuery(
        '(SRCTITLE("New England Journal of Medicine") '
        'OR SRCTITLE("JAMA") '
        'OR SRCTITLE("The Lancet") '
        'OR SRCTITLE("BMJ")) '
        'AND TITLE-ABS-KEY(trauma AND (surgery OR injur* OR hemorrha* '
        'OR resuscitation OR "acute care"))'
        + NA_FILTER,
        min_year=2020, max_year=2026,
    ),
}

# Per-query limits (papers per year)
QUERY_LIMITS = {
    # Layer 1: Journal sweeps (high limits — these are the unbiased foundation)
    "L1_trauma_journals":     500,  # Raised from 200: JTACS+Injury alone > 700/yr
    "L1_general_surgery":     150,
    "L1_emergency_med":       100,
    "L1_critical_care":       100,
    "L1_neurotrauma_ortho":   100,
    # Layer 2: Broad topic (high limits — wide net)
    "L2_trauma_surgery":      200,
    "L2_trauma_title":        200,
    # Layer 3: Niche supplements (moderate limits — targeted additions)
    "L3_reboa":                50,
    "L3_ai_trauma":            50,
    "L3_blood_products":       50,
    "L3_ecmo_trauma":          30,
    "L3_teletrauma":           30,
    "L3_mass_casualty":        30,
    "L3_high_impact":          50,
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

            # Check for possible truncation (heuristic: if n_new == limit * years)
            n_years = (query.max_year or 2026) - (query.min_year or 2020) + 1
            max_possible = limit * n_years
            if n_new >= max_possible * 0.95:  # Within 5% of cap = likely truncated
                warn_msg = (
                    f"⚠ Query '{name}' returned {n_new} papers, "
                    f"near cap of {max_possible} ({limit}/yr × {n_years} yrs). "
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
        ca_mask = df["affiliation_country"].str.contains("Canada", case=False, na=False)
        print(f"US-affiliated: {us_mask.sum()}")
        print(f"Canada-affiliated: {ca_mask.sum()}")
        print(f"Both US+CA: {(us_mask & ca_mask).sum()}")

        df_ca = df[ca_mask].copy()
        df_ca.to_csv(OUTPUT_DIR / "canadian.csv", index=False)

        df_na = df[us_mask | ca_mask].copy()
        df_na.to_csv(OUTPUT_DIR / "north_america.csv", index=False)

    # --- High-impact papers (>= 20 citations) ---
    df_high = df[df["citations_count"] >= 20].sort_values("citations_count", ascending=False)
    df_high.to_csv(OUTPUT_DIR / "high_impact.csv", index=False)
    print(f"High-impact papers (>=20 cites): {len(df_high)}")

    # --- Hot topics (2023+, >= 10 cites) ---
    df_hot = df[(df["year"] >= 2023) & (df["citations_count"] >= 10)].sort_values(
        "citations_count", ascending=False
    )
    df_hot.to_csv(OUTPUT_DIR / "hot_topics_recent.csv", index=False)
    print(f"Hot topics (2023+, >=10 cites): {len(df_hot)}")

    # --- Cutting edge (2025-2026) ---
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

    # --- Top 20 journal club candidates ---
    print(f"\n{'='*60}")
    print("TOP 20 JOURNAL CLUB CANDIDATES")
    print(f"{'='*60}")
    import numpy as np
    from config import BASE_YEAR
    df["score"] = (
        np.log1p(df["citations_count"]) * 10
        + (df["year"] - BASE_YEAR) * 5
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
        "geographic_filter": NA_FILTER.strip(),
        "year_range": "2020-2026",
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
            "Per-query annual caps may truncate results — check truncation_events. "
            "Cross-validation against PubMed and Web of Science is recommended."
        ),
    }

    metadata_file = OUTPUT_DIR / "search_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\n  📋 Search metadata saved to: {metadata_file}")

    if truncation_events:
        print(f"\n  ⚠ {len(truncation_events)} queries may have been truncated:")
        for te in truncation_events:
            print(f"    {te['query']}: {te['papers_returned']}/{te['cap']} papers")

    if query_errors:
        print(f"\n  ❌ {len(query_errors)} queries failed:")
        for qe in query_errors:
            print(f"    {qe['query']}: {qe['error']}")

    print(f"\n\nAll CSVs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_search()
