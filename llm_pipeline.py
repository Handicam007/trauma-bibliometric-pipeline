#!/usr/bin/env python3
"""
LLM Pipeline Orchestrator
==========================
Runs all LLM-augmented analysis tasks in sequence with resume capability.

Usage:
    python llm_pipeline.py --provider openai --steps all
    python llm_pipeline.py --provider google --model gemini-2.0-flash --steps screen,classify
    python llm_pipeline.py --provider ollama --model llama3.2 --steps extract --limit 100
    python llm_pipeline.py --dry-run                    # Cost estimate only

Steps:
    abstract  — Fetch missing abstracts (Scopus + PubMed)
    screen    — LLM relevance screening
    classify  — Hierarchical concept classification
    extract   — Structured data extraction (Golden Prompt)
    validate  — Run validation (kappa, semantic gap, self-consistency)
    all       — Run all steps in sequence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import FIELD_NAME, LLM_PROVIDER, LLM_MODEL

# ── Paths ─────────────────────────────────────────────────────────────
RAW_INPUT = Path(__file__).parent / "results_refined" / "all_results.csv"
FILTERED_INPUT = Path(__file__).parent / "results_curated" / "all_filtered.csv"
ENRICHED_OUTPUT = Path(__file__).parent / "results_curated" / "llm_enriched.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-augmented bibliometric analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider", type=str, default=LLM_PROVIDER,
        choices=["openai", "anthropic", "google", "ollama"],
        help=f"LLM provider (default: {LLM_PROVIDER})",
    )
    parser.add_argument(
        "--model", type=str, default=LLM_MODEL,
        help=f"Model name (default: provider default)",
    )
    parser.add_argument(
        "--steps", type=str, default="all",
        help="Comma-separated steps: abstract,screen,classify,extract,validate,all",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max papers to process (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print cost estimate only, don't process",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key (overrides environment variable)",
    )
    parser.add_argument(
        "--skip-consistency", action="store_true",
        help="Skip self-consistency check (saves API calls)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse steps
    if args.steps == "all":
        steps = ["abstract", "screen", "classify", "extract", "validate"]
    else:
        steps = [s.strip() for s in args.steps.split(",")]

    print("=" * 70)
    print("LLM-AUGMENTED BIBLIOMETRIC ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"  Field:    {FIELD_NAME}")
    print(f"  Provider: {args.provider}")
    print(f"  Model:    {args.model or '(provider default)'}")
    print(f"  Steps:    {', '.join(steps)}")
    if args.limit:
        print(f"  Limit:    {args.limit} papers")
    print()

    # ── Load data ─────────────────────────────────────────────────────
    if not RAW_INPUT.exists():
        print(f"ERROR: Raw results not found: {RAW_INPUT}")
        print("Run search_trauma_v3_unbiased.py first.")
        sys.exit(1)

    df_raw = pd.read_csv(RAW_INPUT)
    df_raw["title"] = df_raw["title"].fillna("")
    df_raw["abstract"] = df_raw["abstract"].fillna("")
    df_raw["doi"] = df_raw["doi"].fillna("").astype(str)
    print(f"  Raw papers loaded: {len(df_raw):,}")

    # Load filtered data if available
    df_filtered = None
    if FILTERED_INPUT.exists():
        df_filtered = pd.read_csv(FILTERED_INPUT)
        df_filtered["title"] = df_filtered["title"].fillna("")
        df_filtered["abstract"] = df_filtered["abstract"].fillna("")
        df_filtered["doi"] = df_filtered["doi"].fillna("").astype(str)
        print(f"  Filtered papers loaded: {len(df_filtered):,}")

    # Working dataset (use filtered if available, else raw)
    df = df_filtered if df_filtered is not None else df_raw
    if args.limit:
        df_limited = df.head(args.limit)
    else:
        df_limited = df

    # ── Initialize LLM provider ──────────────────────────────────────
    from llm_providers import LLMProvider
    llm = LLMProvider(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
    )

    # ── Dry run: cost estimation ──────────────────────────────────────
    if args.dry_run:
        _print_cost_estimate(llm, len(df_limited), steps)
        return

    # ── Cost estimate before proceeding ───────────────────────────────
    if args.provider != "ollama" and not args.limit:
        _print_cost_estimate(llm, len(df_limited), steps)
        try:
            confirm = input("\nProceed? [y/N] ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Aborted.")
                return
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: ABSTRACT RETRIEVAL
    # ══════════════════════════════════════════════════════════════════
    if "abstract" in steps:
        print(f"\n{'═' * 70}")
        print("STEP 1: ABSTRACT RETRIEVAL")
        print(f"{'═' * 70}")
        from fetch_abstracts import main as fetch_main
        fetch_main(input_path=str(RAW_INPUT))

        # Reload data with new abstracts
        df_raw = pd.read_csv(RAW_INPUT)
        df_raw["title"] = df_raw["title"].fillna("")
        df_raw["abstract"] = df_raw["abstract"].fillna("")
        df_raw["doi"] = df_raw["doi"].fillna("").astype(str)

        if FILTERED_INPUT.exists():
            # Also update filtered dataset with new abstracts
            df_filtered = pd.read_csv(FILTERED_INPUT)
            abstract_map = dict(zip(df_raw["doi"].astype(str), df_raw["abstract"]))
            df_filtered["abstract"] = df_filtered["doi"].astype(str).map(abstract_map).fillna("")
            df_filtered.to_csv(FILTERED_INPUT, index=False)

        df = df_filtered if df_filtered is not None else df_raw
        if args.limit:
            df_limited = df.head(args.limit)
        else:
            df_limited = df

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: LLM RELEVANCE SCREENING
    # ══════════════════════════════════════════════════════════════════
    screening_results = None
    if "screen" in steps:
        print(f"\n{'═' * 70}")
        print("STEP 2: LLM RELEVANCE SCREENING")
        print(f"{'═' * 70}")
        from llm_screening import run_screening
        # Screen ALL raw papers (not just filtered)
        screen_input = df_raw.head(args.limit) if args.limit else df_raw
        screening_results = run_screening(screen_input, llm, limit=args.limit)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: HIERARCHICAL CONCEPT CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════
    classification_results = None
    if "classify" in steps:
        print(f"\n{'═' * 70}")
        print("STEP 3: HIERARCHICAL CONCEPT CLASSIFICATION")
        print(f"{'═' * 70}")
        from llm_classification import run_classification
        classification_results = run_classification(df_limited, llm, limit=args.limit)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: STRUCTURED DATA EXTRACTION
    # ══════════════════════════════════════════════════════════════════
    extraction_results = None
    if "extract" in steps:
        print(f"\n{'═' * 70}")
        print("STEP 4: STRUCTURED DATA EXTRACTION")
        print(f"{'═' * 70}")
        from llm_extraction import run_extraction
        extraction_results = run_extraction(df_limited, llm, limit=args.limit)

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: VALIDATION
    # ══════════════════════════════════════════════════════════════════
    if "validate" in steps:
        print(f"\n{'═' * 70}")
        print("STEP 5: VALIDATION")
        print(f"{'═' * 70}")
        _run_validation(
            df_raw=df_raw,
            df_filtered=df_filtered,
            screening_results=screening_results,
            classification_results=classification_results,
            extraction_results=extraction_results,
            llm=llm,
            skip_consistency=args.skip_consistency,
        )

    # ══════════════════════════════════════════════════════════════════
    # MERGE & SAVE ENRICHED DATASET
    # ══════════════════════════════════════════════════════════════════
    _merge_results(df, classification_results, extraction_results, screening_results)

    # ── Final usage report ────────────────────────────────────────────
    usage = llm.get_usage_summary()
    print(f"\n{'═' * 70}")
    print("LLM USAGE SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Provider:       {usage['provider']}")
    print(f"  Model:          {usage['model']}")
    print(f"  Total calls:    {usage['total_calls']:,}")
    print(f"  Input tokens:   {usage['total_input_tokens']:,}")
    print(f"  Output tokens:  {usage['total_output_tokens']:,}")
    print(f"  Estimated cost: ${usage['estimated_cost_usd']:.4f}")

    print(f"\n✅ Pipeline complete!")


def _print_cost_estimate(llm, n_papers: int, steps: list[str]):
    """Print cost estimation for all steps."""
    print(f"\n{'─' * 70}")
    print("COST ESTIMATION")
    print(f"{'─' * 70}")

    total_cost = 0.0
    step_configs = {
        "screen": ("Screening", 600, 80),
        "classify": ("Classification (2 calls)", 800, 120),
        "extract": ("Extraction", 900, 200),
    }

    for step, (label, avg_in, avg_out) in step_configs.items():
        if step in steps:
            est = llm.estimate_cost(n_papers, avg_in, avg_out)
            cost = est["estimated_cost_usd"]
            total_cost += cost
            print(f"  {label:30s}: {n_papers:,} papers × ~{avg_in}+{avg_out} tokens = ${cost:.4f}")

    if "classify" in steps:
        # Classification makes 2 calls per paper (domain + concept)
        est2 = llm.estimate_cost(n_papers, 500, 80)
        total_cost += est2["estimated_cost_usd"]

    print(f"\n  {'TOTAL ESTIMATED COST':30s}: ${total_cost:.4f}")
    if llm.provider == "ollama":
        print(f"  (Ollama is free — running locally)")


def _run_validation(
    df_raw, df_filtered, screening_results,
    classification_results, extraction_results,
    llm, skip_consistency=False,
):
    """Run all validation checks."""
    from llm_validation import (
        semantic_gap_audit,
        concept_agreement,
        screening_agreement,
        export_disputes,
        self_consistency_check,
        print_audit_table,
        save_validation_report,
    )

    report = {}

    # Load results from disk if not in memory
    screening_file = Path(__file__).parent / "results_curated" / "llm_screening_results.csv"
    concepts_file = Path(__file__).parent / "results_curated" / "llm_concepts.csv"
    extraction_file = Path(__file__).parent / "results_curated" / "llm_extracted_data.csv"

    if screening_results is None and screening_file.exists():
        screening_results = pd.read_csv(screening_file)

    if classification_results is None and concepts_file.exists():
        classification_results = pd.read_csv(concepts_file)

    if extraction_results is None and extraction_file.exists():
        extraction_results = pd.read_csv(extraction_file)

    # Screening agreement
    screening_agree = {}
    if screening_results is not None and df_filtered is not None:
        screening_agree = screening_agreement(df_filtered, df_raw, screening_results)
        report["screening_agreement"] = screening_agree

        # Export disputes
        disputes = export_disputes(df_raw, df_filtered, screening_results)
        report["n_disputes"] = len(disputes)

    # Semantic gap audit
    gap_result = {}
    if screening_results is not None:
        # Re-create exclude mask from filter_results.py logic
        from filter_results import EXCLUDE_TITLE_KEYWORDS, EXCLUDE_ABSTRACT_KEYWORDS
        df_temp = df_raw.copy()
        df_temp["title"] = df_temp["title"].fillna("")
        df_temp["abstract"] = df_temp["abstract"].fillna("")
        title_lower = df_temp["title"].str.lower()
        abstract_lower = df_temp["abstract"].str.lower()
        exclude_mask = pd.Series(False, index=df_temp.index)
        for kw in EXCLUDE_TITLE_KEYWORDS:
            exclude_mask |= title_lower.str.contains(kw, case=False, na=False)
        for kw in EXCLUDE_ABSTRACT_KEYWORDS:
            exclude_mask |= abstract_lower.str.contains(kw, case=False, na=False)

        gap_result = semantic_gap_audit(df_raw, screening_results, exclude_mask)
        report["semantic_gap"] = gap_result

    # Concept agreement
    concept_agree_df = pd.DataFrame()
    if classification_results is not None and df_filtered is not None:
        concept_agree_df = concept_agreement(df_filtered, classification_results)
        report["concept_agreement_mean_kappa"] = round(concept_agree_df["kappa"].mean(), 4)

    # Self-consistency (optional — costs API calls)
    consistency_result = None
    if not skip_consistency and df_filtered is not None:
        consistency_result = self_consistency_check(df_filtered, llm, n_sample=50)
        report["self_consistency"] = consistency_result

    # Audit summary table
    print_audit_table(
        screening_agreement_result=screening_agree,
        concept_agreement_df=concept_agree_df,
        semantic_gap_result=gap_result,
        extraction_df=extraction_results,
        consistency_result=consistency_result,
    )

    # Save report
    report["llm_usage"] = llm.get_usage_summary()
    save_validation_report(report)


def _merge_results(df, classification_results, extraction_results, screening_results):
    """Merge all LLM results into a single enriched CSV."""
    print(f"\n{'─' * 70}")
    print("MERGING ENRICHED DATASET")
    print(f"{'─' * 70}")

    enriched = df.copy()
    enriched["doi"] = enriched["doi"].astype(str)

    # Merge classification
    if classification_results is not None:
        cls_cols = ["doi", "domains", "concepts", "primary_concept",
                    "n_concepts", "domain_confidence", "concept_confidence"]
        cls_df = classification_results[[c for c in cls_cols if c in classification_results.columns]].copy()
        cls_df["doi"] = cls_df["doi"].astype(str)
        # Prefix columns to avoid collisions
        rename = {c: f"llm_{c}" for c in cls_df.columns if c != "doi"}
        cls_df = cls_df.rename(columns=rename)
        enriched = enriched.merge(cls_df, on="doi", how="left")

    # Merge extraction
    if extraction_results is not None:
        ext_cols = ["doi", "study_design", "level_of_evidence", "sample_size",
                    "population", "setting", "intervention_type", "key_finding",
                    "mortality_reported"]
        ext_df = extraction_results[[c for c in ext_cols if c in extraction_results.columns]].copy()
        ext_df["doi"] = ext_df["doi"].astype(str)
        rename = {c: f"llm_{c}" for c in ext_df.columns if c != "doi"}
        ext_df = ext_df.rename(columns=rename)
        enriched = enriched.merge(ext_df, on="doi", how="left")

    # Merge screening
    if screening_results is not None:
        scr_cols = ["doi", "relevant", "confidence", "exclusion_category"]
        scr_df = screening_results[[c for c in scr_cols if c in screening_results.columns]].copy()
        scr_df["doi"] = scr_df["doi"].astype(str)
        rename = {c: f"llm_screen_{c}" for c in scr_df.columns if c != "doi"}
        scr_df = scr_df.rename(columns=rename)
        enriched = enriched.merge(scr_df, on="doi", how="left")

    # Save
    ENRICHED_OUTPUT.parent.mkdir(exist_ok=True)
    enriched.to_csv(ENRICHED_OUTPUT, index=False)
    print(f"  Enriched dataset: {len(enriched):,} papers × {len(enriched.columns)} columns")
    print(f"  Saved to: {ENRICHED_OUTPUT}")


if __name__ == "__main__":
    main()
