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
import json
import sys
import time
from pathlib import Path

import pandas as pd

from config import (
    FIELD_NAME, LLM_PROVIDER, LLM_MODEL,
    CONSENSUS_TEMPERATURES,
)
from llm_utils import setup_logging


# ═══════════════════════════════════════════════════════════════════════
# STEP COST TRACKER
# ═══════════════════════════════════════════════════════════════════════

class StepCostTracker:
    """Track per-step token usage, cost, and wall time for publication reporting."""

    def __init__(self, llm):
        self.llm = llm
        self.steps = []
        self._snapshot = None

    def start_step(self, step_name: str):
        """Snapshot current counters before a step begins."""
        self._snapshot = {
            "step": step_name,
            "start_tokens_in": self.llm.total_input_tokens,
            "start_tokens_out": self.llm.total_output_tokens,
            "start_calls": self.llm.total_calls,
            "start_time": time.time(),
        }

    def end_step(self):
        """Record deltas since start_step()."""
        if self._snapshot is None:
            return
        s = self._snapshot
        s["tokens_in"] = self.llm.total_input_tokens - s["start_tokens_in"]
        s["tokens_out"] = self.llm.total_output_tokens - s["start_tokens_out"]
        s["calls"] = self.llm.total_calls - s["start_calls"]
        s["wall_time_s"] = round(time.time() - s["start_time"], 2)

        # Estimate cost for this step
        usage = self.llm.get_usage_summary()
        total_tokens = self.llm.total_input_tokens + self.llm.total_output_tokens
        if total_tokens > 0 and usage.get("estimated_cost_usd", 0) > 0:
            step_tokens = s["tokens_in"] + s["tokens_out"]
            s["cost_usd"] = round(usage["estimated_cost_usd"] * step_tokens / total_tokens, 6)
        else:
            s["cost_usd"] = 0.0

        # Clean up internal keys before storing
        clean = {k: v for k, v in s.items() if not k.startswith("start_")}
        self.steps.append(clean)
        self._snapshot = None

    def get_report(self) -> dict:
        """Return structured cost report for JSON serialization."""
        total = {
            "total_calls": sum(s["calls"] for s in self.steps),
            "total_tokens_in": sum(s["tokens_in"] for s in self.steps),
            "total_tokens_out": sum(s["tokens_out"] for s in self.steps),
            "total_wall_time_s": round(sum(s["wall_time_s"] for s in self.steps), 2),
            "total_cost_usd": round(sum(s["cost_usd"] for s in self.steps), 6),
        }
        return {"steps": self.steps, "totals": total}

    def print_report(self):
        """Pretty-print per-step cost breakdown."""
        if not self.steps:
            return

        print(f"\n{'─' * 70}")
        print("COST BREAKDOWN (PER STEP)")
        print(f"{'─' * 70}")
        print(f"  {'Step':<20s} {'Calls':>7s} {'Tokens In':>12s} {'Tokens Out':>12s} "
              f"{'Time':>8s} {'Cost':>10s}")
        print(f"  {'─' * 68}")

        for s in self.steps:
            time_str = f"{s['wall_time_s']:.1f}s"
            if s["wall_time_s"] > 60:
                time_str = f"{s['wall_time_s']/60:.1f}m"
            print(f"  {s['step']:<20s} {s['calls']:>7,d} {s['tokens_in']:>12,d} "
                  f"{s['tokens_out']:>12,d} {time_str:>8s} ${s['cost_usd']:>9.4f}")

        report = self.get_report()
        t = report["totals"]
        total_time = f"{t['total_wall_time_s']:.1f}s"
        if t["total_wall_time_s"] > 60:
            total_time = f"{t['total_wall_time_s']/60:.1f}m"
        print(f"  {'─' * 68}")
        print(f"  {'TOTAL':<20s} {t['total_calls']:>7,d} {t['total_tokens_in']:>12,d} "
              f"{t['total_tokens_out']:>12,d} {total_time:>8s} ${t['total_cost_usd']:>9.4f}")

    def save_report(self, path: Path = None):
        """Save cost report to JSON."""
        if path is None:
            path = Path(__file__).parent / "llm_cache" / "cost_report.json"
        path.parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_report(), f, indent=2)
        print(f"  Cost report saved to: {path}")

# ── Paths ─────────────────────────────────────────────────────────────
RAW_INPUT = Path(__file__).parent / "results_refined" / "all_results.csv"
FILTERED_INPUT = Path(__file__).parent / "results_curated" / "all_filtered.csv"
ENRICHED_OUTPUT = Path(__file__).parent / "results_curated" / "llm_enriched.csv"

# ── Valid step names ──────────────────────────────────────────────────
VALID_STEPS = {"abstract", "screen", "classify", "extract", "validate", "all"}


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
    parser.add_argument(
        "--generate-validation-sample", action="store_true",
        help="Generate stratified validation sample for human annotation, then exit",
    )
    parser.add_argument(
        "--compute-human-agreement", action="store_true",
        help="Compute LLM vs human agreement from annotated validation sample, then exit",
    )
    return parser.parse_args()


def _validate_steps(steps_str: str) -> list[str]:
    """Validate and parse step names, catching typos early."""
    if steps_str == "all":
        return ["abstract", "screen", "classify", "extract", "validate"]

    steps = [s.strip() for s in steps_str.split(",")]
    invalid = [s for s in steps if s not in VALID_STEPS]

    if invalid:
        print(f"ERROR: Unknown step(s): {invalid}")
        print(f"Valid steps: {sorted(VALID_STEPS)}")
        sys.exit(1)

    return steps


def main():
    # Set up logging
    setup_logging()

    args = parse_args()

    # Validate steps (catches typos like "scren" instead of "screen")
    steps = _validate_steps(args.steps)

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
        print("Run the search script first (e.g., search_trauma_v3_unbiased.py).")
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

    # ── Handle standalone publication commands ────────────────────────
    if args.generate_validation_sample:
        _handle_generate_validation_sample(df_raw, df_filtered)
        return

    if args.compute_human_agreement:
        _handle_compute_human_agreement()
        return

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

    # ── Initialize cost tracker ────────────────────────────────────────
    tracker = StepCostTracker(llm)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: ABSTRACT RETRIEVAL
    # ══════════════════════════════════════════════════════════════════
    if "abstract" in steps:
        print(f"\n{'=' * 70}")
        print("STEP 1: ABSTRACT RETRIEVAL")
        print(f"{'=' * 70}")
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
        print(f"\n{'=' * 70}")
        print("STEP 2: LLM RELEVANCE SCREENING")
        print(f"{'=' * 70}")
        tracker.start_step("Screening")
        from llm_screening import run_screening
        # Screen ALL raw papers (not just filtered)
        screen_input = df_raw.head(args.limit) if args.limit else df_raw
        screening_results = run_screening(screen_input, llm, limit=args.limit)

        # Apply human overrides from disputes.csv if they exist
        from llm_validation import import_disputes
        screening_results = import_disputes(screening_results)
        tracker.end_step()

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: HIERARCHICAL CONCEPT CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════
    classification_results = None
    if "classify" in steps:
        print(f"\n{'=' * 70}")
        print("STEP 3: HIERARCHICAL CONCEPT CLASSIFICATION")
        print(f"{'=' * 70}")
        tracker.start_step("Classification")
        from llm_classification import run_classification
        classification_results = run_classification(df_limited, llm, limit=args.limit)
        tracker.end_step()

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: STRUCTURED DATA EXTRACTION
    # ══════════════════════════════════════════════════════════════════
    extraction_results = None
    if "extract" in steps:
        print(f"\n{'=' * 70}")
        print("STEP 4: STRUCTURED DATA EXTRACTION")
        print(f"{'=' * 70}")
        tracker.start_step("Extraction")
        from llm_extraction import run_extraction
        extraction_results = run_extraction(df_limited, llm, limit=args.limit)
        tracker.end_step()

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: VALIDATION
    # ══════════════════════════════════════════════════════════════════
    if "validate" in steps:
        print(f"\n{'=' * 70}")
        print("STEP 5: VALIDATION")
        print(f"{'=' * 70}")
        tracker.start_step("Validation")
        _run_validation(
            df_raw=df_raw,
            df_filtered=df_filtered,
            screening_results=screening_results,
            classification_results=classification_results,
            extraction_results=extraction_results,
            llm=llm,
            skip_consistency=args.skip_consistency,
        )
        tracker.end_step()

    # ══════════════════════════════════════════════════════════════════
    # MERGE & SAVE ENRICHED DATASET
    # ══════════════════════════════════════════════════════════════════
    _merge_results(df, classification_results, extraction_results, screening_results)

    # ── Final usage report ────────────────────────────────────────────
    usage = llm.get_usage_summary()
    print(f"\n{'=' * 70}")
    print("LLM USAGE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Provider:       {usage['provider']}")
    print(f"  Model:          {usage['model']}")
    print(f"  Total calls:    {usage['total_calls']:,}")
    print(f"  Input tokens:   {usage['total_input_tokens']:,}")
    print(f"  Output tokens:  {usage['total_output_tokens']:,}")
    print(f"  Estimated cost: ${usage['estimated_cost_usd']:.4f}")

    # Per-step cost breakdown
    tracker.print_report()
    tracker.save_report()

    print(f"\nPipeline complete!")


def _print_cost_estimate(llm, n_papers: int, steps: list[str]):
    """Print cost estimation for all steps."""
    print(f"\n{'─' * 70}")
    print("COST ESTIMATION")
    print(f"{'─' * 70}")

    total_cost = 0.0

    # Stage 1 (domain) + Stage 2 (concept) estimated separately
    step_configs = {
        "screen": [("Screening", 600, 80)],
        "classify": [
            ("Classification Stage 1 (domain)", 400, 60),
            ("Classification Stage 2 (concept)", 600, 100),
        ],
        "extract": [("Extraction", 900, 200)],
    }

    for step, configs in step_configs.items():
        if step in steps:
            for label, avg_in, avg_out in configs:
                est = llm.estimate_cost(n_papers, avg_in, avg_out)
                cost = est["estimated_cost_usd"]
                total_cost += cost
                print(f"  {label:40s}: {n_papers:,} papers x ~{avg_in}+{avg_out} tok = ${cost:.4f}")

    print(f"\n  {'TOTAL ESTIMATED COST':40s}: ${total_cost:.4f}")
    if llm.provider == "ollama":
        print(f"  (Ollama is free — running locally)")


def _handle_generate_validation_sample(df_raw, df_filtered):
    """Handle --generate-validation-sample standalone command."""
    from llm_validation import generate_validation_sample

    screening_file = Path(__file__).parent / "results_curated" / "llm_screening_results.csv"
    concepts_file = Path(__file__).parent / "results_curated" / "llm_concepts.csv"
    extraction_file = Path(__file__).parent / "results_curated" / "llm_extracted_data.csv"

    if not screening_file.exists():
        print("ERROR: Screening results not found. Run --steps screen first.")
        sys.exit(1)

    screening = pd.read_csv(screening_file)
    classification = pd.read_csv(concepts_file) if concepts_file.exists() else None
    extraction = pd.read_csv(extraction_file) if extraction_file.exists() else None

    df = df_filtered if df_filtered is not None else df_raw
    generate_validation_sample(
        df=df,
        screening_results=screening,
        classification_results=classification,
        extraction_results=extraction,
    )


def _handle_compute_human_agreement():
    """Handle --compute-human-agreement standalone command."""
    from llm_validation import compute_human_llm_agreement
    compute_human_llm_agreement()


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
        consensus_check,
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

    # Consensus check (3-run majority vote — replaces 2-run self-consistency)
    consistency_result = None
    if not skip_consistency and df_filtered is not None:
        consistency_result = consensus_check(
            df_filtered, llm,
            n_sample=50,
            temperatures=CONSENSUS_TEMPERATURES,
        )
        report["consensus_check"] = consistency_result

        # Also keep legacy key for backward compatibility
        report["self_consistency"] = {
            "n_tested": consistency_result.get("n_tested", 0),
            "consistency_pct": consistency_result.get("unanimous_pct", 0),
            "pass": consistency_result.get("pass", False),
        }

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
    """Merge all LLM results into a single enriched CSV (idempotent)."""
    print(f"\n{'─' * 70}")
    print("MERGING ENRICHED DATASET")
    print(f"{'─' * 70}")

    enriched = df.copy()
    # Replace empty/NaN DOIs with unique placeholders to prevent false dedup matches
    enriched["doi"] = enriched["doi"].fillna("").astype(str)
    missing_doi = enriched["doi"].isin(["", "nan", "None"])
    if missing_doi.any():
        enriched.loc[missing_doi, "doi"] = [
            f"__NO_DOI_{i}" for i in range(missing_doi.sum())
        ]

    # Remove any previously merged LLM columns to prevent _x/_y duplicates
    llm_cols = [c for c in enriched.columns if c.startswith("llm_")]
    if llm_cols:
        enriched = enriched.drop(columns=llm_cols)

    # Merge classification
    if classification_results is not None:
        cls_cols = ["doi", "domains", "concepts", "primary_concept",
                    "n_concepts", "domain_confidence", "concept_confidence"]
        cls_df = classification_results[[c for c in cls_cols if c in classification_results.columns]].copy()
        cls_df["doi"] = cls_df["doi"].astype(str)
        # Prefix columns to avoid collisions
        rename = {c: f"llm_{c}" for c in cls_df.columns if c != "doi"}
        cls_df = cls_df.rename(columns=rename)
        cls_df = cls_df.drop_duplicates(subset=["doi"], keep="last")  # Guard against DOI dupes
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
        ext_df = ext_df.drop_duplicates(subset=["doi"], keep="last")  # Guard against DOI dupes
        enriched = enriched.merge(ext_df, on="doi", how="left")

    # Merge screening
    if screening_results is not None:
        scr_cols = ["doi", "relevant", "confidence", "exclusion_category"]
        scr_df = screening_results[[c for c in scr_cols if c in screening_results.columns]].copy()
        scr_df["doi"] = scr_df["doi"].astype(str)
        rename = {c: f"llm_screen_{c}" for c in scr_df.columns if c != "doi"}
        scr_df = scr_df.rename(columns=rename)
        scr_df = scr_df.drop_duplicates(subset=["doi"], keep="last")  # Guard against DOI dupes
        enriched = enriched.merge(scr_df, on="doi", how="left")

    # Save
    ENRICHED_OUTPUT.parent.mkdir(exist_ok=True)
    enriched.to_csv(ENRICHED_OUTPUT, index=False)
    print(f"  Enriched dataset: {len(enriched):,} papers x {len(enriched.columns)} columns")
    print(f"  Saved to: {ENRICHED_OUTPUT}")


if __name__ == "__main__":
    main()
