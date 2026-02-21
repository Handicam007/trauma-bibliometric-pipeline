#!/usr/bin/env python3
"""
Gold-Standard Benchmark: Precision / Recall / F1
==================================================
Given a manually annotated gold-standard CSV (from generate_validation_sample),
compute precision, recall, F1, and Cohen's kappa for each LLM task:

  1. Screening (binary: relevant / not relevant)
  2. Concept classification (multi-label: per-concept P/R/F1)
  3. Structured data extraction (study_design accuracy, sample_size MAE)

Usage:
    python benchmark_accuracy.py --gold llm_cache/validation_sample.csv
    python benchmark_accuracy.py --gold my_annotations.csv --output stats_output/benchmark.json

Output: stats_output/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from concept_definitions import CLINICAL_CONCEPTS
from llm_validation import cohens_kappa, kappa_interpretation, wilson_ci

# ── Paths ─────────────────────────────────────────────────────────────
DEFAULT_GOLD = Path(__file__).parent / "llm_cache" / "validation_sample.csv"
OUTPUT_DIR = Path(__file__).parent / "stats_output"
OUTPUT_FILE = OUTPUT_DIR / "benchmark_results.json"


# ═══════════════════════════════════════════════════════════════════════
# LOAD GOLD STANDARD
# ═══════════════════════════════════════════════════════════════════════

def load_gold_standard(path: Path) -> pd.DataFrame:
    """
    Load gold-standard annotations from CSV.

    Expected columns (at minimum):
        doi, llm_relevant, human_relevant
    Optional:
        llm_concepts, human_concepts
        llm_study_design, human_study_design
        llm_sample_size, human_sample_size
        human2_relevant, adjudicated_relevant
    """
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} annotated papers from {path}")

    # Use adjudicated columns if available
    if "adjudicated_relevant" in df.columns and df["adjudicated_relevant"].notna().any():
        non_empty = df["adjudicated_relevant"].astype(str).str.strip() != ""
        if non_empty.any():
            print(f"  Using adjudicated_relevant column ({non_empty.sum()} entries)")
            df.loc[non_empty, "human_relevant"] = df.loc[non_empty, "adjudicated_relevant"]

    if "adjudicated_concepts" in df.columns and df["adjudicated_concepts"].notna().any():
        non_empty = df["adjudicated_concepts"].astype(str).str.strip() != ""
        if non_empty.any():
            print(f"  Using adjudicated_concepts column ({non_empty.sum()} entries)")
            df.loc[non_empty, "human_concepts"] = df.loc[non_empty, "adjudicated_concepts"]

    return df


# ═══════════════════════════════════════════════════════════════════════
# SCREENING METRICS (Binary Classification)
# ═══════════════════════════════════════════════════════════════════════

def compute_screening_metrics(df: pd.DataFrame) -> dict:
    """
    Compute binary classification metrics for screening.

    Returns dict with: precision, recall, F1, specificity, accuracy,
    Cohen's kappa, confusion matrix, Wilson CIs.
    """
    print(f"\n{'─' * 70}")
    print("SCREENING BENCHMARK")
    print(f"{'─' * 70}")

    # Filter to papers with human annotations
    valid = df.dropna(subset=["human_relevant"])
    valid = valid[valid["human_relevant"].astype(str).str.strip() != ""]

    if len(valid) == 0:
        print("  No human screening annotations found.")
        return {}

    # Parse
    human = valid["human_relevant"].astype(str).str.strip().str.lower().isin(
        ["yes", "true", "1", "include"]
    ).values
    llm = valid["llm_relevant"].astype(bool).values

    # Confusion matrix
    tp = int((human & llm).sum())
    tn = int((~human & ~llm).sum())
    fp = int((~human & llm).sum())
    fn = int((human & ~llm).sum())
    n = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0
    kappa = cohens_kappa(human, llm)

    # Wilson CIs
    prec_ci = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)
    rec_ci = wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (0.0, 0.0)
    f1_ci_n = tp + fp + fn  # approximate CI for F1 using total relevant predictions
    acc_ci = wilson_ci(tp + tn, n) if n > 0 else (0.0, 0.0)

    result = {
        "n": n,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": round(precision, 4),
        "precision_ci": [round(x, 4) for x in prec_ci],
        "recall": round(recall, 4),
        "recall_ci": [round(x, 4) for x in rec_ci],
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "accuracy_ci": [round(x, 4) for x in acc_ci],
        "kappa": kappa,
        "kappa_interpretation": kappa_interpretation(kappa),
    }

    print(f"  n = {n}")
    print(f"  Confusion: TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Precision: {precision:.3f} [{prec_ci[0]:.3f}-{prec_ci[1]:.3f}]")
    print(f"  Recall:    {recall:.3f} [{rec_ci[0]:.3f}-{rec_ci[1]:.3f}]")
    print(f"  F1:        {f1:.3f}")
    print(f"  Accuracy:  {accuracy:.3f} [{acc_ci[0]:.3f}-{acc_ci[1]:.3f}]")
    print(f"  Specificity: {specificity:.3f}")
    print(f"  Cohen's k: {kappa:.3f} ({kappa_interpretation(kappa)})")

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFICATION METRICS (Multi-label)
# ═══════════════════════════════════════════════════════════════════════

def compute_classification_metrics(df: pd.DataFrame) -> dict:
    """
    Compute multi-label classification metrics per concept.

    Human annotations use pipe-separated concept names in 'human_concepts'.
    """
    print(f"\n{'─' * 70}")
    print("CLASSIFICATION BENCHMARK")
    print(f"{'─' * 70}")

    if "human_concepts" not in df.columns:
        print("  No human concept annotations found.")
        return {}

    valid = df.dropna(subset=["human_concepts"])
    valid = valid[valid["human_concepts"].astype(str).str.strip() != ""]

    if len(valid) == 0:
        print("  No human concept annotations found.")
        return {}

    per_concept = []
    total_tp = total_fp = total_fn = 0

    for concept in sorted(CLINICAL_CONCEPTS.keys()):
        human_has = valid["human_concepts"].astype(str).str.contains(
            concept, na=False, regex=False
        ).values

        llm_col = "llm_concepts" if "llm_concepts" in valid.columns else "concepts"
        llm_has = valid[llm_col].astype(str).str.contains(
            concept, na=False, regex=False
        ).values

        tp = int((human_has & llm_has).sum())
        fp = int((~human_has & llm_has).sum())
        fn = int((human_has & ~llm_has).sum())
        support = int(human_has.sum())

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        kappa = cohens_kappa(human_has, llm_has)

        if support > 0:
            per_concept.append({
                "concept": concept,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "kappa": kappa,
                "support": support,
            })

    if not per_concept:
        print("  No concepts had human annotations.")
        return {}

    # Macro averages
    macro_prec = np.mean([m["precision"] for m in per_concept])
    macro_rec = np.mean([m["recall"] for m in per_concept])
    macro_f1 = np.mean([m["f1"] for m in per_concept])
    macro_kappa = np.mean([m["kappa"] for m in per_concept])

    # Micro averages
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) \
        if (micro_prec + micro_rec) > 0 else 0.0

    result = {
        "n": len(valid),
        "concepts_evaluated": len(per_concept),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_kappa": round(macro_kappa, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "micro_f1": round(micro_f1, 4),
        "per_concept": per_concept,
    }

    print(f"  n = {len(valid)}, concepts evaluated = {len(per_concept)}")
    print(f"  Macro: P={macro_prec:.3f}  R={macro_rec:.3f}  F1={macro_f1:.3f}  k={macro_kappa:.3f}")
    print(f"  Micro: P={micro_prec:.3f}  R={micro_rec:.3f}  F1={micro_f1:.3f}")

    # Top/bottom by F1
    sorted_c = sorted(per_concept, key=lambda x: x["f1"], reverse=True)
    print(f"\n  Top 5 by F1:")
    for m in sorted_c[:5]:
        print(f"    {m['concept']:>30s}: F1={m['f1']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} (n={m['support']})")
    if len(sorted_c) > 5:
        print(f"  Bottom 5 by F1:")
        for m in sorted_c[-5:]:
            print(f"    {m['concept']:>30s}: F1={m['f1']:.3f} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} (n={m['support']})")

    return result


# ═══════════════════════════════════════════════════════════════════════
# EXTRACTION METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_extraction_metrics(df: pd.DataFrame) -> dict:
    """
    Compute accuracy for structured data extraction fields.

    - study_design: exact match accuracy
    - sample_size: within-10% agreement + MAE
    """
    print(f"\n{'─' * 70}")
    print("EXTRACTION BENCHMARK")
    print(f"{'─' * 70}")

    result = {}

    # Study design accuracy
    if "human_study_design" in df.columns and "llm_study_design" in df.columns:
        valid = df.dropna(subset=["human_study_design"])
        valid = valid[valid["human_study_design"].astype(str).str.strip() != ""]

        if len(valid) > 0:
            human = valid["human_study_design"].astype(str).str.strip().str.lower()
            llm = valid["llm_study_design"].astype(str).str.strip().str.lower()
            matches = (human == llm).sum()
            accuracy = matches / len(valid)
            acc_ci = wilson_ci(int(matches), len(valid))

            result["study_design"] = {
                "n": len(valid),
                "accuracy": round(accuracy, 4),
                "accuracy_ci": [round(x, 4) for x in acc_ci],
                "correct": int(matches),
            }
            print(f"  Study design: {accuracy:.3f} accuracy "
                  f"[{acc_ci[0]:.3f}-{acc_ci[1]:.3f}] (n={len(valid)})")
        else:
            print("  No study design annotations found.")

    # Sample size agreement
    if "human_sample_size" in df.columns and "llm_sample_size" in df.columns:
        valid = df.copy()
        valid["h_ss"] = pd.to_numeric(valid["human_sample_size"], errors="coerce")
        valid["l_ss"] = pd.to_numeric(valid["llm_sample_size"], errors="coerce")
        valid = valid.dropna(subset=["h_ss", "l_ss"])
        valid = valid[valid["h_ss"] > 0]

        if len(valid) > 0:
            h = valid["h_ss"].values
            l = valid["l_ss"].values
            within_10 = np.abs(h - l) / h <= 0.10
            mae = float(np.mean(np.abs(h - l)))
            median_ae = float(np.median(np.abs(h - l)))

            result["sample_size"] = {
                "n": len(valid),
                "within_10pct": round(within_10.sum() / len(within_10), 4),
                "mae": round(mae, 2),
                "median_ae": round(median_ae, 2),
            }
            print(f"  Sample size: {within_10.sum()}/{len(valid)} within 10% "
                  f"({within_10.sum()/len(valid)*100:.1f}%), MAE={mae:.1f}")
        else:
            print("  No sample size annotations found.")

    if not result:
        print("  No extraction annotations found.")

    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark(gold_path: Path, output_path: Path = None) -> dict:
    """
    Run full benchmark suite against gold-standard annotations.

    Args:
        gold_path: Path to annotated CSV
        output_path: Where to save results JSON

    Returns:
        dict with all benchmark metrics
    """
    if output_path is None:
        output_path = OUTPUT_FILE

    print(f"\n{'=' * 70}")
    print("GOLD-STANDARD BENCHMARK")
    print(f"{'=' * 70}")

    df = load_gold_standard(gold_path)

    results = {
        "gold_standard_path": str(gold_path),
        "n_total": len(df),
    }

    # Run each benchmark
    screening = compute_screening_metrics(df)
    if screening:
        results["screening"] = screening

    classification = compute_classification_metrics(df)
    if classification:
        results["classification"] = classification

    extraction = compute_extraction_metrics(df)
    if extraction:
        results["extraction"] = extraction

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  Benchmark results saved to: {output_path}")
    print(f"{'=' * 70}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run gold-standard benchmark against human annotations",
    )
    parser.add_argument(
        "--gold", type=Path, default=DEFAULT_GOLD,
        help=f"Path to annotated CSV (default: {DEFAULT_GOLD})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_FILE,
        help=f"Output JSON path (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    if not args.gold.exists():
        print(f"ERROR: Gold standard file not found: {args.gold}")
        print(f"  Generate it with: python llm_pipeline.py --generate-validation-sample")
        print(f"  Then annotate the CSV before running this script.")
        sys.exit(1)

    run_benchmark(args.gold, args.output)


if __name__ == "__main__":
    main()
