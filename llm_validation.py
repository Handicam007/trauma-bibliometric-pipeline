"""
LLM Validation Module
======================
Five validation layers for publication-defensible LLM-augmented bibliometrics:

  1. Semantic Gap Audit — quantifies search recall improvement (with Wilson CI)
  2. LLM vs Regex Agreement — Cohen's kappa per concept
  3. Screening Agreement — LLM vs keyword filter with kappa
  4. Human-in-the-Loop Dispute Export + Override Import
  5. Self-Consistency Check — same LLM on same papers twice at temp=0

Note on kappa interpretation: Landis & Koch (1977) thresholds were developed
for human-vs-human agreement. Their applicability to algorithm-vs-algorithm
comparisons is debated. We report them as conventional benchmarks with this caveat.

Output: llm_cache/validation_report.json, llm_cache/disputes.csv
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from concept_definitions import CLINICAL_CONCEPTS
from config import (
    FIELD_NAME, LLM_CONFIDENCE_AUTO, LLM_CONFIDENCE_REVIEW,
)
from llm_utils import load_cache, save_cache

logger = logging.getLogger("llm_pipeline.validation")

CACHE_DIR = Path(__file__).parent / "llm_cache"
VALIDATION_REPORT = CACHE_DIR / "validation_report.json"
DISPUTES_FILE = CACHE_DIR / "disputes.csv"


# ═══════════════════════════════════════════════════════════════════════
# COHEN'S KAPPA
# ═══════════════════════════════════════════════════════════════════════

def cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Compute Cohen's kappa for two binary arrays.

    Note: Interpretation thresholds (Landis & Koch, 1977) were developed
    for human-vs-human reliability. Use with caution for algorithm comparisons.

    Returns:
        kappa in [-1, 1]. Values:
          >=0.81: almost perfect
          0.61-0.80: substantial
          0.41-0.60: moderate
          0.21-0.40: fair
          <=0.20: poor
    """
    y1, y2 = np.asarray(y1, dtype=bool), np.asarray(y2, dtype=bool)
    n = len(y1)
    if n == 0:
        return 0.0

    # Agreement
    agree = np.sum(y1 == y2)
    p_o = agree / n

    # Expected agreement
    p_yes_1 = np.sum(y1) / n
    p_yes_2 = np.sum(y2) / n
    p_no_1 = 1 - p_yes_1
    p_no_2 = 1 - p_yes_2
    p_e = p_yes_1 * p_yes_2 + p_no_1 * p_no_2

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1 - p_e)
    return round(kappa, 4)


def kappa_interpretation(k: float) -> str:
    """Human-readable interpretation of Cohen's kappa (Landis & Koch, 1977)."""
    if k >= 0.81:
        return "almost perfect"
    elif k >= 0.61:
        return "substantial"
    elif k >= 0.41:
        return "moderate"
    elif k >= 0.21:
        return "fair"
    else:
        return "poor"


# ═══════════════════════════════════════════════════════════════════════
# WILSON SCORE CONFIDENCE INTERVAL
# ═══════════════════════════════════════════════════════════════════════

def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    More accurate than the normal approximation for small n or extreme p.
    Reference: Wilson (1927), Agresti & Coull (1998).

    Args:
        n_success: Number of successes
        n_total: Total number of trials
        z: Z-score (1.96 for 95% CI)

    Returns:
        (lower, upper) bounds of the confidence interval
    """
    if n_total == 0:
        return (0.0, 0.0)

    p = n_success / n_total
    denominator = 1 + z ** 2 / n_total
    center = (p + z ** 2 / (2 * n_total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n_total)) / n_total) / denominator

    return (max(0.0, center - margin), min(1.0, center + margin))


# ═══════════════════════════════════════════════════════════════════════
# 1. SEMANTIC GAP AUDIT
# ═══════════════════════════════════════════════════════════════════════

def semantic_gap_audit(
    df_raw: pd.DataFrame,
    screening_results: pd.DataFrame,
    exclude_mask: pd.Series,
    n_sample: int = 200,
    seed: int = 42,
) -> dict:
    """
    Audit: How many papers did the keyword filter exclude
    that the LLM would have included?

    Reports point estimate with 95% Wilson confidence interval.

    Args:
        df_raw: Raw (unfiltered) DataFrame
        screening_results: LLM screening results with 'doi', 'relevant' columns
        exclude_mask: Boolean mask of papers excluded by keyword filter
        n_sample: Number of excluded papers to sample (default 200)
        seed: Random seed for reproducibility

    Returns:
        dict with audit metrics including confidence interval
    """
    print(f"\n{'─' * 70}")
    print("SEMANTIC GAP AUDIT")
    print(f"{'─' * 70}")

    excluded = df_raw[exclude_mask].copy()
    n_excluded_total = len(excluded)

    if n_excluded_total == 0:
        print("  No excluded papers to audit.")
        return {"n_excluded": 0, "n_sampled": 0, "recall_gain": 0}

    # Sample using pandas for consistent random state
    n_sample = min(n_sample, n_excluded_total)
    sample = excluded.sample(n=n_sample, random_state=seed)

    # Build screening lookup — handle NaN in 'relevant' safely
    screening_clean = screening_results.dropna(subset=["relevant"])
    screening_dict = dict(
        zip(
            screening_clean["doi"].astype(str),
            screening_clean["relevant"].astype(bool),
        )
    )

    llm_relevant = 0
    llm_relevant_papers = []
    for _, row in sample.iterrows():
        doi = str(row.get("doi", ""))
        if screening_dict.get(doi, False):
            llm_relevant += 1
            llm_relevant_papers.append(row.get("title", ""))

    recall_gain_pct = llm_relevant / n_sample * 100

    # Wilson 95% CI
    ci_lower, ci_upper = wilson_ci(llm_relevant, n_sample)

    print(f"  Total keyword-excluded papers: {n_excluded_total:,}")
    print(f"  Sampled for audit: {n_sample}")
    print(f"  LLM says relevant: {llm_relevant} ({recall_gain_pct:.1f}%)")
    print(f"  95% CI (Wilson): [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
    print(f"  -> Potential Recall Gain: {recall_gain_pct:.1f}% "
          f"[{ci_lower*100:.1f}%-{ci_upper*100:.1f}%]")

    if llm_relevant_papers:
        print(f"\n  Examples of papers regex missed but LLM found:")
        for t in llm_relevant_papers[:5]:
            print(f"    -> {str(t)[:90]}")

    result = {
        "n_excluded_total": n_excluded_total,
        "n_sampled": n_sample,
        "llm_relevant_in_sample": llm_relevant,
        "recall_gain_pct": round(recall_gain_pct, 2),
        "recall_gain_ci_lower": round(ci_lower * 100, 2),
        "recall_gain_ci_upper": round(ci_upper * 100, 2),
        "seed": seed,
    }

    return result


# ═══════════════════════════════════════════════════════════════════════
# 2. LLM vs REGEX CONCEPT AGREEMENT
# ═══════════════════════════════════════════════════════════════════════

def concept_agreement(
    df: pd.DataFrame,
    llm_concepts_df: pd.DataFrame,
    min_prevalence: int = 10,
) -> pd.DataFrame:
    """
    Compare LLM concept classification vs regex concept detection.

    Computes Cohen's kappa per concept. Reports both mean and
    prevalence-weighted mean kappa. Flags concepts below min_prevalence.

    Args:
        df: Filtered DataFrame with 'doi', 'title' columns
        llm_concepts_df: LLM concept results with 'doi', 'concepts' columns
        min_prevalence: Minimum combined count for stable kappa (default 10)

    Returns:
        DataFrame with per-concept agreement metrics
    """
    print(f"\n{'─' * 70}")
    print("LLM vs REGEX CONCEPT AGREEMENT")
    print(f"{'─' * 70}")

    # Merge on DOI
    merged = df[["doi", "title"]].copy()
    merged["doi"] = merged["doi"].astype(str)
    llm_copy = llm_concepts_df[["doi", "concepts"]].copy()
    llm_copy["doi"] = llm_copy["doi"].astype(str)
    merged = merged.merge(llm_copy, on="doi", how="left")
    merged["concepts"] = merged["concepts"].fillna("")

    titles_lower = merged["title"].fillna("").str.lower()

    results = []
    for concept, pattern in CLINICAL_CONCEPTS.items():
        # Regex detection (title-only)
        try:
            regex_match = titles_lower.str.contains(pattern, regex=True, na=False)
        except Exception:
            continue

        # LLM detection
        llm_match = merged["concepts"].str.contains(concept, na=False, regex=False)

        # Compute kappa
        kappa = cohens_kappa(regex_match.values, llm_match.values)
        interp = kappa_interpretation(kappa)

        # Counts
        regex_count = int(regex_match.sum())
        llm_count = int(llm_match.sum())
        both = int((regex_match & llm_match).sum())
        regex_only = int((regex_match & ~llm_match).sum())
        llm_only = int((~regex_match & llm_match).sum())
        neither = int((~regex_match & ~llm_match).sum())

        # Flag low-prevalence concepts where kappa is unstable
        combined = regex_count + llm_count - both
        stable = combined >= min_prevalence

        results.append({
            "concept": concept,
            "regex_count": regex_count,
            "llm_count": llm_count,
            "both": both,
            "regex_only": regex_only,
            "llm_only": llm_only,
            "kappa": kappa,
            "interpretation": interp,
            "agreement_pct": round((both + neither) / len(merged) * 100, 2),
            "stable_kappa": stable,
        })

    results_df = pd.DataFrame(results).sort_values("kappa", ascending=True)

    # Print summary — both mean and prevalence-weighted mean
    mean_kappa = results_df["kappa"].mean()
    stable_mask = results_df["stable_kappa"]
    stable_mean_kappa = results_df.loc[stable_mask, "kappa"].mean() if stable_mask.any() else 0

    # Prevalence-weighted kappa
    total_count = results_df["regex_count"] + results_df["llm_count"]
    if total_count.sum() > 0:
        weighted_kappa = (results_df["kappa"] * total_count).sum() / total_count.sum()
    else:
        weighted_kappa = 0

    print(f"\n  Mean Cohen's k across all concepts: {mean_kappa:.3f} ({kappa_interpretation(mean_kappa)})")
    print(f"  Mean k (stable concepts, n>={min_prevalence}): {stable_mean_kappa:.3f}")
    print(f"  Prevalence-weighted k: {weighted_kappa:.3f}")
    print(f"  Unstable concepts (n<{min_prevalence}): {(~stable_mask).sum()}")
    print(f"\n  Per-concept agreement:")
    print(f"  {'Concept':>30s}  {'Regex':>5}  {'LLM':>5}  {'Both':>5}  {'k':>6}  {'Interpretation'}")
    print(f"  {'─' * 85}")
    for _, row in results_df.iterrows():
        flag = " *" if not row["stable_kappa"] else ""
        print(f"  {row['concept']:>30s}  {row['regex_count']:>5}  {row['llm_count']:>5}  "
              f"{row['both']:>5}  {row['kappa']:>6.3f}  {row['interpretation']}{flag}")

    if not stable_mask.all():
        print(f"\n  * = kappa unstable (concept prevalence < {min_prevalence})")

    return results_df


# ═══════════════════════════════════════════════════════════════════════
# 3. SCREENING AGREEMENT (LLM vs Keyword Filter)
# ═══════════════════════════════════════════════════════════════════════

def screening_agreement(
    df_keyword_filtered: pd.DataFrame,
    df_raw: pd.DataFrame,
    screening_results: pd.DataFrame,
) -> dict:
    """
    Compare LLM screening decisions vs keyword filter decisions.

    Args:
        df_keyword_filtered: Papers that PASSED the keyword filter
        df_raw: ALL raw papers
        screening_results: LLM screening results

    Returns:
        dict with agreement metrics
    """
    print(f"\n{'─' * 70}")
    print("SCREENING AGREEMENT: LLM vs KEYWORD FILTER")
    print(f"{'─' * 70}")

    # Build lookup: DOI -> keyword filter decision
    keyword_included = set(df_keyword_filtered["doi"].astype(str).values)
    all_dois = set(df_raw["doi"].astype(str).values)

    # Build lookup: DOI -> LLM decision (drop NaN/None values safely)
    screening_clean = screening_results.dropna(subset=["relevant"])
    llm_decisions = dict(
        zip(
            screening_clean["doi"].astype(str),
            screening_clean["relevant"].astype(bool),
        )
    )

    # Compare on papers that have both decisions
    common_dois = list(all_dois & set(llm_decisions.keys()))

    if not common_dois:
        print("  No common papers to compare.")
        return {"n_compared": 0}

    keyword_arr = np.array([doi in keyword_included for doi in common_dois])
    llm_arr = np.array([llm_decisions.get(doi, False) for doi in common_dois])

    kappa = cohens_kappa(keyword_arr, llm_arr)

    agree = np.sum(keyword_arr == llm_arr)
    disagree = len(common_dois) - agree
    agree_pct = agree / len(common_dois) * 100

    # Disagreement breakdown
    keyword_only = int(np.sum(keyword_arr & ~llm_arr))
    llm_only = int(np.sum(~keyword_arr & llm_arr))

    print(f"  Papers compared: {len(common_dois):,}")
    print(f"  Agreement: {agree:,} ({agree_pct:.1f}%)")
    print(f"  Disagreement: {disagree:,}")
    print(f"    Keyword Y / LLM N: {keyword_only:,} (keyword may over-include)")
    print(f"    Keyword N / LLM Y: {llm_only:,} (keyword may miss these)")
    print(f"  Cohen's k: {kappa:.3f} ({kappa_interpretation(kappa)})")

    return {
        "n_compared": len(common_dois),
        "n_agree": int(agree),
        "n_disagree": int(disagree),
        "agreement_pct": round(agree_pct, 2),
        "keyword_only": keyword_only,
        "llm_only": llm_only,
        "kappa": kappa,
        "interpretation": kappa_interpretation(kappa),
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. DISPUTE EXPORT + IMPORT (Human-in-the-Loop)
# ═══════════════════════════════════════════════════════════════════════

def export_disputes(
    df_raw: pd.DataFrame,
    df_keyword_filtered: pd.DataFrame,
    screening_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Export papers where LLM and keyword filter disagree for human review.

    Returns:
        DataFrame of disputes
    """
    print(f"\n{'─' * 70}")
    print("EXPORTING DISPUTES FOR HUMAN REVIEW")
    print(f"{'─' * 70}")

    keyword_included = set(df_keyword_filtered["doi"].astype(str).values)

    disputes = []
    for _, row in screening_results.iterrows():
        doi = str(row.get("doi", ""))
        relevant_raw = row.get("relevant")

        # Skip error results
        if pd.isna(relevant_raw):
            continue

        llm_relevant = bool(relevant_raw)
        keyword_relevant = doi in keyword_included
        confidence = float(row.get("confidence", 0))

        if llm_relevant != keyword_relevant:
            # Safely get title from df_raw
            title_match = df_raw.loc[df_raw["doi"].astype(str) == doi, "title"]
            title = str(title_match.iloc[0]) if len(title_match) > 0 else ""

            disputes.append({
                "doi": doi,
                "title": title,
                "keyword_decision": "include" if keyword_relevant else "exclude",
                "llm_decision": "include" if llm_relevant else "exclude",
                "llm_confidence": confidence,
                "llm_reason": str(row.get("reason", "")),
                "human_override": "",  # To be filled by human reviewer
            })

    disputes_df = pd.DataFrame(disputes)

    if len(disputes_df) > 0:
        # Sort by confidence (lowest first — most uncertain)
        disputes_df = disputes_df.sort_values("llm_confidence", ascending=True)
        disputes_df.to_csv(DISPUTES_FILE, index=False)
        print(f"  Disputes exported: {len(disputes_df):,}")
        print(f"  High-priority (confidence < {LLM_CONFIDENCE_REVIEW}): "
              f"{(disputes_df['llm_confidence'] < LLM_CONFIDENCE_REVIEW).sum():,}")
        print(f"  Saved to: {DISPUTES_FILE}")
        print(f"\n  To close the HITL loop:")
        print(f"    1. Edit 'human_override' column (include/exclude)")
        print(f"    2. Re-run pipeline — overrides will be applied automatically")
    else:
        print("  No disputes found — LLM and keyword filter fully agree!")

    return disputes_df


def import_disputes(screening_results: pd.DataFrame) -> pd.DataFrame:
    """
    Import human overrides from disputes.csv and apply them to screening results.

    This closes the HITL loop: human decisions in the 'human_override' column
    override the original LLM or keyword decisions.

    Args:
        screening_results: Original LLM screening results

    Returns:
        Updated screening results with human overrides applied
    """
    if not DISPUTES_FILE.exists():
        return screening_results

    try:
        disputes = pd.read_csv(DISPUTES_FILE)
    except Exception as e:
        logger.warning(f"Could not read disputes file: {e}")
        return screening_results

    # Filter to rows where human_override is filled in
    overrides = disputes[disputes["human_override"].notna() & (disputes["human_override"] != "")]

    if len(overrides) == 0:
        return screening_results

    print(f"\n  Applying {len(overrides)} human overrides from disputes.csv")

    # Apply overrides
    results = screening_results.copy()
    override_dict = dict(
        zip(overrides["doi"].astype(str), overrides["human_override"].str.strip().str.lower())
    )

    n_applied = 0
    for idx, row in results.iterrows():
        doi = str(row.get("doi", ""))
        if doi in override_dict:
            decision = override_dict[doi]
            if decision in ("include", "yes", "true"):
                results.at[idx, "relevant"] = True
                results.at[idx, "needs_review"] = False
                results.at[idx, "reason"] = "Human override: include"
                n_applied += 1
            elif decision in ("exclude", "no", "false"):
                results.at[idx, "relevant"] = False
                results.at[idx, "needs_review"] = False
                results.at[idx, "reason"] = "Human override: exclude"
                n_applied += 1

    print(f"  Applied {n_applied} overrides ({len(overrides) - n_applied} unrecognized)")
    return results


# ═══════════════════════════════════════════════════════════════════════
# 5. SELF-CONSISTENCY CHECK
# ═══════════════════════════════════════════════════════════════════════

def self_consistency_check(
    df: pd.DataFrame,
    llm: "LLMProvider",
    n_sample: int = 100,
    seed: int = 42,
) -> dict:
    """
    Run the same LLM on the same papers twice (temp=0) and measure agreement.

    This measures the degree of output consistency (not strict determinism —
    temperature=0 does not guarantee identical outputs across API calls due
    to GPU floating-point non-determinism).

    Args:
        df: DataFrame with papers
        llm: LLMProvider instance
        n_sample: Number of papers to test
        seed: Random seed for reproducible sampling

    Returns:
        dict with self-consistency metrics
    """
    from llm_schemas import ScreeningResult

    print(f"\n{'─' * 70}")
    print(f"SELF-CONSISTENCY CHECK (n={n_sample})")
    print(f"{'─' * 70}")

    # Use only pandas random_state (not Python random.seed, which has no effect here)
    sample = df.sample(min(n_sample, len(df)), random_state=seed)

    from llm_screening import build_full_system_prompt, build_user_prompt
    system = build_full_system_prompt()

    run1 = []
    run2 = []

    for i, (_, row) in enumerate(sample.iterrows()):
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        user = build_user_prompt(title, abstract)

        try:
            r1 = llm.query(system=system, user=user, schema=ScreeningResult)
            r2 = llm.query(system=system, user=user, schema=ScreeningResult)
            run1.append(r1.relevant)
            run2.append(r2.relevant)
        except Exception as e:
            logger.warning(f"Error on paper {i}: {e}")
            continue

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(sample)}")

    if not run1:
        return {"n_tested": 0, "consistency": 0.0}

    run1 = np.array(run1)
    run2 = np.array(run2)
    agree = np.sum(run1 == run2)
    consistency = agree / len(run1) * 100

    print(f"\n  Papers tested: {len(run1)}")
    print(f"  Same decision both runs: {agree} ({consistency:.1f}%)")
    print(f"  {'PASS' if consistency >= 95 else 'WARN'}: "
          f"Self-consistency {'≥' if consistency >= 95 else '<'}95%")

    return {
        "n_tested": len(run1),
        "n_agree": int(agree),
        "consistency_pct": round(consistency, 2),
        "pass": consistency >= 95,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. AUDIT SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════

def print_audit_table(
    screening_agreement_result: dict,
    concept_agreement_df: pd.DataFrame,
    semantic_gap_result: dict,
    extraction_df: Optional[pd.DataFrame] = None,
    consistency_result: Optional[dict] = None,
):
    """Print the publication-ready audit summary table."""
    print(f"\n{'=' * 70}")
    print("AUDIT SUMMARY TABLE (for publication)")
    print(f"{'=' * 70}")

    print(f"\n{'Step':<25} {'Regex Baseline':<25} {'LLM Performance':<25} {'Improvement':<25}")
    print(f"{'─' * 100}")

    # Screening
    kw_agree = screening_agreement_result.get("agreement_pct", 0)
    llm_only = screening_agreement_result.get("llm_only", 0)
    kappa_s = screening_agreement_result.get("kappa", 0)
    print(f"{'Relevance Screen':<25} "
          f"{'Title keyword filter':<25} "
          f"{'Title+abstract LLM':<25} "
          f"{'k=' + str(kappa_s) + ', +' + str(llm_only) + ' papers':<25}")

    # Concept detection
    mean_kappa = concept_agreement_df["kappa"].mean() if len(concept_agreement_df) > 0 else 0
    mean_llm = concept_agreement_df["llm_count"].mean() if len(concept_agreement_df) > 0 else 0
    mean_regex = concept_agreement_df["regex_count"].mean() if len(concept_agreement_df) > 0 else 0
    print(f"{'Concept Detection':<25} "
          f"{'Regex (' + str(int(mean_regex)) + '/concept)':<25} "
          f"{'LLM (' + str(int(mean_llm)) + '/concept)':<25} "
          f"{'k=' + f'{mean_kappa:.3f}':<25}")

    # Data extraction
    if extraction_df is not None:
        has_design = (extraction_df["study_design"] != "other").sum()
        has_n = extraction_df["sample_size"].notna().sum()
        print(f"{'Data Extraction':<25} "
              f"{'N/A (manual only)':<25} "
              f"{'Automated JSON':<25} "
              f"{str(has_design) + ' designs, ' + str(has_n) + ' sample sizes':<25}")
    else:
        print(f"{'Data Extraction':<25} {'N/A':<25} {'Not run':<25} {'—':<25}")

    # Semantic gap
    recall_gain = semantic_gap_result.get("recall_gain_pct", 0)
    ci_lo = semantic_gap_result.get("recall_gain_ci_lower", 0)
    ci_hi = semantic_gap_result.get("recall_gain_ci_upper", 0)
    ci_str = f" [{ci_lo:.1f}%-{ci_hi:.1f}%]" if ci_lo and ci_hi else ""
    print(f"{'Search Recall':<25} "
          f"{'Keyword-based':<25} "
          f"{'LLM-augmented':<25} "
          f"{'Gain: ' + f'{recall_gain:.1f}%' + ci_str:<25}")

    # Self-consistency
    if consistency_result:
        cons = consistency_result.get("consistency_pct", 0)
        print(f"{'Self-Consistency':<25} "
              f"{'N/A':<25} "
              f"{f'{cons:.1f}% (temp=0)':<25} "
              f"{'PASS' if cons >= 95 else 'WARN':<25}")

    print(f"\n{'─' * 100}")
    print(f"  LLM Provider: (see validation_report.json)")
    print(f"  Confidence threshold: auto-accept >={LLM_CONFIDENCE_AUTO}, "
          f"review {LLM_CONFIDENCE_REVIEW}-{LLM_CONFIDENCE_AUTO}")
    print(f"  Kappa benchmarks: Landis & Koch (1977) — note: developed for human-vs-human agreement")


def save_validation_report(report: dict):
    """Save full validation report to JSON."""
    CACHE_DIR.mkdir(exist_ok=True)
    from llm_utils import save_cache as _atomic_save
    _atomic_save(report, VALIDATION_REPORT)
    print(f"\n  Full report saved to: {VALIDATION_REPORT}")
