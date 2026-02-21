"""
LLM Validation Module
======================
Eight validation layers for publication-defensible LLM-augmented bibliometrics:

  1. Semantic Gap Audit — quantifies search recall improvement (with Wilson CI)
  2. LLM vs Regex Agreement — Cohen's kappa per concept
  3. Screening Agreement — LLM vs keyword filter with kappa
  4. Human-in-the-Loop Dispute Export + Override Import
  5. Self-Consistency Check — same LLM on same papers twice at temp=0
  6. Three-Run Consensus Check — majority-vote with temperature variation
  7. Validation Sample Generator — stratified sample for human annotation
  8. Human-LLM Agreement — P/R/F1 and kappa against human gold standard

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
    CONSENSUS_RUNS, CONSENSUS_TEMPERATURES, VALIDATION_SAMPLE_SIZE,
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
# 6. THREE-RUN CONSENSUS CHECK
# ═══════════════════════════════════════════════════════════════════════

def _krippendorff_alpha_nominal(ratings_matrix: np.ndarray) -> float:
    """
    Compute Krippendorff's alpha for nominal data.

    Args:
        ratings_matrix: shape (n_raters, n_items) — each cell is a category label.
                        NaN values indicate missing ratings.

    Returns:
        alpha in [-1, 1]. 1.0 = perfect agreement, 0.0 = chance level.

    Reference: Krippendorff (2004). Content Analysis, 3rd ed.
    """
    n_raters, n_items = ratings_matrix.shape
    if n_items < 2 or n_raters < 2:
        return 0.0

    # Collect all categories present
    categories = set()
    for val in ratings_matrix.flatten():
        if not (isinstance(val, float) and np.isnan(val)):
            categories.add(val)
    categories = sorted(categories)

    if len(categories) < 2:
        return 1.0  # All same category → perfect agreement

    # Observed disagreement (Do)
    total_pairs = 0
    disagreement_pairs = 0
    for j in range(n_items):
        col = ratings_matrix[:, j]
        valid = [v for v in col if not (isinstance(v, float) and np.isnan(v))]
        m = len(valid)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                total_pairs += 1
                if valid[a] != valid[b]:
                    disagreement_pairs += 1

    if total_pairs == 0:
        return 0.0

    D_o = disagreement_pairs / total_pairs

    # Expected disagreement (De) — based on marginal frequencies
    all_values = []
    for j in range(n_items):
        col = ratings_matrix[:, j]
        for v in col:
            if not (isinstance(v, float) and np.isnan(v)):
                all_values.append(v)

    n_total = len(all_values)
    if n_total < 2:
        return 0.0

    freq = Counter(all_values)
    D_e = 1.0 - sum((count / n_total) ** 2 for count in freq.values())

    if D_e == 0:
        return 1.0

    alpha = 1.0 - D_o / D_e
    return round(alpha, 4)


def consensus_check(
    df: pd.DataFrame,
    llm: "LLMProvider",
    n_sample: int = 100,
    seed: int = 42,
    temperatures: tuple = None,
    task: str = "screen",
) -> dict:
    """
    Three-run consensus check with temperature variation.

    Runs the LLM multiple times on the same papers and measures agreement
    via majority voting. This validates reproducibility claims:
    - Runs 1-2 at temp=0: tests API non-determinism (GPU floating-point)
    - Run 3 at temp=0.3: tests prompt sensitivity / robustness

    Args:
        df: DataFrame with papers (needs 'title', 'abstract' columns)
        llm: LLMProvider instance
        n_sample: Number of papers to test
        seed: Random seed for reproducible sampling
        temperatures: Tuple of temperatures per run (default from config)
        task: "screen" (only screening supported currently)

    Returns:
        dict with consensus metrics including Krippendorff's alpha
    """
    from llm_schemas import ScreeningResult

    if temperatures is None:
        temperatures = CONSENSUS_TEMPERATURES

    n_runs = len(temperatures)

    print(f"\n{'─' * 70}")
    print(f"CONSENSUS CHECK ({n_runs}-run, n={n_sample})")
    print(f"{'─' * 70}")
    print(f"  Temperatures: {temperatures}")

    sample = df.sample(min(n_sample, len(df)), random_state=seed)

    from llm_screening import build_full_system_prompt, build_user_prompt
    system = build_full_system_prompt()

    # Store results per paper
    all_runs = [[] for _ in range(n_runs)]
    original_temp = llm.temperature
    completed = 0

    for i, (_, row) in enumerate(sample.iterrows()):
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        user = build_user_prompt(title, abstract)

        paper_ok = True
        paper_results = []
        for run_idx, temp in enumerate(temperatures):
            try:
                llm.temperature = temp
                result = llm.query(system=system, user=user, schema=ScreeningResult)
                paper_results.append(result.relevant)
            except Exception as e:
                logger.warning(f"Error on paper {i}, run {run_idx}: {e}")
                paper_ok = False
                break

        if paper_ok and len(paper_results) == n_runs:
            for run_idx, val in enumerate(paper_results):
                all_runs[run_idx].append(val)
            completed += 1

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(sample)} (completed: {completed})")

    # Restore original temperature
    llm.temperature = original_temp

    if completed == 0:
        return {"n_tested": 0, "pass": False}

    # Convert to arrays
    runs = [np.array(r) for r in all_runs]

    # Majority vote analysis
    vote_matrix = np.column_stack(runs)  # shape: (n_papers, n_runs)
    majority_votes = (vote_matrix.sum(axis=1) > n_runs / 2)

    unanimous_mask = (vote_matrix.sum(axis=1) == 0) | (vote_matrix.sum(axis=1) == n_runs)
    unanimous_pct = unanimous_mask.sum() / completed * 100

    # Majority = at least ceil(n_runs/2) agree
    majority_count = np.maximum(vote_matrix.sum(axis=1), n_runs - vote_matrix.sum(axis=1))
    majority_mask = majority_count >= math.ceil(n_runs / 2)
    majority_pct = majority_mask.sum() / completed * 100

    # Flip rate: papers where at least one run disagrees with majority
    flip_count = completed - unanimous_mask.sum()
    flip_rate = flip_count / completed * 100

    # Pairwise agreement between temperature groups
    per_temp_agreement = {}
    for a_idx in range(n_runs):
        for b_idx in range(a_idx + 1, n_runs):
            key = f"t{temperatures[a_idx]}_vs_t{temperatures[b_idx]}"
            agree = np.sum(runs[a_idx] == runs[b_idx])
            per_temp_agreement[key] = round(agree / completed * 100, 2)

    # Krippendorff's alpha
    ratings = np.array([[int(r) for r in run] for run in all_runs], dtype=float)
    alpha = _krippendorff_alpha_nominal(ratings)

    # Detailed per-paper vote breakdown for figures
    vote_details = []
    for j in range(completed):
        yes_votes = int(vote_matrix[j].sum())
        vote_details.append({
            "paper_idx": j,
            "yes_votes": yes_votes,
            "no_votes": n_runs - yes_votes,
            "unanimous": bool(unanimous_mask[j]),
            "majority_decision": bool(majority_votes[j]),
        })

    passed = unanimous_pct >= 85  # 85% unanimous is robust

    print(f"\n  Papers tested: {completed}")
    print(f"  Unanimous ({n_runs}/{n_runs}): {unanimous_mask.sum()} ({unanimous_pct:.1f}%)")
    print(f"  Majority vote: {majority_mask.sum()} ({majority_pct:.1f}%)")
    print(f"  Flip rate (≥1 run disagrees): {flip_count} ({flip_rate:.1f}%)")
    print(f"  Krippendorff's alpha: {alpha:.4f}")
    for key, val in per_temp_agreement.items():
        print(f"    {key}: {val:.1f}%")
    print(f"  {'PASS' if passed else 'WARN'}: "
          f"Unanimous rate {'≥' if unanimous_pct >= 85 else '<'}85%")

    return {
        "n_tested": completed,
        "n_runs": n_runs,
        "temperatures": list(temperatures),
        "unanimous_count": int(unanimous_mask.sum()),
        "unanimous_pct": round(unanimous_pct, 2),
        "majority_pct": round(majority_pct, 2),
        "flip_count": int(flip_count),
        "flip_rate": round(flip_rate, 2),
        "krippendorff_alpha": alpha,
        "per_temperature_agreement": per_temp_agreement,
        "vote_details": vote_details,
        "pass": passed,
    }


# ═══════════════════════════════════════════════════════════════════════
# 7. VALIDATION SAMPLE GENERATOR (for Human Annotation)
# ═══════════════════════════════════════════════════════════════════════

VALIDATION_SAMPLE_FILE = CACHE_DIR / "validation_sample.csv"
VALIDATION_INSTRUCTIONS_FILE = CACHE_DIR / "validation_instructions.txt"


def generate_validation_sample(
    df: pd.DataFrame,
    screening_results: pd.DataFrame,
    classification_results: pd.DataFrame = None,
    extraction_results: pd.DataFrame = None,
    n_total: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a stratified random sample for human validation.

    Stratifies by LLM screening confidence quartiles and ensures each
    major concept has at least 3 papers in the sample.

    Args:
        df: Full DataFrame with 'doi', 'title', 'abstract'
        screening_results: LLM screening with 'doi', 'relevant', 'confidence'
        classification_results: LLM concepts with 'doi', 'concepts', 'primary_concept'
        extraction_results: LLM extraction with 'doi', 'study_design', 'sample_size'
        n_total: Total sample size (default from config)
        seed: Random seed

    Returns:
        DataFrame of sampled papers with blank human annotation columns
    """
    if n_total is None:
        n_total = VALIDATION_SAMPLE_SIZE

    print(f"\n{'─' * 70}")
    print(f"GENERATING VALIDATION SAMPLE (n={n_total})")
    print(f"{'─' * 70}")

    # Merge screening results onto papers
    merged = df[["doi", "title", "abstract"]].copy()
    merged["doi"] = merged["doi"].astype(str)

    screen_copy = screening_results[["doi", "relevant", "confidence", "reason"]].copy()
    screen_copy["doi"] = screen_copy["doi"].astype(str)
    screen_copy = screen_copy.rename(columns={
        "relevant": "llm_relevant",
        "confidence": "llm_confidence",
        "reason": "llm_reason",
    })
    merged = merged.merge(screen_copy, on="doi", how="inner")

    # Add classification data if available
    if classification_results is not None:
        class_copy = classification_results[["doi", "concepts", "primary_concept"]].copy()
        class_copy["doi"] = class_copy["doi"].astype(str)
        class_copy = class_copy.rename(columns={
            "concepts": "llm_concepts",
            "primary_concept": "llm_primary_concept",
        })
        merged = merged.merge(class_copy, on="doi", how="left")
    else:
        merged["llm_concepts"] = ""
        merged["llm_primary_concept"] = ""

    # Add extraction data if available
    if extraction_results is not None:
        ext_cols = ["doi"]
        for col in ["study_design", "sample_size", "level_of_evidence"]:
            if col in extraction_results.columns:
                ext_cols.append(col)
        ext_copy = extraction_results[ext_cols].copy()
        ext_copy["doi"] = ext_copy["doi"].astype(str)
        ext_copy = ext_copy.rename(columns={
            "study_design": "llm_study_design",
            "sample_size": "llm_sample_size",
            "level_of_evidence": "llm_level_of_evidence",
        })
        merged = merged.merge(ext_copy, on="doi", how="left")
    else:
        merged["llm_study_design"] = ""
        merged["llm_sample_size"] = ""

    # Drop rows without valid confidence
    merged = merged.dropna(subset=["llm_confidence"])

    # Stratify by confidence quartiles
    merged["confidence_quartile"] = pd.qcut(
        merged["llm_confidence"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        duplicates="drop",
    )

    n_per_quartile = n_total // 4
    remainder = n_total - n_per_quartile * 4

    sampled_parts = []
    for i, (quartile, group) in enumerate(merged.groupby("confidence_quartile", observed=True)):
        n_take = n_per_quartile + (1 if i < remainder else 0)
        n_take = min(n_take, len(group))
        sampled_parts.append(group.sample(n=n_take, random_state=seed))

    sample = pd.concat(sampled_parts, ignore_index=True)

    # Truncate abstracts for readability
    sample["abstract"] = sample["abstract"].fillna("").str[:500]

    # Add blank columns for human annotation
    sample["human_relevant"] = ""
    sample["human_concepts"] = ""
    sample["human_study_design"] = ""
    sample["human_sample_size"] = ""
    sample["human_notes"] = ""

    # If two raters, add second rater columns
    sample["human2_relevant"] = ""
    sample["human2_concepts"] = ""
    sample["adjudicated_relevant"] = ""
    sample["adjudicated_concepts"] = ""

    # Drop internal columns
    sample = sample.drop(columns=["confidence_quartile"], errors="ignore")

    # Save
    CACHE_DIR.mkdir(exist_ok=True)
    sample.to_csv(VALIDATION_SAMPLE_FILE, index=False)

    # Write annotation instructions
    instructions = f"""HUMAN VALIDATION INSTRUCTIONS
==============================
Field: {FIELD_NAME}
Sample size: {len(sample)} papers
Generated: {pd.Timestamp.now().isoformat()}

TASK 1: RELEVANCE SCREENING
  For each paper, fill in the 'human_relevant' column:
    - "yes" if the paper is relevant to {FIELD_NAME}
    - "no" if it is NOT relevant
  Compare your decision against 'llm_relevant' and 'llm_reason'.

TASK 2: CONCEPT CLASSIFICATION
  Fill in 'human_concepts' with pipe-separated concept names from this list:
    {', '.join(sorted(CLINICAL_CONCEPTS.keys()))}
  Example: "REBOA|Hemorrhage Control|Damage Control"
  If no concepts apply, leave blank.

TASK 3: STUDY DESIGN (optional)
  Fill in 'human_study_design' with one of:
    RCT, prospective_observational, retrospective_cohort, case_control,
    cross_sectional, systematic_review, meta_analysis, case_series,
    case_report, narrative_review, guideline, registry_study,
    experimental, qualitative, other

TASK 4: SAMPLE SIZE (optional)
  Fill in 'human_sample_size' with the integer sample size, or leave blank.

INTER-RATER RELIABILITY
  If a second rater is available, they should fill 'human2_relevant' and
  'human2_concepts'. Disagreements should be resolved in 'adjudicated_relevant'
  and 'adjudicated_concepts'.

After completing annotation, run:
  python llm_pipeline.py --compute-human-agreement
"""
    VALIDATION_INSTRUCTIONS_FILE.write_text(instructions)

    # Summary
    quartile_counts = sample.groupby(
        pd.qcut(sample["llm_confidence"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"],
                duplicates="drop"),
        observed=True,
    ).size()

    print(f"  Total sampled: {len(sample)}")
    print(f"  Confidence distribution:")
    for q, count in quartile_counts.items():
        print(f"    {q}: {count}")
    print(f"  Saved to: {VALIDATION_SAMPLE_FILE}")
    print(f"  Instructions: {VALIDATION_INSTRUCTIONS_FILE}")
    print(f"\n  Next step: annotate the CSV, then run --compute-human-agreement")

    return sample


# ═══════════════════════════════════════════════════════════════════════
# 8. HUMAN-LLM AGREEMENT (P/R/F1 against Gold Standard)
# ═══════════════════════════════════════════════════════════════════════

def compute_human_llm_agreement(
    sample_path: Path = None,
) -> dict:
    """
    Compute agreement between LLM predictions and human annotations.

    Loads the completed validation_sample.csv (with human columns filled)
    and computes precision, recall, F1, Cohen's kappa, sensitivity,
    specificity for screening and classification.

    Args:
        sample_path: Path to annotated CSV (default: llm_cache/validation_sample.csv)

    Returns:
        dict with structured agreement metrics for JMIR Table 2
    """
    if sample_path is None:
        sample_path = VALIDATION_SAMPLE_FILE

    if not sample_path.exists():
        print(f"  ERROR: Validation sample not found at {sample_path}")
        print(f"  Run --generate-validation-sample first, then annotate the CSV.")
        return {}

    print(f"\n{'─' * 70}")
    print("HUMAN-LLM AGREEMENT ANALYSIS")
    print(f"{'─' * 70}")

    df = pd.read_csv(sample_path)

    results = {}

    # ── Screening Agreement ──────────────────────────────────────────
    # Use adjudicated column if available, otherwise human_relevant
    if "adjudicated_relevant" in df.columns and df["adjudicated_relevant"].notna().any():
        human_col = "adjudicated_relevant"
    elif "human_relevant" in df.columns:
        human_col = "human_relevant"
    else:
        print("  No human annotation columns found.")
        return {}

    screen_df = df.dropna(subset=[human_col])
    screen_df = screen_df[screen_df[human_col].astype(str).str.strip() != ""]

    if len(screen_df) == 0:
        print("  No human screening annotations found.")
        return {}

    # Parse human decisions
    human_rel = screen_df[human_col].astype(str).str.strip().str.lower()
    human_bool = human_rel.isin(["yes", "true", "1", "include"])

    llm_bool = screen_df["llm_relevant"].astype(bool)

    # Confusion matrix
    tp = int((human_bool & llm_bool).sum())
    tn = int((~human_bool & ~llm_bool).sum())
    fp = int((~human_bool & llm_bool).sum())
    fn = int((human_bool & ~llm_bool).sum())
    n_total = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0

    kappa_screen = cohens_kappa(human_bool.values, llm_bool.values)

    # Wilson CIs
    precision_ci = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)
    recall_ci = wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (0.0, 0.0)

    results["screening"] = {
        "n": n_total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "precision_ci": [round(x, 4) for x in precision_ci],
        "recall": round(recall, 4),
        "recall_ci": [round(x, 4) for x in recall_ci],
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "kappa": kappa_screen,
        "kappa_interpretation": kappa_interpretation(kappa_screen),
    }

    print(f"\n  SCREENING (n={n_total}):")
    print(f"    Confusion matrix: TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"    Precision: {precision:.3f} [{precision_ci[0]:.3f}-{precision_ci[1]:.3f}]")
    print(f"    Recall:    {recall:.3f} [{recall_ci[0]:.3f}-{recall_ci[1]:.3f}]")
    print(f"    F1:        {f1:.3f}")
    print(f"    Specificity: {specificity:.3f}")
    print(f"    Cohen's k: {kappa_screen:.3f} ({kappa_interpretation(kappa_screen)})")

    # ── Classification Agreement ─────────────────────────────────────
    if "adjudicated_concepts" in df.columns and df["adjudicated_concepts"].notna().any():
        concept_human_col = "adjudicated_concepts"
    elif "human_concepts" in df.columns:
        concept_human_col = "human_concepts"
    else:
        concept_human_col = None

    if concept_human_col:
        concept_df = df.dropna(subset=[concept_human_col])
        concept_df = concept_df[concept_df[concept_human_col].astype(str).str.strip() != ""]

        if len(concept_df) > 0:
            per_concept_metrics = []
            for concept in CLINICAL_CONCEPTS:
                human_has = concept_df[concept_human_col].astype(str).str.contains(
                    concept, na=False, regex=False
                )
                llm_has = concept_df["llm_concepts"].astype(str).str.contains(
                    concept, na=False, regex=False
                )

                c_tp = int((human_has & llm_has).sum())
                c_fp = int((~human_has & llm_has).sum())
                c_fn = int((human_has & ~llm_has).sum())

                c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
                c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
                c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
                c_kappa = cohens_kappa(human_has.values, llm_has.values)

                support = int(human_has.sum())
                if support > 0:  # Only report concepts with human annotations
                    per_concept_metrics.append({
                        "concept": concept,
                        "precision": round(c_prec, 4),
                        "recall": round(c_rec, 4),
                        "f1": round(c_f1, 4),
                        "kappa": c_kappa,
                        "support": support,
                    })

            if per_concept_metrics:
                macro_f1 = np.mean([m["f1"] for m in per_concept_metrics])
                macro_kappa = np.mean([m["kappa"] for m in per_concept_metrics])

                results["classification"] = {
                    "n": len(concept_df),
                    "macro_f1": round(macro_f1, 4),
                    "macro_kappa": round(macro_kappa, 4),
                    "per_concept": per_concept_metrics,
                }

                print(f"\n  CLASSIFICATION (n={len(concept_df)}):")
                print(f"    Macro F1:    {macro_f1:.3f}")
                print(f"    Macro kappa: {macro_kappa:.3f}")
                print(f"    Concepts evaluated: {len(per_concept_metrics)}")

                # Top/bottom 5 by F1
                sorted_concepts = sorted(per_concept_metrics, key=lambda x: x["f1"], reverse=True)
                print(f"    Top 5 by F1:")
                for m in sorted_concepts[:5]:
                    print(f"      {m['concept']:>30s}: F1={m['f1']:.3f} (n={m['support']})")
                if len(sorted_concepts) > 5:
                    print(f"    Bottom 5 by F1:")
                    for m in sorted_concepts[-5:]:
                        print(f"      {m['concept']:>30s}: F1={m['f1']:.3f} (n={m['support']})")

    # ── Extraction Agreement ─────────────────────────────────────────
    if "human_study_design" in df.columns:
        ext_df = df.dropna(subset=["human_study_design"])
        ext_df = ext_df[ext_df["human_study_design"].astype(str).str.strip() != ""]

        if len(ext_df) > 0 and "llm_study_design" in ext_df.columns:
            human_design = ext_df["human_study_design"].astype(str).str.strip().str.lower()
            llm_design = ext_df["llm_study_design"].astype(str).str.strip().str.lower()
            design_match = (human_design == llm_design).sum()
            design_accuracy = design_match / len(ext_df)

            extraction_metrics = {
                "n": len(ext_df),
                "study_design_accuracy": round(design_accuracy, 4),
                "study_design_ci": [round(x, 4) for x in wilson_ci(int(design_match), len(ext_df))],
            }

            # Sample size agreement (within 10%)
            if "human_sample_size" in ext_df.columns and "llm_sample_size" in ext_df.columns:
                ss_df = ext_df.dropna(subset=["human_sample_size", "llm_sample_size"])
                ss_df = ss_df[
                    (ss_df["human_sample_size"].astype(str).str.strip() != "") &
                    (ss_df["llm_sample_size"].astype(str).str.strip() != "")
                ]
                if len(ss_df) > 0:
                    try:
                        human_ss = pd.to_numeric(ss_df["human_sample_size"], errors="coerce")
                        llm_ss = pd.to_numeric(ss_df["llm_sample_size"], errors="coerce")
                        valid = human_ss.notna() & llm_ss.notna() & (human_ss > 0)
                        if valid.sum() > 0:
                            human_v = human_ss[valid].values
                            llm_v = llm_ss[valid].values
                            within_10 = np.abs(human_v - llm_v) / human_v <= 0.10
                            extraction_metrics["sample_size_n"] = int(valid.sum())
                            extraction_metrics["sample_size_within_10pct"] = round(
                                within_10.sum() / len(within_10), 4
                            )
                            extraction_metrics["sample_size_mae"] = round(
                                float(np.mean(np.abs(human_v - llm_v))), 2
                            )
                    except Exception as e:
                        logger.warning(f"Sample size comparison failed: {e}")

            results["extraction"] = extraction_metrics

            print(f"\n  EXTRACTION (n={len(ext_df)}):")
            print(f"    Study design accuracy: {design_accuracy:.3f} "
                  f"[{extraction_metrics['study_design_ci'][0]:.3f}-"
                  f"{extraction_metrics['study_design_ci'][1]:.3f}]")
            if "sample_size_within_10pct" in extraction_metrics:
                print(f"    Sample size within 10%: "
                      f"{extraction_metrics['sample_size_within_10pct']:.3f} "
                      f"(n={extraction_metrics['sample_size_n']})")
                print(f"    Sample size MAE: {extraction_metrics['sample_size_mae']}")

    # ── Inter-Rater Reliability ──────────────────────────────────────
    if "human2_relevant" in df.columns:
        irr_df = df.dropna(subset=["human_relevant", "human2_relevant"])
        irr_df = irr_df[
            (irr_df["human_relevant"].astype(str).str.strip() != "") &
            (irr_df["human2_relevant"].astype(str).str.strip() != "")
        ]
        if len(irr_df) > 0:
            h1 = irr_df["human_relevant"].astype(str).str.strip().str.lower().isin(
                ["yes", "true", "1", "include"]
            )
            h2 = irr_df["human2_relevant"].astype(str).str.strip().str.lower().isin(
                ["yes", "true", "1", "include"]
            )
            irr_kappa = cohens_kappa(h1.values, h2.values)
            results["inter_rater"] = {
                "n": len(irr_df),
                "kappa": irr_kappa,
                "interpretation": kappa_interpretation(irr_kappa),
            }
            print(f"\n  INTER-RATER RELIABILITY (n={len(irr_df)}):")
            print(f"    Cohen's k: {irr_kappa:.3f} ({kappa_interpretation(irr_kappa)})")

    # Save results
    report_path = CACHE_DIR / "human_llm_agreement.json"
    CACHE_DIR.mkdir(exist_ok=True)
    from llm_utils import save_cache as _atomic_save
    _atomic_save(results, report_path)
    print(f"\n  Agreement report saved to: {report_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 9. AUDIT SUMMARY TABLE
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
