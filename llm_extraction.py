"""
Structured Data Extraction — Golden Prompt
============================================
Extracts study design, level of evidence, sample size, population,
setting, intervention type, key finding, and mortality reporting
from each paper's title and abstract.

Uses anti-hallucination "Golden Prompt" with:
- Explicit null-handling rules
- Common trap warnings
- Few-shot examples with null values
- Pydantic schema enforcement
- Post-processing guards for implausible values

Output: results_curated/llm_extracted_data.csv
Cache:  llm_cache/extraction_progress.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import FIELD_NAME, LLM_BATCH_SIZE
from llm_providers import LLMProvider
from llm_schemas import ExtractionResult
from llm_utils import (
    safe_doi_key, load_cache, save_cache, should_report_progress,
)

logger = logging.getLogger("llm_pipeline.extraction")

CACHE_FILE = Path(__file__).parent / "llm_cache" / "extraction_progress.json"
OUTPUT_FILE = Path(__file__).parent / "results_curated" / "llm_extracted_data.csv"

# Maximum plausible sample size — anything above this is likely hallucinated
MAX_PLAUSIBLE_SAMPLE_SIZE = 5_000_000

# ── Golden Prompt ─────────────────────────────────────────────────────

GOLDEN_PROMPT = f"""You are a medical research data extractor for a bibliometric analysis of {FIELD_NAME}.
Extract structured information from this paper's title and abstract.

CRITICAL RULES — READ BEFORE RESPONDING:
1. ONLY extract information EXPLICITLY STATED in the title/abstract.
2. If a field is NOT mentioned or cannot be determined, use null (not a guess).
3. NEVER infer sample_size from percentages, subgroup numbers, or context.
4. NEVER guess level_of_evidence — if the study design is ambiguous, use "unclear".
5. key_finding must be a DIRECT PARAPHRASE of the abstract's conclusion, not your interpretation. If no conclusion is stated, use null.
6. For study_design: match the design AS DESCRIBED by the authors.
   "Retrospective review" or "chart review" = retrospective_cohort
   "We randomized patients" = RCT
   "Prospective cohort" or "prospective study" = prospective_observational
   "Systematic review and meta-analysis" = meta_analysis
   "Systematic review" (without meta-analysis) = systematic_review
   "Case series" or "series of N patients" = case_series
   "Case report" = case_report
   "National registry" or "NTDB" or "TQIP" = registry_study
   "Guideline" or "practice management guideline" = guideline
   If unclear, use "other"
7. mortality_reported = true ONLY if the abstract mentions death/mortality/survival as an OUTCOME MEASURE, NOT just as background context or introduction.
8. population: use "mixed" if both adults and children, "geriatric" if specifically elderly/frail, "military" if combat/tactical.
9. setting: "multi_center" if >=2 centers stated, "registry" if using NTDB/TQIP/other registry.

COMMON TRAPS — DO NOT FALL FOR THESE:
X Abstract says "outcomes of 150 patients" -> sample_size = 150
X Abstract says "30-day mortality was 12%" as an outcome -> mortality_reported = true
X Abstract says "mortality rates have increased" in introduction -> mortality_reported = false (background, not outcome)
X Abstract says "we reviewed the literature" -> study_design = narrative_review
X Abstract says "data from the NTDB" -> setting = registry, study_design = registry_study
X Abstract mentions no patient numbers -> sample_size = null (NOT 0)
X Abstract is a guideline -> level_of_evidence = "unclear" (guidelines themselves don't have LoE)
X Abstract says "n=50 in each group" -> sample_size = 100 (total, not per-group)

=== EXAMPLES ===

EXAMPLE 1 (RCT with clear results):
TITLE: "Whole blood vs component therapy in pediatric trauma: A randomized trial"
ABSTRACT: "Background: Current resuscitation protocols vary. Methods: We randomized 142 pediatric trauma patients (age 2-17) at 3 Level I trauma centers to receive either whole blood or standard component therapy. Primary outcome: 24-hour mortality. Results: Mortality was 8.5% vs 14.1% (p=0.04). Conclusion: Whole blood reduces 24-hour mortality in pediatric trauma."
-> {{"study_design": "RCT", "level_of_evidence": "I", "sample_size": 142, "population": "pediatric", "setting": "level_1_trauma_center", "intervention_type": "resuscitation", "key_finding": "Whole blood reduced 24-hour mortality compared to component therapy in pediatric trauma (8.5% vs 14.1%, p=0.04)", "mortality_reported": true, "confidence": 0.95}}

EXAMPLE 2 (Review with no numbers):
TITLE: "Trends in REBOA use: A narrative review"
ABSTRACT: "Resuscitative endovascular balloon occlusion of the aorta (REBOA) has gained increasing attention as a bridge to definitive hemorrhage control. This review summarizes current evidence and ongoing controversies."
-> {{"study_design": "narrative_review", "level_of_evidence": "V", "sample_size": null, "population": "unclear", "setting": "unclear", "intervention_type": null, "key_finding": null, "mortality_reported": false, "confidence": 0.90}}

EXAMPLE 3 (Registry/ML study):
TITLE: "Machine learning prediction of hemorrhagic shock using NTDB data"
ABSTRACT: "We developed a gradient boosting model using vital signs from 2,847 trauma patients in the National Trauma Data Bank to predict hemorrhagic shock within 1 hour of arrival. AUC was 0.89 on the held-out test set."
-> {{"study_design": "registry_study", "level_of_evidence": "III", "sample_size": 2847, "population": "adult", "setting": "registry", "intervention_type": "technology", "key_finding": "Gradient boosting model predicted hemorrhagic shock with AUC 0.89 using NTDB vital signs data", "mortality_reported": false, "confidence": 0.92}}

EXAMPLE 4 (No abstract available):
TITLE: "Damage control surgery for the acute care surgeon"
ABSTRACT: "(No abstract available)"
-> {{"study_design": "other", "level_of_evidence": "unclear", "sample_size": null, "population": "unclear", "setting": "unclear", "intervention_type": null, "key_finding": null, "mortality_reported": false, "confidence": 0.30}}

=== NOW EXTRACT DATA FROM THIS PAPER ===

Respond ONLY with valid JSON matching the required schema."""


def build_user_prompt(title: str, abstract: str) -> str:
    """Format a single paper for extraction."""
    abs_text = abstract.strip() if abstract and len(str(abstract)) > 10 else "(No abstract available)"
    return f'TITLE: "{title}"\nABSTRACT: "{abs_text}"'


def run_extraction(
    df: pd.DataFrame,
    llm: LLMProvider,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract structured data from all papers.

    Args:
        df: DataFrame with 'doi', 'title', 'abstract' columns
        llm: Configured LLMProvider instance
        limit: Max papers to process (for testing)

    Returns:
        DataFrame with extraction results
    """
    print(f"\n{'─' * 70}")
    print("STRUCTURED DATA EXTRACTION (Golden Prompt)")
    print(f"{'─' * 70}")

    cache = load_cache(CACHE_FILE)
    papers = df.head(limit) if limit else df

    total = len(papers)
    already_done = sum(
        1 for _, row in papers.iterrows()
        if safe_doi_key(row.get("doi"), row.get("title", "")) in cache
    )
    print(f"  Total papers: {total:,}")
    print(f"  Already extracted (cached): {already_done:,}")
    print(f"  Remaining: {total - already_done:,}")

    results = []
    processed = 0

    for i, (idx, row) in enumerate(papers.iterrows()):
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        doi_str = str(row.get("doi", ""))
        cache_key = safe_doi_key(row.get("doi"), title)

        # Check cache
        if cache_key in cache:
            results.append({**cache[cache_key], "doi": doi_str})
            continue

        user_prompt = build_user_prompt(title, abstract)

        try:
            result = llm.query(
                system=GOLDEN_PROMPT,
                user=user_prompt,
                schema=ExtractionResult,
            )

            # Post-processing guards
            sample_size = result.sample_size
            if sample_size is not None:
                if sample_size <= 0:
                    sample_size = None
                elif sample_size > MAX_PLAUSIBLE_SAMPLE_SIZE:
                    logger.warning(
                        f"Implausible sample_size={sample_size} for {cache_key}, setting to null"
                    )
                    sample_size = None

            entry = {
                "doi": doi_str,
                "study_design": result.study_design.value,
                "level_of_evidence": result.level_of_evidence,
                "sample_size": sample_size,
                "population": result.population,
                "setting": result.setting,
                "intervention_type": result.intervention_type,
                "key_finding": result.key_finding,
                "mortality_reported": result.mortality_reported,
                "confidence": result.confidence,
            }
            results.append(entry)

            # Cache
            cache_entry = {k: v for k, v in entry.items() if k != "doi"}
            cache[cache_key] = cache_entry
            processed += 1

        except Exception as e:
            logger.warning(f"Error on paper {cache_key}: {e}")
            results.append({
                "doi": doi_str,
                "study_design": "other",
                "level_of_evidence": "unclear",
                "sample_size": None,
                "population": "unclear",
                "setting": "unclear",
                "intervention_type": None,
                "key_finding": None,
                "mortality_reported": False,
                "confidence": 0.0,
            })

        # Progress
        if should_report_progress(i, total, LLM_BATCH_SIZE):
            pct = (i + 1) / total * 100
            print(f"  Progress: {i+1:,}/{total:,} ({pct:.1f}%) "
                  f"| Processed this run: {processed:,}")
            save_cache(cache, CACHE_FILE)

    save_cache(cache, CACHE_FILE)

    # Build results DataFrame
    results_df = pd.DataFrame(results)

    # Save output
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print(f"\n  Results:")
    if "study_design" in results_df:
        design_counts = results_df["study_design"].value_counts()
        print(f"\n  Study design distribution:")
        for design, count in design_counts.head(8).items():
            print(f"    {count:>4} — {design}")

    if "sample_size" in results_df:
        has_n = results_df["sample_size"].notna().sum()
        median_n = results_df["sample_size"].dropna().median()
        print(f"\n  Sample sizes:")
        print(f"    Papers with sample size: {has_n:,}")
        print(f"    Median sample size: {median_n:.0f}" if has_n > 0 else "    N/A")

    if "mortality_reported" in results_df:
        mort = results_df["mortality_reported"].sum()
        print(f"\n  Mortality reported as outcome: {mort:,} ({mort/total*100:.1f}%)")

    if "level_of_evidence" in results_df:
        loe = results_df["level_of_evidence"].value_counts()
        print(f"\n  Level of evidence:")
        for level, count in sorted(loe.items()):
            print(f"    Level {level}: {count:,}")

    print(f"\n  Saved to: {OUTPUT_FILE}")

    return results_df
