"""
LLM Relevance Screening
========================
Determines if each paper is relevant to the field defined in config.py.
Uses few-shot prompting with confidence scoring for human-in-the-loop.

Output: results_curated/llm_screening_results.csv
Cache:  llm_cache/screening_progress.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    FIELD_NAME, LLM_CONFIDENCE_AUTO, LLM_CONFIDENCE_REVIEW,
)
from llm_providers import LLMProvider
from llm_schemas import ScreeningResult

CACHE_DIR = Path(__file__).parent / "llm_cache"
CACHE_FILE = CACHE_DIR / "screening_progress.json"
OUTPUT_FILE = Path(__file__).parent / "results_curated" / "llm_screening_results.csv"

# ── System Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a literature screener for a bibliometric analysis of {FIELD_NAME}.
Your task: determine if this paper is RELEVANT to the field.

INCLUDE criteria (any one is sufficient):
- Surgical management of traumatic injuries (blunt, penetrating, burns)
- Resuscitation, hemorrhage control, damage control surgery
- Trauma systems, triage, prehospital care, EMS
- Emergency general surgery for acute conditions
- Critical care of trauma patients (ICU, ventilation, complications)
- Trauma-specific imaging, procedures, devices (REBOA, FAST, etc.)
- Epidemiology of traumatic injury
- Trauma education and simulation
- Novel blood products or transfusion strategies for trauma
- Neurotrauma / traumatic brain injury management
- Orthopaedic trauma (fractures from acute injury)

EXCLUDE criteria (any one is sufficient):
- Psychological trauma only (PTSD, adverse childhood experiences, mental health)
- Veterinary or animal studies
- Materials science / chemistry (even if "TIC" or "trauma" appears)
- Chronic disease management without acute injury context
- Elective surgery (cosmetic, bariatric, transplant, joint replacement)
- Oncology (cancer surgery, chemotherapy, radiotherapy)
- Non-English papers with no English abstract

RULES:
- If UNCERTAIN, mark relevant=true with lower confidence (0.5-0.7)
- confidence must reflect YOUR certainty, not the paper's quality
- reason must be ≤200 characters
- If the abstract is missing/empty, base decision on title only (lower confidence)

Respond ONLY with valid JSON matching the required schema."""

# ── Few-shot examples ─────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "input": 'TITLE: "Whole blood resuscitation in pediatric trauma: A multicenter study"\nABSTRACT: "Background: Pediatric trauma patients have unique resuscitation needs. We conducted a multicenter trial comparing whole blood to component therapy in 142 children."',
        "output": '{"relevant": true, "confidence": 0.98, "reason": "Pediatric trauma resuscitation study - core topic", "exclusion_category": null}',
    },
    {
        "input": 'TITLE: "Post-traumatic stress disorder following motor vehicle collisions"\nABSTRACT: "This study examines PTSD prevalence among crash survivors using validated screening instruments."',
        "output": '{"relevant": false, "confidence": 0.95, "reason": "Psychological trauma (PTSD), not surgical management", "exclusion_category": "psychology_ptsd"}',
    },
    {
        "input": 'TITLE: "Ti3C2 MXene nanocomposite for electrochemical sensing"\nABSTRACT: "We synthesized Ti3C2Tx MXene with improved electrical conductivity for biosensor applications."',
        "output": '{"relevant": false, "confidence": 0.99, "reason": "Materials science paper, no medical content", "exclusion_category": "materials_science"}',
    },
    {
        "input": 'TITLE: "Emergency laparotomy outcomes in elderly patients with blunt abdominal trauma"\nABSTRACT: "Objective: To evaluate outcomes of emergency laparotomy in geriatric blunt trauma patients at a Level I center."',
        "output": '{"relevant": true, "confidence": 0.99, "reason": "Geriatric blunt trauma surgical outcomes - core topic", "exclusion_category": null}',
    },
]


def build_user_prompt(title: str, abstract: str) -> str:
    """Format a single paper for the screening prompt."""
    abs_text = abstract.strip() if abstract and len(str(abstract)) > 10 else "(No abstract available)"
    return f'TITLE: "{title}"\nABSTRACT: "{abs_text}"'


def build_full_system_prompt() -> str:
    """Combine system prompt with few-shot examples."""
    parts = [SYSTEM_PROMPT, "\n=== EXAMPLES ===\n"]
    for ex in FEW_SHOT_EXAMPLES:
        parts.append(f"INPUT:\n{ex['input']}\n\nOUTPUT:\n{ex['output']}\n")
    parts.append("=== NOW SCREEN THIS PAPER ===")
    return "\n".join(parts)


def load_cache() -> dict:
    """Load screening progress from cache."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache: dict):
    """Save screening progress to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def run_screening(
    df: pd.DataFrame,
    llm: LLMProvider,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Screen all papers for relevance.

    Args:
        df: DataFrame with 'doi', 'title', 'abstract' columns
        llm: Configured LLMProvider instance
        limit: Max papers to process (for testing)

    Returns:
        DataFrame with screening results
    """
    print(f"\n{'─' * 70}")
    print("LLM RELEVANCE SCREENING")
    print(f"{'─' * 70}")

    system = build_full_system_prompt()
    cache = load_cache()

    papers = df.copy()
    if limit:
        papers = papers.head(limit)

    total = len(papers)
    already_done = sum(1 for doi in papers["doi"] if str(doi) in cache)
    print(f"  Total papers: {total:,}")
    print(f"  Already screened (cached): {already_done:,}")
    print(f"  Remaining: {total - already_done:,}")

    results = []
    processed = 0

    for i, (idx, row) in enumerate(papers.iterrows()):
        doi = str(row.get("doi", ""))
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))

        # Check cache
        if doi in cache:
            results.append({**cache[doi], "doi": doi, "title": title})
            continue

        # Query LLM
        user_prompt = build_user_prompt(title, abstract)
        try:
            result = llm.query(system=system, user=user_prompt, schema=ScreeningResult)
            entry = {
                "doi": doi,
                "title": title,
                "relevant": result.relevant,
                "confidence": result.confidence,
                "reason": result.reason,
                "exclusion_category": result.exclusion_category,
                "needs_review": result.confidence < LLM_CONFIDENCE_AUTO,
            }
            results.append(entry)

            # Cache (without title to save space)
            cache[doi] = {
                "relevant": result.relevant,
                "confidence": result.confidence,
                "reason": result.reason,
                "exclusion_category": result.exclusion_category,
                "needs_review": result.confidence < LLM_CONFIDENCE_AUTO,
            }
            processed += 1

        except Exception as e:
            print(f"  ⚠ Error on paper {doi}: {e}")
            results.append({
                "doi": doi,
                "title": title,
                "relevant": None,
                "confidence": 0.0,
                "reason": f"Error: {str(e)[:100]}",
                "exclusion_category": None,
                "needs_review": True,
            })

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"  Progress: {i+1:,}/{total:,} ({pct:.1f}%) "
                  f"| Processed this run: {processed:,}")
            save_cache(cache)

    save_cache(cache)

    # Build results DataFrame
    results_df = pd.DataFrame(results)

    # Save output
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    n_relevant = results_df["relevant"].sum() if "relevant" in results_df else 0
    n_excluded = (~results_df["relevant"].fillna(False)).sum()
    n_review = results_df["needs_review"].sum() if "needs_review" in results_df else 0

    print(f"\n  Results:")
    print(f"    Relevant:     {n_relevant:,}")
    print(f"    Excluded:     {n_excluded:,}")
    print(f"    Needs review: {n_review:,} (confidence < {LLM_CONFIDENCE_AUTO})")
    print(f"  Saved to: {OUTPUT_FILE}")

    return results_df
