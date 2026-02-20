"""
Hierarchical Concept Classification
=====================================
Two-stage "tournament" classification to avoid LLM position bias:

  Stage 1: Assign paper to 1-N of 6 broad domains (6 choices)
  Stage 2: Assign specific concepts WITHIN each domain (~4-10 choices)

This reduces each call from 41 choices to ~6-10 choices, eliminating
the middle-of-list blindness that degrades LLM classification accuracy.

Output: results_curated/llm_concepts.csv
Cache:  llm_cache/classification_progress.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from concept_definitions import CLINICAL_CONCEPTS, DOMAIN_GROUPS
from config import FIELD_NAME
from llm_providers import LLMProvider
from llm_schemas import ConceptClassification, DomainClassification, DOMAIN_NAMES

CACHE_DIR = Path(__file__).parent / "llm_cache"
CACHE_FILE = CACHE_DIR / "classification_progress.json"
OUTPUT_FILE = Path(__file__).parent / "results_curated" / "llm_concepts.csv"


# ── Stage 1: Domain classification prompt ─────────────────────────────

DOMAIN_SYSTEM_PROMPT = f"""You are classifying {FIELD_NAME} papers into broad research domains.
Given a paper's title and abstract, classify it into ALL applicable domains.

DOMAINS:
1. Resuscitation & Blood Products — hemorrhage control, transfusion, coagulopathy, damage control resuscitation, blood products, TEG/ROTEM, fibrinogen
2. Surgical Techniques & Approaches — REBOA, thoracotomy, laparotomy, angioembolization, non-operative management, rib fixation, specific surgical procedures
3. Technology & Innovation — AI/machine learning, POCUS/eFAST, simulation/training, telemedicine, ECMO
4. Populations & Systems — geriatric/elderly, pediatric, trauma systems, prehospital/EMS, triage, military/combat, mass casualty
5. Injury Patterns & Ortho — fractures, orthopaedic trauma, pelvic injuries, TBI/neurotrauma, spinal cord, blunt/penetrating, organ injuries (spleen, liver), polytrauma
6. Other Emerging Topics — COVID-19 impact, VTE prevention, airway management, firearm/gun violence

RULES:
- A paper may belong to MULTIPLE domains
- Select ALL that apply based on the paper's actual content
- If the abstract is missing, use the title only (lower confidence)

Respond ONLY with valid JSON matching the required schema."""


def _build_stage2_prompt(domains: list[str]) -> str:
    """Build the Stage 2 concept classification prompt for given domains."""
    # Collect concepts from the assigned domains
    concepts_for_domains = {}
    for domain in domains:
        if domain in DOMAIN_GROUPS:
            concepts_for_domains[domain] = DOMAIN_GROUPS[domain]

    if not concepts_for_domains:
        # Fallback: use all concepts
        all_concepts = list(CLINICAL_CONCEPTS.keys())
        return _build_concept_prompt(all_concepts)

    # Build per-domain concept list
    all_relevant_concepts = []
    sections = []
    for domain, concepts in concepts_for_domains.items():
        valid = [c for c in concepts if c in CLINICAL_CONCEPTS]
        all_relevant_concepts.extend(valid)
        if valid:
            concept_list = "\n".join(f"  - {c}" for c in valid)
            sections.append(f"[{domain}]\n{concept_list}")

    return _build_concept_prompt(all_relevant_concepts, sections)


def _build_concept_prompt(concepts: list[str], sections: list[str] = None) -> str:
    """Build the concept classification system prompt."""
    if sections:
        concept_text = "\n\n".join(sections)
    else:
        concept_text = "\n".join(f"  - {c}" for c in concepts)

    return f"""You are classifying {FIELD_NAME} papers into specific clinical concepts.
Given a paper's title and abstract, identify ALL applicable concepts from the list below.

AVAILABLE CONCEPTS:
{concept_text}

RULES:
- A paper may match 0, 1, or MULTIPLE concepts
- Only assign concepts that are CLEARLY ADDRESSED in the paper (not merely mentioned in passing)
- primary_concept should be the SINGLE most central concept, or null if none clearly dominates
- If NONE of the listed concepts apply, return an empty concepts list
- concepts must use EXACT names from the list above

Respond ONLY with valid JSON matching the required schema."""


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache: dict):
    CACHE_DIR.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def run_classification(
    df: pd.DataFrame,
    llm: LLMProvider,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Hierarchical concept classification pipeline.

    Args:
        df: DataFrame with 'doi', 'title', 'abstract' columns
        llm: Configured LLMProvider instance
        limit: Max papers to process (for testing)

    Returns:
        DataFrame with classification results
    """
    print(f"\n{'─' * 70}")
    print("HIERARCHICAL CONCEPT CLASSIFICATION")
    print(f"{'─' * 70}")
    print(f"  Stage 1: {len(DOMAIN_NAMES)} domains")
    print(f"  Stage 2: {len(CLINICAL_CONCEPTS)} concepts across domains")

    cache = load_cache()
    papers = df.copy()
    if limit:
        papers = papers.head(limit)

    total = len(papers)
    already_done = sum(1 for doi in papers["doi"] if str(doi) in cache)
    print(f"  Total papers: {total:,}")
    print(f"  Already classified (cached): {already_done:,}")
    print(f"  Remaining: {total - already_done:,}")

    results = []
    processed = 0

    for i, (idx, row) in enumerate(papers.iterrows()):
        doi = str(row.get("doi", ""))
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))

        # Check cache
        if doi in cache:
            results.append({**cache[doi], "doi": doi})
            continue

        abs_text = abstract.strip() if abstract and len(abstract) > 10 else "(No abstract available)"
        user_prompt = f'TITLE: "{title}"\nABSTRACT: "{abs_text}"'

        try:
            # ── Stage 1: Domain classification ────────────────────────
            domain_result = llm.query(
                system=DOMAIN_SYSTEM_PROMPT,
                user=user_prompt,
                schema=DomainClassification,
            )

            # ── Stage 2: Concept classification within domains ────────
            stage2_system = _build_stage2_prompt(domain_result.domains)
            concept_result = llm.query(
                system=stage2_system,
                user=user_prompt,
                schema=ConceptClassification,
            )

            # Validate concept names against known concepts
            valid_concepts = [c for c in concept_result.concepts if c in CLINICAL_CONCEPTS]

            entry = {
                "doi": doi,
                "domains": "|".join(domain_result.domains),
                "concepts": "|".join(valid_concepts),
                "primary_concept": concept_result.primary_concept
                    if concept_result.primary_concept in CLINICAL_CONCEPTS else None,
                "n_concepts": len(valid_concepts),
                "domain_confidence": domain_result.confidence,
                "concept_confidence": concept_result.confidence,
            }
            results.append(entry)

            # Cache (without doi)
            cache_entry = {k: v for k, v in entry.items() if k != "doi"}
            cache[doi] = cache_entry
            processed += 1

        except Exception as e:
            print(f"  ⚠ Error on paper {doi}: {e}")
            results.append({
                "doi": doi,
                "domains": "",
                "concepts": "",
                "primary_concept": None,
                "n_concepts": 0,
                "domain_confidence": 0.0,
                "concept_confidence": 0.0,
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
    n_with_concepts = (results_df["n_concepts"] > 0).sum()
    avg_concepts = results_df["n_concepts"].mean()
    print(f"\n  Results:")
    print(f"    Papers with ≥1 concept: {n_with_concepts:,} ({n_with_concepts/total*100:.1f}%)")
    print(f"    Average concepts/paper: {avg_concepts:.2f}")

    # Top concepts
    all_concepts = []
    for concepts_str in results_df["concepts"].fillna(""):
        if concepts_str:
            all_concepts.extend(concepts_str.split("|"))
    if all_concepts:
        from collections import Counter
        top10 = Counter(all_concepts).most_common(10)
        print(f"\n  Top 10 LLM-detected concepts:")
        for concept, count in top10:
            print(f"    {count:>4} — {concept}")

    print(f"  Saved to: {OUTPUT_FILE}")

    return results_df
