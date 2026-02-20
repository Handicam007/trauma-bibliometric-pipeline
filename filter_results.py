#!/usr/bin/env python3
"""
Post-Search Filter & Curation
===============================
Takes the raw Scopus results and filters for journal club relevance:
1. Remove off-topic papers (materials science, unrelated specialties)
2. Prioritize trauma-specific journals
3. Create thematic categories for presentation
4. Rank by relevance score (citations + recency + journal quality)
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path(__file__).parent / "results_refined" / "all_results.csv"
OUTPUT_DIR = Path(__file__).parent / "results_curated"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# TRAUMA-RELEVANT JOURNALS (high weight)
# ============================================================
TOP_TRAUMA_JOURNALS = [
    "journal of trauma and acute care surgery",
    "injury",
    "world journal of emergency surgery",
    "trauma surgery and acute care open",
    "annals of surgery",
    "jama surgery",
    "british journal of surgery",
    "journal of the american college of surgeons",
    "surgery",
    "annals of emergency medicine",
    "academic emergency medicine",
    "emergency medicine journal",
    "critical care medicine",
    "critical care",
    "intensive care medicine",
    "shock",
    "journal of surgical research",
    "canadian journal of surgery",
    "jama",
    "new england journal of medicine",
    "lancet",
    "bmj",
    "nature reviews disease primers",
    "cochrane database of systematic reviews",
    "resuscitation",
    "transfusion",
    "european journal of trauma and emergency surgery",
    "scandinavian journal of trauma resuscitation and emergency medicine",
    "prehospital emergency care",
    "journal of neurotrauma",
    "neurosurgery",
    "journal of neurosurgery",
    "journal of surgical education",
    "american journal of surgery",
    "current opinion in critical care",
    "current problems in surgery",
    "surgical clinics of north america",
    "trauma care",
    "american surgeon",
    "journal of gastrointestinal surgery",
    "journal of emergency medicine",
    "western journal of emergency medicine",
    "canadian journal of emergency medicine",
    "cjem",
    "journal of intensive care medicine",
    "neurocritical care",
    "journal of orthopaedic trauma",
    "world neurosurgery",
    "injury epidemiology",
    "injury prevention",
]

# ============================================================
# EXCLUSION KEYWORDS (remove off-topic noise)
# ============================================================
# HOW TO EDIT: Add one keyword per line, in quotes, ending with a comma.
# IMPORTANT: A missing comma silently merges two keywords (Python quirk).
# WARNING: If you're adapting for a different field, REBUILD this list.
#   e.g., "cardiac arrest" is excluded here but is CORE to cardiology/resuscitation.
#   e.g., "chemotherapy" is excluded here but is CORE to oncology.
# After editing, run: python validate.py --dry-run
#   This shows how many papers each keyword removes.
EXCLUDE_TITLE_KEYWORDS = [
    # Materials science / chemistry noise from "TIC" abbreviation
    "photoelectron spectroscopy", "ti3c2", "ti3alc2", "mxene",
    "tribological", "coatings", "ceramics", "nanocomposite", "alloy",
    "electrochemical", "corrosion", "photocatalyst", "semiconductor",
    # Unrelated medical specialties
    "ivf", "in vitro fertilization", "assisted reproduction",
    "dental implant", "orthodontic",
    "cosmetic surgery", "breast augmentation", "rhinoplasty",
    "alzheimer", "parkinson",  # not acute trauma
    # Psychology/social (not surgical trauma)
    "childhood adversity", "adverse childhood experience",
    "post-traumatic stress disorder", "ptsd",
    "psychological trauma", "emotional trauma",
    "interpersonal violence prevention",
    "moral injury",
    "telemental health",
    # Veterinary
    "veterinary", "canine", "feline", "equine",
    # Off-topic found in v2 audit
    "arthroplasty", "joint replacement", "osteoarthritis",
    "knee replacement", "total hip replacement",
    "hepatocellular carcinoma", "tumor excision",
    "urethral stricture", "urethroplasty", "erectile dysfunction",
    "neuropsychiatric impairment",
    # Additional noise from broader search
    "bariatric", "sleeve gastrectomy", "weight loss surgery",
    "breast cancer", "prostate cancer", "colorectal cancer",
    "melanoma", "lymphoma", "chemotherapy", "radiotherapy",
    "organ transplant", "kidney transplant", "liver transplant",
    "heart transplant", "dialysis",
    "neonatal", "preterm infant", "congenital",
    "dental", "periodontal", "endodontic",
    "dermatol", "psoriasis", "eczema",
    "ophthalmol", "cataract", "glaucoma", "retinal",
    "infertility", "prenatal", "obstetric",
    "chronic obstructive", "asthma", "copd",
    "diabetes mellitus", "insulin resistance",
    "multiple sclerosis", "amyotrophic lateral",
    "cardiac arrest",  # not trauma-specific
]

EXCLUDE_ABSTRACT_KEYWORDS = [
    "photoelectron", "mxene", "nanoparticle", "catalyst",
    "semiconductor", "electrode",
]

# ============================================================
# INCLUSION: must match at least one of these in title or abstract
# ============================================================
# Papers must match AT LEAST ONE of these keywords in title or abstract.
# This is a broad safety net, NOT a topic selector.
# Aim for 50-70 terms covering all aspects of your field.
# After editing, run: python validate.py --dry-run
#   Survival rate should be 60-80%. Below 50% = list too narrow.
REQUIRE_ANY_KEYWORD = [
    # Core trauma terms
    "trauma", "injur", "hemorrha", "bleed", "resuscitat",
    "damage control", "emergency surg", "acute care surg",
    # Transfusion & blood
    "transfusion", "coagulopathy", "REBOA", "aortic occlusion",
    "whole blood", "massive transfusion", "blood product",
    "TEG", "ROTEM", "thromboelastography", "fibrinogen",
    # Procedures
    "endovascular", "thoracotomy", "laparotomy", "fasciotomy",
    "angioembolization", "embolization", "craniotomy", "craniectomy",
    "non-operative", "nonoperative", "exploratory",
    # Imaging & diagnostics
    "FAST", "eFAST", "POCUS", "point-of-care ultrasound",
    # Populations
    "geriatric", "frailty", "elderly", "pediatric",
    # Injury patterns
    "fracture", "polytrauma", "spleen", "splenic", "hepatic",
    "liver laceration", "blunt", "penetrating", "stab", "gunshot",
    "burn", "spinal cord", "spinal injur", "pelvic", "rib",
    "vascular injur", "aortic", "chest trauma", "abdominal trauma",
    "thoracic trauma", "head injur",
    # TBI / Neuro
    "traumatic brain", "TBI", "neurotrauma", "intracranial",
    "subdural", "epidural hematoma", "ICP",
    # Systems & triage
    "triage", "prehospital", "level 1", "level i", "trauma cent",
    "trauma system", "trauma team", "emergency department",
    # Training & technology
    "simulation", "surgical training", "ATLS",
    "machine learning", "artificial intelligence", "deep learning",
    # Mass casualty & military
    "mass casualty", "mass shooting", "blast", "military",
    "combat casualty", "tactical", "ECMO", "ECLS",
    # Telemedicine
    "telemedicine", "teletrauma", "telementoring",
    # Miscellaneous trauma-relevant
    "wound", "shock", "tourniquet", "hemostatic", "critical care",
    "ICU", "intensive care", "mortality", "survival",
    "emergency general surg", "acute appendicitis",
]


def load_and_filter():
    df = pd.read_csv(INPUT)
    print(f"Raw papers loaded: {len(df)}")

    # Fill NaN
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["publication"] = df["publication"].fillna("")
    df["affiliation_country"] = df["affiliation_country"].fillna("")
    df["citations_count"] = pd.to_numeric(df["citations_count"], errors="coerce").fillna(0).astype(int)

    # --- Step 1: Remove exclusions by title ---
    title_lower = df["title"].str.lower()
    abstract_lower = df["abstract"].str.lower()

    exclude_mask = pd.Series(False, index=df.index)
    for kw in EXCLUDE_TITLE_KEYWORDS:
        exclude_mask |= title_lower.str.contains(kw, case=False, na=False)
    for kw in EXCLUDE_ABSTRACT_KEYWORDS:
        exclude_mask |= abstract_lower.str.contains(kw, case=False, na=False)

    df_filtered = df[~exclude_mask].copy()
    print(f"After exclusion filter: {len(df_filtered)} (removed {exclude_mask.sum()} off-topic)")

    # --- Step 2: Require at least one trauma-relevant keyword ---
    combined = df_filtered["title"].str.lower() + " " + df_filtered["abstract"].str.lower()
    include_mask = pd.Series(False, index=df_filtered.index)
    for kw in REQUIRE_ANY_KEYWORD:
        include_mask |= combined.str.contains(kw, case=False, na=False)

    df_filtered = df_filtered[include_mask].copy()
    print(f"After inclusion filter: {len(df_filtered)}")

    # --- Step 3: Score papers ---
    from config import (
        BASE_YEAR, CITE_WEIGHT, RECENCY_WEIGHT, JOURNAL_BONUS,
        GEO_PRIMARY_BONUS, GEO_SECONDARY_BONUS,
        GEO_PRIMARY_REGEX, GEO_SECONDARY_REGEX,
    )
    pub_lower = df_filtered["publication"].str.lower()

    # Journal quality bonus (exact match to avoid substring stacking)
    journal_bonus = pd.Series(0, index=df_filtered.index)
    matched = pd.Series(False, index=df_filtered.index)
    for j in TOP_TRAUMA_JOURNALS:
        # Match journals not already matched (prevents stacking from substrings)
        is_match = pub_lower.str.contains(j, case=False, na=False) & ~matched
        journal_bonus += is_match.astype(int) * JOURNAL_BONUS
        matched |= is_match

    # Citation score (log-scaled to avoid domination by mega-cited reviews)
    cite_score = np.log1p(df_filtered["citations_count"]) * CITE_WEIGHT

    # Recency bonus
    recency = (df_filtered["year"] - BASE_YEAR) * RECENCY_WEIGHT

    # Geographic bonuses
    na_bonus = df_filtered["affiliation_country"].str.contains(
        GEO_PRIMARY_REGEX, case=False, na=False
    ).astype(int) * GEO_PRIMARY_BONUS if GEO_PRIMARY_REGEX else 0

    ca_bonus = df_filtered["affiliation_country"].str.contains(
        GEO_SECONDARY_REGEX, case=False, na=False
    ).astype(int) * GEO_SECONDARY_BONUS if GEO_SECONDARY_REGEX else 0

    df_filtered["relevance_score"] = cite_score + recency + journal_bonus + na_bonus + ca_bonus
    df_filtered = df_filtered.sort_values("relevance_score", ascending=False)

    # --- Save outputs ---
    df_filtered.to_csv(OUTPUT_DIR / "all_filtered.csv", index=False)
    print(f"\nSaved {len(df_filtered)} filtered papers to all_filtered.csv")

    # North America
    na_mask = df_filtered["affiliation_country"].str.contains(
        "United States|Canada", case=False, na=False
    )
    df_na = df_filtered[na_mask]
    df_na.to_csv(OUTPUT_DIR / "north_america.csv", index=False)
    print(f"North American: {len(df_na)}")

    # Canadian
    ca_mask = df_filtered["affiliation_country"].str.contains("Canada", case=False, na=False)
    df_ca = df_filtered[ca_mask]
    df_ca.to_csv(OUTPUT_DIR / "canadian.csv", index=False)
    print(f"Canadian: {len(df_ca)}")

    # Hot topics: 2023+, >=5 cites
    df_hot = df_filtered[(df_filtered["year"] >= 2023) & (df_filtered["citations_count"] >= 5)]
    df_hot.to_csv(OUTPUT_DIR / "hot_topics.csv", index=False)
    print(f"Hot topics (2023+, >=5 cites): {len(df_hot)}")

    # Cutting edge 2025+
    df_new = df_filtered[df_filtered["year"] >= 2025]
    df_new.to_csv(OUTPUT_DIR / "cutting_edge_2025.csv", index=False)
    print(f"Cutting edge (2025+): {len(df_new)}")

    # --- THEMATIC CATEGORIES for journal club ---
    # Uses canonical patterns from concept_definitions.py (single source of truth)
    from concept_definitions import CLINICAL_CONCEPTS, DOMAIN_GROUPS

    print(f"\n{'='*70}")
    print("THEMATIC BREAKDOWN FOR JOURNAL CLUB")
    print(f"(Using canonical patterns from concept_definitions.py)")
    print(f"{'='*70}")

    # Build theme→regex mapping from DOMAIN_GROUPS + CLINICAL_CONCEPTS
    # Each theme combines the regex patterns of its constituent concepts
    themes_regex = {}
    for group_name, concept_list in DOMAIN_GROUPS.items():
        patterns = []
        for concept in concept_list:
            if concept in CLINICAL_CONCEPTS:
                patterns.append(CLINICAL_CONCEPTS[concept])
        if patterns:
            themes_regex[group_name.upper()] = "|".join(patterns)

    # Additional themes not covered by DOMAIN_GROUPS
    if "TBI / Neurotrauma" in CLINICAL_CONCEPTS:
        themes_regex["TBI & NEUROTRAUMA"] = CLINICAL_CONCEPTS["TBI / Neurotrauma"]
    if "POCUS / eFAST" in CLINICAL_CONCEPTS:
        themes_regex["IMAGING & DIAGNOSTICS"] = CLINICAL_CONCEPTS["POCUS / eFAST"]

    # Canadian context handled separately (not regex-based)
    themes_regex["CANADIAN CONTEXT"] = None

    combined_text = df_filtered["title"].str.lower() + " " + df_filtered["abstract"].str.lower()

    for theme_name, pattern in themes_regex.items():
        if theme_name == "CANADIAN CONTEXT":
            full_theme = df_ca
            theme_df = df_ca.nlargest(10, "relevance_score")
        else:
            mask = combined_text.str.contains(pattern, regex=True, case=False, na=False)
            full_theme = df_filtered[mask]
            theme_df = full_theme.nlargest(10, "relevance_score")

        if full_theme.empty:
            continue

        theme_file = OUTPUT_DIR / f"theme_{theme_name.lower().replace(' & ', '_').replace(' ', '_')}.csv"
        full_theme.to_csv(theme_file, index=False)

        print(f"\n--- {theme_name} ({len(full_theme)} papers) ---")
        for _, row in theme_df.head(5).iterrows():
            print(f"  [{row['citations_count']:>4} cites | {row['year']}] {row['title'][:90]}")
            print(f"        {row['publication']}")

    # --- TOP 30 OVERALL for journal club ---
    print(f"\n{'='*70}")
    print("TOP 30 JOURNAL CLUB CANDIDATES")
    print("(scored by citations + recency + journal quality + NA affiliation)")
    print(f"{'='*70}")

    top30 = df_filtered.nlargest(30, "relevance_score")
    for rank, (_, row) in enumerate(top30.iterrows(), 1):
        print(f"\n{rank:>2}. {row['title']}")
        print(f"    {row['publication']} ({row['year']}) — {row['citations_count']} citations")
        if row.get("abstract") and str(row["abstract"]) != "nan":
            abstract_preview = str(row["abstract"])[:200].replace("\n", " ")
            print(f"    {abstract_preview}...")


if __name__ == "__main__":
    load_and_filter()
