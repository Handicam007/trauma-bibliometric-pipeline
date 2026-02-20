"""
Pydantic response schemas for all LLM tasks.
=============================================
Strict type enforcement ensures LLM outputs are validated automatically.
If the LLM returns malformed JSON (e.g., "sampleCount" instead of
"sample_size"), Pydantic raises a ValidationError and the provider
retries the call.

All schemas include a `confidence` field (0.0-1.0) for human-in-the-loop
thresholds defined in config.py.
"""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
# SCREENING
# ═══════════════════════════════════════════════════════════════════════

class ScreeningResult(BaseModel):
    """LLM relevance screening output."""
    relevant: bool
    confidence: float = Field(ge=0.0, le=1.0,
        description="Model confidence in this decision (0=uncertain, 1=certain)")
    reason: str = Field(max_length=200,
        description="Brief explanation for the decision")
    exclusion_category: Optional[Literal[
        "materials_science",
        "psychology_ptsd",
        "veterinary",
        "chronic_disease",
        "elective_surgery",
        "unrelated_specialty",
        "non_english",
        "non_human",
    ]] = Field(None,
        description="If excluded, the reason category. Null if relevant.")


# ═══════════════════════════════════════════════════════════════════════
# HIERARCHICAL CLASSIFICATION — Stage 1 (Domain)
# ═══════════════════════════════════════════════════════════════════════

DOMAIN_NAMES = [
    "Resuscitation & Blood Products",
    "Surgical Techniques & Approaches",
    "Technology & Innovation",
    "Populations & Systems",
    "Injury Patterns & Ortho",
    "Other Emerging Topics",
]

DomainLiteral = Literal[
    "Resuscitation & Blood Products",
    "Surgical Techniques & Approaches",
    "Technology & Innovation",
    "Populations & Systems",
    "Injury Patterns & Ortho",
    "Other Emerging Topics",
]


class DomainClassification(BaseModel):
    """Level 1: Broad domain assignment (6 domains)."""
    domains: List[DomainLiteral] = Field(
        description="All applicable domains for this paper")
    confidence: float = Field(ge=0.0, le=1.0,
        description="Confidence in domain assignment")


# ═══════════════════════════════════════════════════════════════════════
# HIERARCHICAL CLASSIFICATION — Stage 2 (Concept)
# ═══════════════════════════════════════════════════════════════════════

class ConceptClassification(BaseModel):
    """Level 2: Specific concepts within assigned domain(s)."""
    concepts: List[str] = Field(
        description="All applicable clinical concepts from the provided list")
    primary_concept: Optional[str] = Field(None,
        description="The single most relevant concept, if any")
    confidence: float = Field(ge=0.0, le=1.0,
        description="Confidence in concept assignment")


# ═══════════════════════════════════════════════════════════════════════
# STRUCTURED DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

class StudyDesign(str, Enum):
    """Enumeration of study designs recognized by the extraction pipeline."""
    RCT = "RCT"
    PROSPECTIVE_OBSERVATIONAL = "prospective_observational"
    RETROSPECTIVE_COHORT = "retrospective_cohort"
    CASE_CONTROL = "case_control"
    CROSS_SECTIONAL = "cross_sectional"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"
    NARRATIVE_REVIEW = "narrative_review"
    GUIDELINE = "guideline"
    REGISTRY_STUDY = "registry_study"
    EXPERIMENTAL = "experimental"
    QUALITATIVE = "qualitative"
    OTHER = "other"


class ExtractionResult(BaseModel):
    """Structured data extracted from a paper's title + abstract."""
    study_design: StudyDesign = Field(
        description="Study design as described by the authors")
    level_of_evidence: Literal["I", "II", "III", "IV", "V", "unclear"] = Field(
        description="Oxford/GRADE level of evidence. Use 'unclear' if ambiguous.")
    sample_size: Optional[int] = Field(None,
        description="Number of patients/subjects. Null if not stated in abstract.")
    population: Literal[
        "adult", "pediatric", "geriatric", "military", "mixed", "unclear"
    ] = Field(description="Primary study population")
    setting: Literal[
        "level_1_trauma_center", "level_2_trauma_center",
        "community", "prehospital", "military",
        "multi_center", "registry", "unclear"
    ] = Field(description="Clinical setting")
    intervention_type: Optional[Literal[
        "surgical", "resuscitation", "diagnostic", "system",
        "training", "pharmacological", "technology", "observational"
    ]] = Field(None,
        description="Type of intervention studied. Null for observational/review papers.")
    key_finding: Optional[str] = Field(None, max_length=300,
        description="One-sentence paraphrase of the main result. Null if no conclusion stated.")
    mortality_reported: bool = Field(
        description="True only if death/mortality/survival is an outcome measure (not just background)")
    confidence: float = Field(ge=0.0, le=1.0,
        description="Confidence in extraction accuracy")


# ═══════════════════════════════════════════════════════════════════════
# UTILITY: Schema → JSON dict for LLM providers
# ═══════════════════════════════════════════════════════════════════════

def schema_to_json_schema(model_class: type[BaseModel]) -> dict:
    """Convert a Pydantic model to a JSON Schema dict for LLM structured output."""
    return model_class.model_json_schema()


def schema_to_openai_format(model_class: type[BaseModel]) -> dict:
    """Convert Pydantic model to OpenAI's response_format parameter."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_class.__name__,
            "schema": model_class.model_json_schema(),
            "strict": True,
        },
    }
