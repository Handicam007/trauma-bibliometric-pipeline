"""
FIELD-SPECIFIC CONFIGURATION — ORTHOPEDIC SURGERY
====================================================
Copy this file to the project root as config.py to switch the pipeline
from Trauma Acute Care Surgery to Orthopedic Surgery.

    cp configs/ortho_config.py config.py
    cp configs/ortho_concepts.py concept_definitions.py
    # Then update llm_schemas.py CONCEPT_NAMES + ConceptLiteral to match

See FIELD_GUIDE.md for the full adaptation guide.
"""

# ── FIELD IDENTITY ────────────────────────────────────────────────
FIELD_NAME = "Orthopedic Surgery"
FIELD_SHORT = "Ortho"                        # For figure titles

# ── TIME WINDOW ───────────────────────────────────────────────────
YEAR_MIN = 2018
YEAR_MAX = 2026
BASE_YEAR = YEAR_MIN - 1                     # Recency score: (year - BASE_YEAR) * weight

# ── LAST COMPLETE YEAR ───────────────────────────────────────────
YEAR_STATS_MAX = 2025                        # Last complete year for inference

# Trending split: "early" vs "late" periods (statistical tests only)
_mid = YEAR_MIN + (YEAR_STATS_MAX - YEAR_MIN) // 2 + 1  # 2018 + 3 + 1 = 2022
TREND_EARLY = (YEAR_MIN, _mid - 1)             # (2018, 2021)
TREND_LATE  = (_mid, YEAR_STATS_MAX)            # (2022, 2025)

# ── GEOGRAPHY ─────────────────────────────────────────────────────
# Orthopedic surgery is a global field — no geographic filter by default.
# Uncomment one of these to restrict:
#   GEO_FILTER_SCOPUS = ' AND (AFFILCOUNTRY(United States) OR AFFILCOUNTRY(Canada))'
#   GEO_FILTER_SCOPUS = ' AND AFFILCOUNTRY(United Kingdom)'
GEO_FILTER_SCOPUS = None                     # Global (no filter)

GEO_HIGHLIGHT_COUNTRIES = ["United States", "United Kingdom"]
GEO_LABEL = "US + UK"                        # Short label for figure titles

GEO_PRIMARY_REGEX = "United States|United Kingdom"
GEO_SECONDARY_REGEX = "United Kingdom"
GEO_PRIMARY_LABEL = "US + UK"
GEO_SECONDARY_LABEL = "UK"

# ── WORD CLOUD ────────────────────────────────────────────────────
WORDCLOUD_STOPWORDS = [
    "orthopedic", "orthopaedic", "surgery", "surgical", "patient",
    "patients", "study", "clinical", "outcome", "outcomes", "results",
    "analysis", "retrospective", "treatment", "fracture", "bone",
    "joint", "operative", "postoperative", "associated",
]

# ── JOURNAL HIGHLIGHTING ──────────────────────────────────────────
JOURNAL_HIGHLIGHT_KEYWORDS = [
    "orthop", "orthopaed", "bone", "joint", "musculoskeletal",
    "arthroplasty", "arthroscopy", "spine", "fracture",
]
JOURNAL_HIGHLIGHT_LABEL = "Orthopedic-focused journals"
JOURNAL_OTHER_LABEL = "Other journals"

# ── TRENDING ANALYSIS ─────────────────────────────────────────────
TRENDING_MIN_PAPERS = 8
TRENDING_NEW_TOPIC_CAP = 200

# ── FIGURE OUTPUT ─────────────────────────────────────────────────
FIG_DIR_NAME = "figures_ortho"

# ── RELEVANCE SCORING WEIGHTS ─────────────────────────────────────
CITE_WEIGHT = 10
RECENCY_WEIGHT = 5
JOURNAL_BONUS = 30
GEO_PRIMARY_BONUS = 10
GEO_SECONDARY_BONUS = 10

# ── LLM SETTINGS ────────────────────────────────────────────────────
LLM_PROVIDER = "openai"
LLM_MODEL = None                  # None = provider default
LLM_TEMPERATURE = 0.0             # Deterministic for reproducibility
LLM_BATCH_SIZE = 50
LLM_MAX_RETRIES = 3
LLM_CACHE_DIR = "llm_cache"

# Human-in-the-loop confidence thresholds
LLM_CONFIDENCE_AUTO = 0.8
LLM_CONFIDENCE_REVIEW = 0.5

# ── BENCHMARK / PUBLICATION SETTINGS ────────────────────────────────
BENCHMARK_SAMPLE_SIZE = 200
VALIDATION_SAMPLE_SIZE = 200
CONSENSUS_RUNS = 3
CONSENSUS_TEMPERATURES = (0.0, 0.0, 0.3)
