"""
FIELD-SPECIFIC CONFIGURATION
==============================
This is the ONLY file you need to edit when adapting this pipeline to a new field.
Every other script imports from here. No hardcoded values elsewhere.

Current field: Trauma Acute Care Surgery (US + Canada, 2020-2026)
"""

# ── FIELD IDENTITY ────────────────────────────────────────────────
FIELD_NAME = "Trauma Acute Care Surgery"
FIELD_SHORT = "Trauma"                       # For figure titles

# ── TIME WINDOW ───────────────────────────────────────────────────
YEAR_MIN = 2020
YEAR_MAX = 2026
BASE_YEAR = YEAR_MIN - 1                     # Recency score: (year - BASE_YEAR) * weight
                                             # So 2020→1, 2026→7 (oldest=lowest, newest=highest)

# ── LAST COMPLETE YEAR ───────────────────────────────────────────
# If your final year is partial (e.g., data pulled mid-year), set this
# to the last FULL calendar year. Statistical tests and trending analysis
# will use YEAR_STATS_MAX to avoid partial-year bias. Descriptive figures
# (bar charts, heatmaps) still show ALL data including partial year.
YEAR_STATS_MAX = 2025                        # Last complete year for inference

# Trending split: "early" vs "late" periods (statistical tests only)
# Uses YEAR_STATS_MAX (not YEAR_MAX) to exclude partial-year data.
_mid = YEAR_MIN + (YEAR_STATS_MAX - YEAR_MIN) // 2 + 1  # 2020 + 2 + 1 = 2023
TREND_EARLY = (YEAR_MIN, _mid - 1)             # (2020, 2022)
TREND_LATE  = (_mid, YEAR_STATS_MAX)            # (2023, 2025) — complete years only

# ── GEOGRAPHY ─────────────────────────────────────────────────────
# Set to None for global (no geographic filter)
# Examples:
#   'AND (AFFILCOUNTRY(United States) OR AFFILCOUNTRY(Canada))'
#   'AND AFFILCOUNTRY(France)'
#   None
GEO_FILTER_SCOPUS = ' AND (AFFILCOUNTRY(United States) OR AFFILCOUNTRY(Canada))'

# For highlighting bars in country chart and computing geo stats
# Set to empty list for global analysis
GEO_HIGHLIGHT_COUNTRIES = ["United States", "Canada"]
GEO_LABEL = "US + Canada"                   # Short label for figure titles

# For affiliation-based stats (computed in audit & analysis)
# These are regex patterns matched against the affiliation_country column
GEO_PRIMARY_REGEX = "United States|Canada"   # "North American" papers
GEO_SECONDARY_REGEX = "Canada"               # Subset of interest (e.g., Canadian)
GEO_PRIMARY_LABEL = "North American"
GEO_SECONDARY_LABEL = "Canadian"

# ── WORD CLOUD ────────────────────────────────────────────────────
# Field-specific words to remove from the word cloud (too obvious / dominant)
WORDCLOUD_STOPWORDS = ["trauma", "injury", "surgical", "surgery", "patient",
                       "patients", "study", "associated", "clinical", "outcome",
                       "outcomes", "results", "analysis", "retrospective"]

# ── JOURNAL HIGHLIGHTING ──────────────────────────────────────────
# Substrings to flag a journal as "core field" (red bars in journal chart)
# A journal is highlighted if any of these appear in its name (case-insensitive)
JOURNAL_HIGHLIGHT_KEYWORDS = ["trauma", "emergency", "injury"]
JOURNAL_HIGHLIGHT_LABEL = "Trauma / Emergency focused"
JOURNAL_OTHER_LABEL = "Other journals"

# ── TRENDING ANALYSIS ─────────────────────────────────────────────
# Minimum total papers (early + late) for a concept to appear in trending chart
TRENDING_MIN_PAPERS = 8
# Cap for concepts that appear only in late period (no early baseline)
TRENDING_NEW_TOPIC_CAP = 200

# ── FIGURE OUTPUT ─────────────────────────────────────────────────
FIG_DIR_NAME = "figures_v3"                  # Subfolder name for figures

# ── RELEVANCE SCORING WEIGHTS ─────────────────────────────────────
# score = CITE_WEIGHT * log(1+citations) + RECENCY_WEIGHT * (year-BASE_YEAR)
#       + JOURNAL_BONUS (if in top journal) + GEO_PRIMARY_BONUS + GEO_SECONDARY_BONUS
CITE_WEIGHT = 10
RECENCY_WEIGHT = 5
JOURNAL_BONUS = 30
GEO_PRIMARY_BONUS = 10       # Papers from primary geography
GEO_SECONDARY_BONUS = 10     # Extra bonus for secondary geography (e.g., Canada)

# ── LLM SETTINGS ────────────────────────────────────────────────────
# Provider: "openai", "anthropic", "google", "ollama"
# Model: None = use provider default (see llm_providers.py for defaults)
# API key: set via environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
#          or pass --api-key to llm_pipeline.py
LLM_PROVIDER = "openai"
LLM_MODEL = None                  # None = provider default
LLM_TEMPERATURE = 0.0             # Deterministic for reproducibility
LLM_BATCH_SIZE = 10               # Papers per progress update
LLM_MAX_RETRIES = 3               # Retries on schema validation failure
LLM_CACHE_DIR = "llm_cache"       # Incremental save directory

# Human-in-the-loop confidence thresholds
LLM_CONFIDENCE_AUTO = 0.8         # Auto-accept if confidence >= this
LLM_CONFIDENCE_REVIEW = 0.5       # Flag for human review if between this and AUTO
