# Bibliometric Analysis Roadmap
## A Step-by-Step Guide to Reproducing This Methodology in Any Field

---

## Overview

This roadmap documents a complete, reproducible bibliometric analysis pipeline. It was built for trauma acute care surgery but is designed so you can swap in any clinical or scientific field.

**What you'll produce:**
- A corpus of thousands of peer-reviewed papers from Scopus
- 18 publication-ready figures (bar charts, heatmaps, trend lines, word clouds, tables)
- An auditable, reproducible pipeline where every number is computed from data

**Time estimate:**
- Technical setup (install, API key, clone repo): ~1 hour
- Intellectual design (journal list, concepts, exclusion terms): 4-8 hours (field expertise required)
- Running the pipeline (Scopus API + figures): 1-2 hours (mostly waiting)
- Audit & regex refinement: 2-4 hours (iterative)
- **Total realistic estimate: 1-2 days for a new field**

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Scopus API key** | Free via [Elsevier Developer Portal](https://dev.elsevier.com/). Institutional affiliation required. |
| **Python 3.10+** | With pip |
| **bibtool library** | The `bibtool/` folder in this project (wraps Scopus API, handles dedup) |
| **Python packages** | Install via `pip install -r requirements.txt` |
| **spaCy model** | `python -m spacy download en_core_web_sm` |

Store your Scopus key in `~/auth_cache.json`:
```json
{"scopus": "YOUR_API_KEY_HERE"}
```

> **Security note:** Add `auth_cache.json` to your `.gitignore` if using version control. Alternatively, use an environment variable.

---

## Architecture: The config.py System

**The single most important design decision:** all field-specific settings live in ONE file: `config.py`. Every other script imports from it. When adapting to a new field, you edit `config.py` first, then the 3 content files (search script, filter keywords, concept definitions).

### What config.py controls:

| Setting | What it does | Example |
|---|---|---|
| `FIELD_NAME` | Full name in figure titles | "Trauma Acute Care Surgery" |
| `FIELD_SHORT` | Short name for tight spaces | "Trauma" |
| `YEAR_MIN / YEAR_MAX` | Time window for search | 2020, 2026 |
| `BASE_YEAR` | Anchor for recency scoring | YEAR_MIN - 1 |
| `GEO_FILTER_SCOPUS` | Scopus geographic filter string | `' AND (AFFILCOUNTRY(...))'` or `None` |
| `GEO_HIGHLIGHT_COUNTRIES` | Countries to highlight in charts | `["United States", "Canada"]` |
| `GEO_LABEL` | Short geography label | "US + Canada" |
| `GEO_PRIMARY_REGEX` | Regex for primary geography stats | `"United States\|Canada"` |
| `GEO_SECONDARY_REGEX` | Regex for secondary subset (optional) | `"Canada"` |
| `WORDCLOUD_STOPWORDS` | Field-obvious words to remove from word cloud | `["trauma", "injury", "surgery"]` |
| `JOURNAL_HIGHLIGHT_KEYWORDS` | Substrings to flag core journals in chart | `["trauma", "emergency"]` |
| `TRENDING_MIN_PAPERS` | Minimum papers to show in trending chart | 8 (increase for larger corpora) |
| `FIG_DIR_NAME` | Output folder for figures | `"figures_v3"` |
| `CITE_WEIGHT`, `RECENCY_WEIGHT`, etc. | Relevance equation weights | See equation section |

### What is NOT in config.py (must be edited per-field):

| File | What you customize |
|---|---|
| `search_[field].py` | Scopus queries (journal names, topic terms) |
| `filter_results.py` | Exclusion keywords, inclusion keywords, top journals list |
| `concept_definitions.py` | Clinical concept regex patterns, domain groups |

---

## STEP 0: Edit config.py

Before anything else, open `config.py` and set your field parameters. Every downstream script reads from here, so you only do this once.

```python
# config.py — edit these for your field

FIELD_NAME = "Interventional Cardiology"    # your field
FIELD_SHORT = "Cardiology"
YEAR_MIN = 2018
YEAR_MAX = 2026
GEO_FILTER_SCOPUS = None                    # None = global search
GEO_HIGHLIGHT_COUNTRIES = []                 # empty = no highlighting
GEO_LABEL = "Global"
GEO_PRIMARY_REGEX = ""                       # empty = skip geo stats
GEO_SECONDARY_REGEX = ""
WORDCLOUD_STOPWORDS = ["cardiac", "heart", "cardiovascular", "patient", "patients"]
JOURNAL_HIGHLIGHT_KEYWORDS = ["cardiol", "heart", "circulat"]
JOURNAL_HIGHLIGHT_LABEL = "Cardiology focused"
```

---

## STEP 1: Define Your Field & Scope

Before writing a single line of code, answer these questions on paper. This is the intellectual foundation — get it right before touching code.

### 1A. What is the clinical/scientific domain?
> Example: "Trauma Acute Care Surgery"
>
> **Your field:** _______________

### 1B. What is the time window?
> Example: 2020-2026 (captures COVID impact + current trends)
> Rule of thumb: 5-7 years gives enough data for trending analysis.
>
> **Your window:** _______________

### 1C. What geographic restriction (if any)?
> Options:
> - No restriction (global) — set `GEO_FILTER_SCOPUS = None` in config.py
> - Single country: `' AND AFFILCOUNTRY(France)'`
> - Multi-country: `' AND (AFFILCOUNTRY(United States) OR AFFILCOUNTRY(Canada))'`
>
> **Your filter:** _______________

### 1D. Target corpus size?
> - Small (500-1,500): Narrow specialty or single-country
> - Medium (2,000-5,000): Most clinical fields (what we used: 4,599)
> - Large (5,000-15,000): Broad fields like oncology, cardiology
>
> Adjust query limits to control this (see Step 3).

---

## STEP 2: Design the 3-Layer Search Strategy

This is the most important methodological decision. The key principle: **let the data decide what matters, don't pre-select topics.**

### Layer 1: Journal-Level Sweep (Unbiased Foundation — ~60-70% of corpus)
**Purpose:** Capture EVERYTHING published in core journals. No topic filter = no bias.

**How to build your journal list:**
1. Identify 4-8 **core specialty journals** (100% relevant, no topic filter needed)
2. Identify 8-12 **adjacent/general journals** (need a broad field keyword filter)
3. Identify 3-5 **subspecialty/education journals** (need a field filter)

> **How to find the right journals:** Check which journals publish the most in your field using Scopus's "Source" tab. Look at where your department's faculty publish. Check the journal list of systematic reviews in your field.

**Template:**
```
CATEGORY A - Core [YOUR FIELD] journals (NO topic filter):
  - [Flagship journal] — sweep everything
  - [2nd most important journal]
  ...

CATEGORY B - General/adjacent journals (+ broad field keyword filter):
  - [General journal] AND TITLE-ABS-KEY([your field] OR [synonym])
  ...

CATEGORY C - Subspecialty journals (+ field filter):
  - [Subspecialty journal] AND TITLE-ABS-KEY([your field] OR [synonym])
  ...
```

**Worked examples:**

| Field | Core (no filter) | General (+filter) | Filter keywords |
|---|---|---|---|
| **Trauma** | JTACS, Injury, WJES, TSACO | Ann Surg, JAMA Surg, BJS | `trauma OR injur* OR hemorrha*` |
| **Cardiology** | Circulation, JACC, Eur Heart J | NEJM, JAMA, Lancet | `cardiac OR heart OR cardiovascular` |
| **Oncology** | J Clin Oncol, Lancet Oncol, Ann Oncol | NEJM, JAMA, BMJ | `cancer OR oncol* OR tumor* OR neoplasm*` |
| **Orthopedics** | JBJS, Clin Orthop, J Orthop Res | Ann Surg, Am J Surg | `orthop* OR fracture OR arthroplasty` |
| **Emergency Med** | Ann Emerg Med, Acad Emerg Med | JAMA, BMJ, Lancet | `emergency OR acute OR urgency` |

### Layer 2: Broad Topic Queries (~20-30% of corpus)
**Purpose:** Catch papers about your field published in ANY journal.

Write 1-3 broad queries using `TITLE-ABS-KEY()`:
```
TITLE-ABS-KEY(("[your field]" OR "[synonym]")
  AND (innovat* OR outcome* OR management OR guideline OR trial
       OR "systematic review" OR "meta-analysis" OR mortalit*))
AND DOCTYPE(ar OR re)
```

And a TITLE-only safety net:
```
TITLE([your field] AND ([key procedure] OR [key outcome]))
AND DOCTYPE(ar OR re)
```

### Layer 3: Targeted Niche Supplements (~5-10% of corpus)
**Purpose:** Emerging cross-disciplinary topics that appear in non-field journals.

Pick 5-8 niche topics:
```
TITLE-ABS-KEY(("[niche topic]") AND ("[your field]"))
```

**Critical:** Give niche queries LOWER rate limits (30-50/year) vs. journal sweeps (150-200/year). This ensures L3 supplements rather than dominates.

---

## STEP 3: Write the Search Script

Create `search_[your_field].py`. See the template in the existing `search_trauma_v3_unbiased.py`.

**Key bibtool API:**
| Function | What it does |
|---|---|
| `SearchQuery(query, min_year, max_year)` | Wraps a Scopus query string |
| `search.scan(query, max_results_per_year, source="scopus")` | Runs query, auto-deduplicates by DOI |
| `search.export_csv(path)` | Saves CSV with: `doi, title, abstract, publication, year, citations_count, affiliation_country, query` |

**Rate limit guidance:**
| Layer | Suggested `max_results_per_year` | Rationale |
|---|---|---|
| L1 journal sweeps | 150-200 | These are the unbiased foundation |
| L2 broad topic | 150-200 | Wide net across all Scopus |
| L3 niche | 30-50 | Supplements, should not dominate |

**Run it:**
```bash
python search_[your_field].py
```
Expected: `results_refined/all_results.csv` with thousands of rows.

> **If a query fails:** The script continues to the next query. Check the terminal output — if a whole layer failed, you'll have a biased corpus. Re-run.

---

## STEP 4: Build the Filtering Pipeline

Edit `filter_results.py` with three components:

### 4A. Exclusion Keywords (EXCLUDE_TITLE_KEYWORDS)
Papers to REMOVE — off-topic noise that slipped through journal sweeps.

**How to build this list:**
1. Run the search first
2. Open `results_refined/all_results.csv`
3. Sort by title, scan for obvious noise
4. Add noise patterns to the exclusion list

**WARNING — Field-specific traps:**
| If your field is... | Do NOT exclude... | Because... |
|---|---|---|
| Cardiology | "cardiac arrest" | It's core to your field |
| Oncology | "chemotherapy", "radiotherapy" | They ARE your field |
| Orthopedics | "arthroplasty", "joint replacement" | They ARE your field |
| Resuscitation | "cardiac arrest" | It's core |
| Bioengineering | "nanocomposite", "alloy" | Could be biomaterials research |

> The current exclusion list is trauma-specific. **You must rebuild it for your field.** Start with just psychology/PTSD and veterinary exclusions, then add more as you audit.

### 4B. Inclusion Keywords (REQUIRE_ANY_KEYWORD)
Papers must match AT LEAST ONE in title+abstract. This is a safety net, not a topic selector.

**How to know you have enough terms:**
- Run the filter, check how many papers survive
- If < 50% survive, your inclusion list is too narrow — add terms
- If > 90% survive, the list is fine (it's just catching noise)
- Target: 60-80% survival rate

### 4C. Top Journals List (TOP_TRAUMA_JOURNALS)
**Rename this variable** for your field (e.g., `TOP_CARDIOLOGY_JOURNALS`).

> **Journal bonus stacking is prevented:** The code now marks journals as "already matched" so a paper doesn't get double-counted if its journal name is a substring of another.

### 4D. Relevance Scoring Equation
All weights come from `config.py`:
```
score = CITE_WEIGHT * log(1+citations) + RECENCY_WEIGHT * (year - BASE_YEAR)
      + JOURNAL_BONUS + GEO_PRIMARY_BONUS + GEO_SECONDARY_BONUS
```

**This score is for RANKING only.** It does not filter papers in or out. It does not weight the bibliometric analysis. The figures use raw counts.

**Run it:**
```bash
python filter_results.py
```

---

## STEP 5: Define Clinical Concepts

Edit `concept_definitions.py` — the single source of truth for all topic detection.

### 5A. Choose Your Concepts (30-50 recommended)
Brainstorm every concept, technique, population, or trend in your field.

> **Concepts WILL overlap.** A single paper can match multiple concepts (e.g., "Geriatric pelvic fracture with REBOA" matches 3 concepts). This is by design. The heatmap and bar charts will sum to more than n. Acknowledge this in your methods slide.

### 5B. Write Precise Regex Patterns
**Critical rules to avoid overcounting:**

1. **Require context for ambiguous terms:**
   ```python
   # BAD: "children" matches non-trauma pediatric papers
   "Pediatric": r"pediatric|\bchild\b|\bchildren\b"

   # GOOD: requires field context
   "Pediatric": r"(?:pediatric|paediatric).*(?:trauma|injur|fracture)|pediatric trauma"
   ```

2. **Use word boundaries for short abbreviations:**
   ```python
   # BAD: matches "telegram", "integral"
   "TEG": r"teg"

   # GOOD:
   "TEG": r"\bteg\b|\brotem\b|thromboelast"
   ```

3. **Cross-disciplinary topics need field anchoring:**
   ```python
   # BAD: matches any ECMO paper
   "ECMO": r"ecmo|ecls|extracorporeal"

   # GOOD:
   "ECMO in Trauma": r"(?:ecmo|ecls).*trauma|trauma.*(?:ecmo|ecls)"
   ```

4. **Avoid bare common words:**
   ```python
   # BAD: "blunt" matches "blunt dissection technique"
   "Blunt Trauma": r"\bblunt\b"

   # GOOD:
   "Blunt Trauma": r"blunt.*(?:trauma|injur|abdominal|thoracic)|(?:trauma|injur).*\bblunt\b"
   ```

### 5C. Validate Every Pattern (MANDATORY)
After defining patterns, run this validation:
```python
import re, pandas as pd
df = pd.read_csv("results_curated/all_filtered.csv")
for concept, pattern in CLINICAL_CONCEPTS.items():
    matches = df[df["title"].str.contains(pattern, flags=re.IGNORECASE, na=False)]
    print(f"\n{concept}: {len(matches)} matches")
    for t in matches["title"].sample(min(5, len(matches))).values:
        print(f"  - {t}")
```

If you see false positives, tighten the regex. **Aim for >90% precision on every concept.** This step typically takes 2-3 iterations.

### 5D. Methodological Note: Title-Only vs Title+Abstract
The concept detection in `analysis.py` uses **title-only** matching. This is a deliberate methodological choice:
- Titles are the most specific indicator of a paper's primary topic
- Abstract matching would inflate counts with tangential mentions
- The inclusion filter (Step 4) uses title+abstract, so relevant papers are in the corpus

Document this in your methodology slide: "Clinical concepts identified via title-based regex matching."

### 5E. Group Concepts into Domains
For subplot grid figures:
```python
DOMAIN_GROUPS = {
    "Domain Name 1": ["Concept A", "Concept B", "Concept C"],
    "Domain Name 2": ["Concept D", "Concept E"],
    # 4-6 groups, each with 3-8 concepts
}
```

---

## STEP 6: Generate Figures

### What analysis.py generates (figures 01-15):
| # | Figure | What it shows | Customization |
|---|---|---|---|
| 01 | Flowchart | PRISMA-style pipeline | **Auto-computed from data** |
| 02 | Query contribution | Papers per search query | Query names in query_map (field-specific) |
| 03 | Year bar chart | Publications per year | Auto |
| 04 | Top journals | Top 20 journals | Journal highlighting via config.py |
| 05 | Geographic distribution | Top 20 countries | Country highlighting via config.py |
| 06 | Word cloud | Title keywords | Stopwords via config.py |
| 07 | Heatmap | Concepts x Years (top 30) | Auto from concept_definitions.py |
| 08 | Trending bar | Rising & declining concepts | Period split via config.py |
| 09 | Top bigrams | Most frequent word pairs | Auto |
| 10 | Citation scatter | Citations vs Year | Auto (fixed random seed for reproducibility) |
| 11 | Time series | 12 key concepts over time | Edit `key_topic_names` list for your field |
| 12 | Geo stacked bar | Primary geo vs world by year | Labels via config.py |
| 13 | Top cited table | 15 most-cited papers | Auto |
| 14 | Theme word clouds | Word cloud per domain group | Auto from DOMAIN_GROUPS |
| 15 | Search queries table | Methodology documentation | **Must edit query descriptions for your field** |

### What line_chart_all_topics.py generates (figures 16-18):
| # | Figure | What it shows |
|---|---|---|
| 16 | All concepts line chart | Every concept over time (full detail) |
| 17 | Top 10 line chart | Cleaner version for main slides |
| 18 | Domain grid | 6-panel subplot by domain group |

**Things you MUST customize in analysis.py:**
1. `query_map` in `fig2_query_categories_bar` — map your Scopus query prefixes to readable names
2. Query descriptions in `fig15_search_queries_table` — document YOUR search strategy
3. `key_topic_names` in `fig11_concept_timeseries` — pick YOUR 8-12 most interesting concepts

**Things that are now automatic (via config.py):**
- Flowchart numbers (all computed from data)
- Figure titles with field name, corpus size, geography
- Trending period split
- Geographic highlighting
- Word cloud stopwords
- Journal highlighting

**Run it:**
```bash
python analysis.py
python line_chart_all_topics.py
```

---

## STEP 7: Audit Everything

```bash
python audit.py
```

The audit script now computes ALL numbers from data — no hardcoded "claimed" values. It checks:

1. **Pipeline math:** raw -> excluded -> post-exclusion -> final
2. **Year distribution:** every year has papers, no gaps
3. **Geographic numbers:** match your config expectations
4. **Journal distribution:** top journals make sense for your field
5. **Concept counts:** per-concept totals
6. **Concept coverage:** what % of papers match at least one concept
7. **Trending:** early vs late period comparison (top 8 concepts)
8. **Data integrity:** 0 duplicate DOIs, 0 empty titles, 0 missing years

> The audit will print "All checks passed" or list specific issues. Fix issues upstream and re-run.

---

## STEP 8: Iterate the Regex-Audit Loop

This is where most of your time goes. The loop:

```
1. Run filter_results.py
2. Run analysis.py + line_chart_all_topics.py
3. Run audit.py
4. Look at the heatmap (fig07) and trending chart (fig08)
5. Ask: "Does this make sense for my field?"
6. If a concept looks inflated:
   a. Run the validation script (Step 5C)
   b. Check sample titles for false positives
   c. Tighten the regex in concept_definitions.py
   d. Go to step 1
7. If a concept is missing:
   a. Add it to CLINICAL_CONCEPTS
   b. Add it to a DOMAIN_GROUP
   c. Go to step 1
```

**Expect 2-4 iterations.** Each takes ~5 minutes to re-run.

---

## Key Methodological Decisions to Document

When presenting or publishing, explicitly state:

1. **Search strategy:** "3-layer unbiased search: journal sweep + broad topic + niche supplements"
2. **Concept detection:** "Title-based regex matching (not abstract)"
3. **Concept overlap:** "Concepts are not mutually exclusive; one paper can match multiple concepts"
4. **Trending method:** "Relative frequency change between early and late periods, normalized by total papers per period"
5. **Relevance score:** "Used for ranking only, not for filtering or weighting analysis"
6. **Limitations:** Single database (Scopus), English-language bias, title-only concept matching, partial final year

---

## Common Pitfalls & How to Avoid Them

| Pitfall | Why it's bad | Fix |
|---|---|---|
| **Starting with topic-specific queries** | Biases corpus toward predetermined topics | Use journal-level sweeps as foundation |
| **Broad regex without context** | "children" matches non-pediatric papers | Require field-specific context words |
| **Exclusion list from another field** | Removes valid papers (e.g., "cardiac arrest" in cardiology) | Rebuild exclusion list from scratch for your field |
| **Using relevance score for analysis** | Conflates ranking with measurement | Score ranks CSVs only; figures use raw counts |
| **Not auditing concept matches** | Silent overcounting | Spot-check 5+ titles per concept |
| **Too many niche L3 queries** | Niche topics over-represented | Keep L3 limits at 30-50/yr |
| **Ignoring partial final year** | Looks like a decline in the last year | Note in figures |
| **Non-English papers in global search** | English regex misses them, creating systematic bias | Add `AND LANGUAGE(English)` to queries, or acknowledge limitation |
| **Not checking for API failures** | Missing whole query layer = biased corpus | Check terminal output for ERROR lines |

---

## File Structure

```
Your_Project/
  config.py                       # STEP 0: All field-specific settings (EDIT FIRST)
  requirements.txt                # Python dependencies
  bibtool/                        # Scopus API wrapper (don't modify)
    src/bibtool/
      search_papers.py
      ...
  search_[your_field].py          # STEP 3: Scopus search queries
  filter_results.py               # STEP 4: Exclusion/inclusion/scoring
  concept_definitions.py          # STEP 5: Regex patterns (single source of truth)
  analysis.py                     # STEP 6: Figure generation (15 figures)
  line_chart_all_topics.py        # STEP 6: Line charts (3 figures)
  audit.py                        # STEP 7: Number verification
  results_refined/                # Raw Scopus output
    all_results.csv
  results_curated/                # Filtered + scored output
    all_filtered.csv
    theme_*.csv
  [FIG_DIR_NAME]/                 # Publication-ready figures
    01_methodology_flowchart.png
    ...
    18_concepts_by_domain_grid.png
```

---

## Execution Order

```bash
# 0. Edit config.py with your field settings

# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Search Scopus (30-60 min depending on corpus size)
python search_[your_field].py

# 3. Filter and score (< 1 min)
python filter_results.py

# 4. Generate all figures (2-3 min)
python analysis.py
python line_chart_all_topics.py

# 5. Audit (< 1 min)
python audit.py

# 6. If concepts are noisy or missing:
#    - Edit concept_definitions.py
#    - Re-run steps 3-5 (no need to re-search)
```

---

## Adapting the Relevance Equation

All weights are in `config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `CITE_WEIGHT` | 10 | How much citations matter |
| `RECENCY_WEIGHT` | 5 | How much newer papers are favored |
| `JOURNAL_BONUS` | 30 | Bonus for top-tier journals |
| `GEO_PRIMARY_BONUS` | 10 | Bonus for primary geography |
| `GEO_SECONDARY_BONUS` | 10 | Extra bonus for secondary subset |

The equation: `score = CITE_WEIGHT * log(1+citations) + RECENCY_WEIGHT * (year - BASE_YEAR) + bonuses`

**BASE_YEAR** is automatically set to `YEAR_MIN - 1` in config.py.

Remember: this only affects the "Top 30 Papers" and theme CSV rankings. It never touches the figures.

---

## Trending Calculation Explained

The trending analysis splits the corpus into two periods (defined in `config.py`):

```
TREND_EARLY = (2020, 2022)    # first half
TREND_LATE  = (2023, 2026)    # second half

For each concept:
  early_share = concept_papers_in_early / total_papers_in_early
  late_share  = concept_papers_in_late  / total_papers_in_late
  trend       = ((late_share - early_share) / early_share) * 100
```

This normalizes by period size, so overall publication growth doesn't inflate trends. A concept can grow in absolute numbers but decline as a share of the literature.

**Configurable:**
- `TRENDING_MIN_PAPERS`: minimum total papers for a concept to appear (default: 8). Increase for large corpora.
- `TRENDING_NEW_TOPIC_CAP`: cap for concepts with no early baseline (default: 200%).

---

## Checklist Before Presenting

- [ ] `audit.py` passes clean ("All checks passed")
- [ ] Spot-checked 5+ titles for each top-10 concept (no false positives)
- [ ] Final year noted as partial in all trend figures
- [ ] Methodology slide explains 3-layer strategy
- [ ] Concept overlap acknowledged ("concepts are not mutually exclusive")
- [ ] Relevance equation described correctly (ranking only, not filtering)
- [ ] Limitations stated (title-only matching, single database, geographic filter, partial year)
- [ ] All figure titles show correct field name and corpus size (auto from config.py)
- [ ] No hardcoded numbers in slides — all pulled from audit output
- [ ] Exclusion list reviewed for field-appropriateness (no accidentally removed terms)
- [ ] Search terminal output checked for API errors (no missing query layers)
