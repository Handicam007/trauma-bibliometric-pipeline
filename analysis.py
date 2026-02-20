#!/usr/bin/env python3
"""
Lexical Analysis & Visualization for Trauma Acute Care Journal Club
=====================================================================
Generates presentation-ready figures and tables from the curated corpus.

Outputs to: /figures/
"""

import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from matplotlib.patches import FancyBboxPatch
from wordcloud import WordCloud

from concept_definitions import CLINICAL_CONCEPTS, DOMAIN_GROUPS
from config import (
    FIELD_NAME, FIELD_SHORT, YEAR_MIN, YEAR_MAX, YEAR_STATS_MAX, BASE_YEAR,
    TREND_EARLY, TREND_LATE, GEO_HIGHLIGHT_COUNTRIES, GEO_LABEL,
    GEO_PRIMARY_REGEX, GEO_SECONDARY_REGEX, GEO_PRIMARY_LABEL, GEO_SECONDARY_LABEL,
    WORDCLOUD_STOPWORDS, JOURNAL_HIGHLIGHT_KEYWORDS, JOURNAL_HIGHLIGHT_LABEL,
    JOURNAL_OTHER_LABEL, TRENDING_MIN_PAPERS, TRENDING_NEW_TOPIC_CAP,
    FIG_DIR_NAME,
)

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────
INPUT = Path(__file__).parent / "results_curated" / "all_filtered.csv"
RAW_INPUT = Path(__file__).parent / "results_refined" / "all_results.csv"
FIG_DIR = Path(__file__).parent / FIG_DIR_NAME
FIG_DIR.mkdir(exist_ok=True)

# Presentation style
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color palette — professional medical/academic
COLORS = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#E74C3C",
    "warm": "#E67E22",
    "green": "#27AE60",
    "purple": "#8E44AD",
    "gray": "#7F8C8D",
}
PALETTE = list(COLORS.values()) + [
    "#1ABC9C", "#D35400", "#2C3E50", "#F39C12", "#7D3C98",
    "#16A085", "#C0392B", "#2980B9", "#D4AC0D", "#1B4F72",
]

# ── NLP Setup ───────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Domain-specific stopwords to exclude from analysis
STOPWORDS = {
    "study", "analysis", "retrospective", "prospective", "cohort", "review",
    "systematic", "meta", "case", "report", "series", "patient", "patients",
    "result", "results", "outcome", "outcomes", "clinical", "data",
    "associated", "association", "impact", "effect", "effects", "use",
    "based", "approach", "using", "novel", "new", "current",
    "multicenter", "multicentre", "single", "center", "centre",
    "united", "states", "national", "international", "global",
    "year", "month", "time", "group", "rate", "risk", "factor",
    "management", "treatment", "care", "evaluation", "assessment",
    "comparison", "experience", "role", "update", "evidence",
    "guideline", "guidelines", "recommendation", "consensus",
    "level", "type", "model", "score", "tool", "method",
    "high", "low", "early", "late", "major", "large",
}


def extract_terms(titles, min_freq=3):
    """Extract meaningful terms from titles using spaCy."""
    # Unigrams
    unigram_counts = Counter()
    # Bigrams & trigrams (from raw text)
    bigram_counts = Counter()
    trigram_counts = Counter()

    for title in titles:
        doc = nlp(title.lower())
        tokens = [
            t.lemma_ for t in doc
            if not t.is_stop and not t.is_punct and t.is_alpha
            and len(t.text) > 2 and t.lemma_ not in STOPWORDS
        ]
        unigram_counts.update(tokens)

        # Bigrams from cleaned tokens
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]} {tokens[i+1]}"
            bigram_counts[bg] += 1

        # Trigrams
        for i in range(len(tokens) - 2):
            tg = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            trigram_counts[tg] += 1

    # Filter by minimum frequency
    unigrams = {k: v for k, v in unigram_counts.items() if v >= min_freq}
    bigrams = {k: v for k, v in bigram_counts.items() if v >= min_freq}
    trigrams = {k: v for k, v in trigram_counts.items() if v >= max(min_freq, 4)}

    return unigrams, bigrams, trigrams


def extract_clinical_concepts(titles):
    """Extract specific clinical concepts using canonical regex patterns.

    Uses CLINICAL_CONCEPTS from concept_definitions.py (single source of truth).
    """
    results = {}
    titles_lower = [t.lower() for t in titles]
    for concept, pattern in CLINICAL_CONCEPTS.items():
        try:
            count = sum(1 for t in titles_lower if re.search(pattern, t))
        except re.error as e:
            print(f"  !! REGEX ERROR in concept '{concept}': {e}")
            print(f"     Pattern: {pattern[:80]}...")
            print(f"     Fix this in concept_definitions.py, then re-run.")
            count = 0
        if count > 0:
            results[concept] = count
        elif count == 0:
            pass  # Zero-match concepts are reported by validate.py and audit.py

    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


# ═══════════════════════════════════════════════════════════════════
# FIGURE GENERATORS
# ═══════════════════════════════════════════════════════════════════

def fig1_methodology_flowchart(df):
    """FIGURE 1: Methodology flowchart — PRISMA-style.
    All numbers are computed from the data, never hardcoded.
    """
    # Compute pipeline numbers from data
    n_final = len(df)
    n_concepts = len(CLINICAL_CONCEPTS)
    n_years = YEAR_MAX - YEAR_MIN + 1

    # Geographic stats
    geo_primary = df["affiliation_country"].fillna("").str.contains(
        GEO_PRIMARY_REGEX, case=False, na=False).sum()
    geo_secondary = df["affiliation_country"].fillna("").str.contains(
        GEO_SECONDARY_REGEX, case=False, na=False).sum() if GEO_SECONDARY_REGEX else 0

    # Read raw file for pipeline numbers
    try:
        df_raw = pd.read_csv(RAW_INPUT)
        n_raw = len(df_raw)
    except FileNotFoundError:
        n_raw = n_final  # fallback

    # Compute exclusion/inclusion numbers by re-running the filter logic
    from filter_results import EXCLUDE_TITLE_KEYWORDS, EXCLUDE_ABSTRACT_KEYWORDS
    df_raw_temp = pd.read_csv(RAW_INPUT) if n_raw > n_final else df.copy()
    df_raw_temp["title"] = df_raw_temp["title"].fillna("")
    df_raw_temp["abstract"] = df_raw_temp["abstract"].fillna("")
    title_lower = df_raw_temp["title"].str.lower()
    abstract_lower = df_raw_temp["abstract"].str.lower()
    exclude_mask = pd.Series(False, index=df_raw_temp.index)
    for kw in EXCLUDE_TITLE_KEYWORDS:
        exclude_mask |= title_lower.str.contains(kw, case=False, na=False)
    for kw in EXCLUDE_ABSTRACT_KEYWORDS:
        exclude_mask |= abstract_lower.str.contains(kw, case=False, na=False)
    n_excluded = exclude_mask.sum()
    n_post_exclusion = n_raw - n_excluded

    # Count queries from search script
    try:
        n_queries = df_raw["query"].nunique()
    except Exception:
        n_queries = "?"

    geo_line = f"{geo_primary:,} {GEO_PRIMARY_LABEL}"
    if GEO_SECONDARY_REGEX:
        geo_line += f"  |  {geo_secondary:,} {GEO_SECONDARY_LABEL}"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    boxes = [
        (5, 9.2, f"SCOPUS Database Search (single database)\n3-layer strategy: journal sweep + broad topic + niche\n{n_queries} queries x {n_years} years ({YEAR_MIN}-{YEAR_MAX}) | {GEO_LABEL}", COLORS["primary"]),
        (5, 7.5, f"Raw Results Retrieved\nn = {n_raw:,} papers (DOI-deduplicated, titles + metadata)", COLORS["secondary"]),
        (5, 5.8, f"Exclusion Filter Applied\nRemoved off-topic papers\n(n = {n_excluded:,} excluded)", COLORS["accent"]),
        (5, 4.1, f"Inclusion Filter Applied\nRequired field-specific keywords\nn = {n_post_exclusion:,} -> {n_final:,}", COLORS["warm"]),
        (5, 2.4, f"Title-Based Lexical Analysis\n{n_concepts} clinical concepts matched via regex\n+ Relevance scoring (citations, recency, journal, geo bonus)", COLORS["purple"]),
        (5, 0.8, f"Final Curated Corpus\nn = {n_final:,} papers  |  {YEAR_MAX} data partial\n{geo_line}", COLORS["green"]),
    ]

    for x, y, text, color in boxes:
        bbox = FancyBboxPatch(
            (x - 3.8, y - 0.55), 7.6, 1.1,
            boxstyle="round,pad=0.15", facecolor=color, edgecolor="white",
            alpha=0.9, linewidth=2
        )
        ax.add_patch(bbox)
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                color="white", fontweight="bold", linespacing=1.4)

    # Arrows
    for i in range(len(boxes) - 1):
        y_start = boxes[i][1] - 0.6
        y_end = boxes[i + 1][1] + 0.6
        ax.annotate("", xy=(5, y_end), xytext=(5, y_start),
                     arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))

    ax.set_title("Search Methodology", fontsize=16, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_methodology_flowchart.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 01_methodology_flowchart.png")


def fig2_query_categories_bar(df):
    """FIGURE 2: Papers per search query category."""
    # Map queries to readable names
    query_map = {
        # Journal-level sweeps
        "(SRCTITLE(\"Journal of Trauma": "Core Trauma Journals",
        "(SRCTITLE(\"Annals of Surgery": "General Surgery Journals",
        "(SRCTITLE(\"Annals of Emergency": "Emergency Medicine Journals",
        "(SRCTITLE(\"Critical Care Med": "Critical Care Journals",
        "(SRCTITLE(\"Journal of Neurotrauma": "Neurotrauma & Ortho Journals",
        # Broad topic queries
        "TITLE-ABS-KEY((\"trauma surgery": "Broad Trauma Surgery",
        "TITLE(trauma AND (surgery OR": "Broad Trauma (Title Search)",
        # Targeted niche queries
        "TITLE-ABS-KEY(REBOA": "REBOA",
        "TITLE-ABS-KEY((\"artificial intelligence": "AI / Machine Learning",
        "TITLE-ABS-KEY((\"whole blood": "Blood Products",
        "TITLE-ABS-KEY((ECMO": "ECMO in Trauma",
        "TITLE-ABS-KEY((telemedicine": "Teletrauma",
        "TITLE-ABS-KEY((\"mass casualty": "Mass Casualty",
        "(SRCTITLE(\"New England Journal": "High-Impact Journals (NEJM, JAMA, Lancet)",
    }

    # Match query prefixes
    def categorize(q):
        q = str(q)[:50]
        for prefix, name in query_map.items():
            if q.startswith(prefix[:45]):
                return name
        return "Other"

    df["category"] = df["query"].apply(categorize)
    cat_counts = df["category"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        range(len(cat_counts)),
        cat_counts.values,
        color=[PALETTE[i % len(PALETTE)] for i in range(len(cat_counts))],
        edgecolor="white", linewidth=0.5
    )
    ax.set_yticks(range(len(cat_counts)))
    ax.set_yticklabels(cat_counts.index, fontsize=11)
    ax.set_xlabel("Number of Papers", fontsize=12)
    ax.set_title("Papers Retrieved per Search Category", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, val in zip(bars, cat_counts.values):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_query_categories.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 02_query_categories.png")


def fig3_publications_by_year(df):
    """FIGURE 3: Publication volume over time."""
    year_counts = df["year"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(year_counts.index.astype(str), year_counts.values,
                  color=COLORS["primary"], edgecolor="white", width=0.7)

    for bar, val in zip(bars, year_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(val), ha="center", fontsize=11, fontweight="bold")

    # Annotate 2026 as partial year
    if "2026" in year_counts.index.astype(str).tolist():
        idx_2026 = list(year_counts.index.astype(str)).index("2026")
        bars[idx_2026].set_alpha(0.5)
        bars[idx_2026].set_hatch("//")
        ax.text(bars[idx_2026].get_x() + bars[idx_2026].get_width()/2,
                bars[idx_2026].get_height() / 2,
                "partial\nyear", ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Papers", fontsize=12)
    ax.set_title(f"{FIELD_NAME} Research Output ({YEAR_MIN}-{YEAR_MAX})", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(year_counts.values) * 1.15)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_publications_by_year.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 03_publications_by_year.png")


def fig4_top_journals(df):
    """FIGURE 4: Top publishing journals."""
    journal_counts = df["publication"].value_counts().head(20)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [COLORS["accent"] if any(kw in j.lower() for kw in JOURNAL_HIGHLIGHT_KEYWORDS)
              else COLORS["secondary"] for j in journal_counts.index]

    bars = ax.barh(range(len(journal_counts)), journal_counts.values,
                   color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(journal_counts)))
    ax.set_yticklabels(journal_counts.index, fontsize=9)
    ax.set_xlabel("Number of Papers", fontsize=12)
    ax.set_title("Top 20 Journals in Corpus", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, val in zip(bars, journal_counts.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["accent"], label=JOURNAL_HIGHLIGHT_LABEL),
        Patch(facecolor=COLORS["secondary"], label=JOURNAL_OTHER_LABEL),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_top_journals.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 04_top_journals.png")


def fig5_geographic_distribution(df):
    """FIGURE 5: Geographic distribution of research."""
    countries = df["affiliation_country"].fillna("").str.split("|")
    country_flat = [c.strip() for sublist in countries for c in sublist if c.strip()]
    country_counts = Counter(country_flat)

    top20 = dict(Counter(country_counts).most_common(20))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = []
    for c in top20.keys():
        if c in GEO_HIGHLIGHT_COUNTRIES:
            colors.append(COLORS["accent"])
        else:
            colors.append(COLORS["secondary"])

    bars = ax.barh(range(len(top20)), list(top20.values()), color=colors,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(list(top20.keys()), fontsize=10)
    ax.set_xlabel("Number of Papers (with affiliation)", fontsize=12)
    ax.set_title("Geographic Distribution of Research", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, val in zip(bars, top20.values()):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["accent"], label=GEO_LABEL if GEO_HIGHLIGHT_COUNTRIES else "Highlighted"),
        Patch(facecolor=COLORS["secondary"], label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_geographic_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 05_geographic_distribution.png")


def fig6_wordcloud_all(df):
    """FIGURE 6: Word cloud of entire corpus titles."""
    text = " ".join(df["title"].tolist()).lower()
    # Clean
    for sw in list(STOPWORDS) + WORDCLOUD_STOPWORDS:
        text = re.sub(rf"\b{sw}\b", "", text)

    wc = WordCloud(
        width=1600, height=800,
        background_color="white",
        colormap="viridis",
        max_words=120,
        min_font_size=10,
        collocations=True,
        prefer_horizontal=0.8,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud -- {FIELD_NAME} Corpus (n={len(df):,}, {GEO_LABEL})",
                 fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_wordcloud_all.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 06_wordcloud_all.png")


def fig7_clinical_concepts_heatmap(df):
    """FIGURE 7: Clinical concepts frequency across years (heatmap)."""
    concepts_per_year = {}
    for year in sorted(df["year"].unique()):
        titles_year = df[df["year"] == year]["title"].tolist()
        concepts = extract_clinical_concepts(titles_year)
        concepts_per_year[year] = concepts

    # Build matrix
    all_concepts = set()
    for c in concepts_per_year.values():
        all_concepts.update(c.keys())

    # Sort by total frequency
    concept_totals = {}
    for concept in all_concepts:
        concept_totals[concept] = sum(
            concepts_per_year[y].get(concept, 0) for y in concepts_per_year
        )
    sorted_concepts = sorted(concept_totals, key=concept_totals.get, reverse=True)[:30]

    years = sorted(concepts_per_year.keys())
    matrix = []
    for concept in sorted_concepts:
        row = [concepts_per_year[y].get(concept, 0) for y in years]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=[str(y) for y in years],
        yticklabels=sorted_concepts,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "Paper Count"}
    )
    ax.set_title("Clinical Concepts × Year — Top 30 (Title Analysis)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_concepts_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 07_concepts_heatmap.png")


def fig8_trending_concepts(df):
    """FIGURE 8: Trending concepts — growth rate from 2020-2022 to 2023-2026.

    Now includes statistical significance markers from chi-squared/Fisher's
    exact tests with Benjamini-Hochberg correction. ★ = significant (p<0.05).
    """
    # Try to load pre-computed stats (from statistical_tests.py)
    stats_file = Path(__file__).parent / "stats_output" / "stats_report.csv"
    sig_concepts = set()
    or_lookup = {}  # concept -> (OR, OR_lower, OR_upper, p_adj)
    if stats_file.exists():
        try:
            stats_df = pd.read_csv(stats_file)
            sig_concepts = set(stats_df[stats_df["significant"] == True]["concept"].values)
            for _, row in stats_df.iterrows():
                or_lookup[row["concept"]] = (
                    row.get("OR", None),
                    row.get("OR_lower", None),
                    row.get("OR_upper", None),
                    row.get("p_adjusted", None),
                )
        except Exception:
            pass

    # Use only complete years for trend analysis (exclude partial year)
    df_stats = df[df["year"] <= YEAR_STATS_MAX]
    early = df_stats[df_stats["year"].between(TREND_EARLY[0], TREND_EARLY[1])]
    late = df_stats[df_stats["year"].between(TREND_LATE[0], TREND_LATE[1])]

    concepts_early = extract_clinical_concepts(early["title"].tolist())
    concepts_late = extract_clinical_concepts(late["title"].tolist())

    # Normalize by total papers in each period
    n_early = len(early)
    n_late = len(late)

    all_concepts = set(list(concepts_early.keys()) + list(concepts_late.keys()))
    trends = {}
    for c in all_concepts:
        freq_early = concepts_early.get(c, 0) / n_early * 100
        freq_late = concepts_late.get(c, 0) / n_late * 100
        if freq_early > 0:
            change = ((freq_late - freq_early) / freq_early) * 100
        elif freq_late > 0:
            change = TRENDING_NEW_TOPIC_CAP  # new topic
        else:
            change = 0
        if concepts_early.get(c, 0) + concepts_late.get(c, 0) >= TRENDING_MIN_PAPERS:
            trends[c] = {
                "early_%": round(freq_early, 2),
                "late_%": round(freq_late, 2),
                "change_%": round(change, 1),
                "total": concepts_early.get(c, 0) + concepts_late.get(c, 0)
            }

    sorted_trends = sorted(trends.items(), key=lambda x: x[1]["change_%"], reverse=True)

    # Cap at top 20 for readability (10 rising + 10 declining)
    if len(sorted_trends) > 20:
        rising = [t for t in sorted_trends if t[1]["change_%"] > 0][:12]
        declining = [t for t in sorted_trends if t[1]["change_%"] <= 0][-8:]
        sorted_trends = rising + declining

    fig, ax = plt.subplots(figsize=(12, 8))

    names = [t[0] for t in sorted_trends]
    changes = [t[1]["change_%"] for t in sorted_trends]
    colors = [COLORS["green"] if c > 0 else COLORS["accent"] for c in changes]

    bars = ax.barh(range(len(names)), changes, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))

    # Add ★ to significant concept labels
    y_labels = []
    for name in names:
        if name in sig_concepts:
            y_labels.append(f"★ {name}")
        else:
            y_labels.append(f"  {name}")
    ax.set_yticklabels(y_labels, fontsize=10)

    ax.set_xlabel("% Change in Relative Frequency", fontsize=12)
    ax.set_title(f"Trending Clinical Concepts\n({TREND_EARLY[0]}-{TREND_EARLY[1]} vs {TREND_LATE[0]}-{TREND_LATE[1]}, normalized by period size)",
                 fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.invert_yaxis()

    for bar, val, name in zip(bars, changes, names):
        offset = 3 if val >= 0 else -3
        ha = "left" if val >= 0 else "right"
        # Show effect size (OR) alongside % change for significant concepts
        if name in or_lookup and name in sig_concepts:
            or_val, or_lo, or_hi, p_adj = or_lookup[name]
            if or_val and not np.isnan(or_val):
                label = f"{val:+.0f}%  OR={or_val:.2f} ({or_lo:.2f}-{or_hi:.2f})"
            else:
                label = f"{val:+.0f}%"
        else:
            label = f"{val:+.0f}%"
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                label, va="center", ha=ha, fontsize=9, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["green"], label="Growing"),
        Patch(facecolor=COLORS["accent"], label="Declining"),
    ]
    if sig_concepts:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
                   markersize=12, label="★ Statistically significant (BH p<0.05)")
        )
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_trending_concepts.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 08_trending_concepts.png")


def fig9_top_bigrams(df):
    """FIGURE 9: Most frequent bigrams in titles."""
    _, bigrams, _ = extract_terms(df["title"].tolist(), min_freq=4)
    top_bg = dict(Counter(bigrams).most_common(25))

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(range(len(top_bg)), list(top_bg.values()),
                   color=COLORS["secondary"], edgecolor="white")
    ax.set_yticks(range(len(top_bg)))
    ax.set_yticklabels(list(top_bg.keys()), fontsize=10)
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_title("Top 25 Bigrams in Paper Titles", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, val in zip(bars, top_bg.values()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_top_bigrams.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 09_top_bigrams.png")


def fig10_citation_vs_recency(df):
    """FIGURE 10: Citations vs Year scatter — find the sweet spot."""
    df_plot = df[df["citations_count"] > 0].copy()
    df_plot["log_cites"] = np.log10(df_plot["citations_count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        df_plot["year"] + np.random.default_rng(42).normal(0, 0.1, len(df_plot)),
        df_plot["citations_count"],
        alpha=0.4, s=20, c=COLORS["secondary"], edgecolors="none"
    )

    # Highlight top papers
    top_papers = df_plot.nlargest(10, "citations_count")
    ax.scatter(top_papers["year"], top_papers["citations_count"],
               s=80, c=COLORS["accent"], edgecolors="white", zorder=5, linewidth=0.5)

    for _, row in top_papers.head(6).iterrows():
        short_title = row["title"][:55] + "..."
        ax.annotate(
            short_title,
            xy=(row["year"], row["citations_count"]),
            xytext=(10, 10), textcoords="offset points",
            fontsize=7, alpha=0.85,
            arrowprops=dict(arrowstyle="-", alpha=0.4, lw=0.5),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax.set_yscale("log")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Citations (log scale)", fontsize=12)
    ax.set_title("Citation Impact vs. Publication Year", fontsize=14, fontweight="bold")
    ax.set_xticks(sorted(df["year"].unique()))
    ax.set_xticklabels([str(y) for y in sorted(df["year"].unique())])

    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_citations_vs_year.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 10_citations_vs_year.png")


def fig11_concept_timeseries(df):
    """FIGURE 11: Time series of key innovation topics (canonical patterns)."""
    # Select a readable subset of key innovation topics (expanded v2)
    key_topic_names = [
        "REBOA", "Whole Blood / MTP", "AI / Machine Learning",
        "Geriatric / Frailty", "Non-Operative Mgmt", "TEG / ROTEM",
        "Simulation / Training", "Damage Control",
        "Fracture Management", "Military / Combat", "Polytrauma",
        "Pelvic / Acetabular",
    ]

    years = sorted(df["year"].unique())
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, topic in enumerate(key_topic_names):
        pattern = CLINICAL_CONCEPTS[topic]
        counts = []
        for y in years:
            titles_y = df[df["year"] == y]["title"].str.lower()
            count = titles_y.str.contains(pattern, regex=True, na=False).sum()
            counts.append(count)
        total = sum(counts)
        ax.plot(years, counts, marker="o", linewidth=2.5,
                label=f"{topic} (n={total})",
                color=PALETTE[i % len(PALETTE)], markersize=6)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Papers", fontsize=12)
    ax.set_title("Key Innovation Topics Over Time (canonical patterns)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_concept_timeseries.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 11_concept_timeseries.png")


def fig12_na_vs_world(df):
    """FIGURE 12: North America vs Rest of World contribution."""
    df_c = df.copy()
    df_c["region"] = "Rest of World"
    na_mask = df_c["affiliation_country"].fillna("").str.contains(
        GEO_PRIMARY_REGEX, case=False
    )
    df_c.loc[na_mask, "region"] = GEO_PRIMARY_LABEL

    pivot = df_c.groupby(["year", "region"]).size().unstack(fill_value=0)
    if GEO_PRIMARY_LABEL not in pivot.columns:
        pivot[GEO_PRIMARY_LABEL] = 0
    if "Rest of World" not in pivot.columns:
        pivot["Rest of World"] = 0

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=[COLORS["accent"], COLORS["secondary"]], edgecolor="white")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Papers", fontsize=12)
    ax.set_title(f"{GEO_PRIMARY_LABEL} vs. Global Research Output", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticklabels([str(y) for y in pivot.index], rotation=0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_na_vs_world.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 12_na_vs_world.png")


def fig13_top_cited_table(df):
    """FIGURE 13: Top 15 most-cited papers — table for slides."""
    top15 = df.nlargest(15, "citations_count")[
        ["title", "publication", "year", "citations_count"]
    ].copy()
    top15["title"] = top15["title"].str[:75]
    top15["publication"] = top15["publication"].str[:35]
    top15.columns = ["Title", "Journal", "Year", "Cites"]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")

    table = ax.table(
        cellText=top15.values,
        colLabels=top15.columns,
        cellLoc="left",
        loc="center",
        colWidths=[0.52, 0.25, 0.08, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(top15.columns)):
        table[0, j].set_facecolor(COLORS["primary"])
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(top15) + 1):
        color = "#F0F4F8" if i % 2 == 0 else "white"
        for j in range(len(top15.columns)):
            table[i, j].set_facecolor(color)

    ax.set_title("Top 15 Most-Cited Papers in Corpus",
                 fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_top_cited_table.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 13_top_cited_table.png")


def fig14_wordcloud_by_theme(df):
    """FIGURE 14: Mini word clouds for each major theme (uses DOMAIN_GROUPS)."""
    # Build theme patterns from DOMAIN_GROUPS (canonical source)
    themes = {}
    for group_name, concept_list in DOMAIN_GROUPS.items():
        patterns = []
        for concept in concept_list:
            if concept in CLINICAL_CONCEPTS:
                patterns.append(CLINICAL_CONCEPTS[concept])
        if patterns:
            # Use short label for subplot title
            label = group_name.replace(" & ", " &\n")
            themes[label] = "|".join(patterns)

    n_themes = len(themes)
    n_cols = 3
    n_rows = (n_themes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for idx, (theme_name, pattern) in enumerate(themes.items()):
        mask = df["title"].str.lower().str.contains(pattern, regex=True, na=False)
        theme_titles = df[mask]["title"].tolist()

        if not theme_titles:
            axes[idx].set_visible(False)
            continue

        text = " ".join(theme_titles).lower()
        for sw in list(STOPWORDS) + WORDCLOUD_STOPWORDS:
            text = re.sub(rf"\b{sw}\b", "", text)

        try:
            wc = WordCloud(
                width=600, height=400,
                background_color="white",
                colormap="viridis",
                max_words=40,
                min_font_size=8,
            ).generate(text)
            axes[idx].imshow(wc, interpolation="bilinear")
        except ValueError:
            axes[idx].text(0.5, 0.5, "Insufficient data", ha="center", va="center")

        axes[idx].set_title(theme_name, fontsize=12, fontweight="bold")
        axes[idx].axis("off")

    # Hide unused subplots
    for idx in range(n_themes, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Word Clouds by Theme", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_wordclouds_by_theme.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 14_wordclouds_by_theme.png")


def fig15_search_queries_table():
    """FIGURE 15: Search queries table for methodology slide."""
    queries = [
        ("Journal Sweep", "Core Trauma Journals", "JTACS, Injury, WJES, TSACO, Eur J Trauma, Scand J Trauma"),
        ("Journal Sweep", "General Surgery Journals", "Ann Surg, JAMA Surg, BJS, JACS, Am J Surg, Surgery, Can J Surg + trauma filter"),
        ("Journal Sweep", "Emergency Medicine", "Ann Emerg Med, Acad Emerg Med, Prehosp Emerg Care, CJEM + trauma filter"),
        ("Journal Sweep", "Critical Care", "Crit Care Med, ICM, Shock, Resuscitation, Transfusion + trauma filter"),
        ("Journal Sweep", "Neurotrauma & Ortho", "J Neurotrauma, Neurosurgery, J Ortho Trauma, J Surg Ed + trauma filter"),
        ("Broad Topic", "Broad Trauma Surgery", "\"trauma surgery\" OR \"acute care surgery\" OR \"emergency general surgery\""),
        ("Broad Topic", "Broad Trauma (Title)", "TITLE: trauma AND (surgery OR resuscitation OR hemorrhage OR damage control)"),
        ("Targeted Niche", "REBOA", "REBOA OR \"resuscitative endovascular balloon\" OR \"aortic occlusion\""),
        ("Targeted Niche", "AI / Machine Learning", "\"artificial intelligence\" OR \"machine learning\" OR \"deep learning\" + trauma"),
        ("Targeted Niche", "Blood Products", "\"whole blood\" OR \"freeze-dried plasma\" OR \"cold stored platelets\" + trauma"),
        ("Targeted Niche", "ECMO in Trauma", "ECMO OR ECLS OR \"extracorporeal\" + trauma"),
        ("Targeted Niche", "Teletrauma", "telemedicine OR teletrauma OR telementoring + trauma"),
        ("Targeted Niche", "Mass Casualty", "\"mass casualty\" OR \"mass shooting\" OR \"blast injury\" OR \"active shooter\""),
        ("Targeted Niche", "High-Impact Journals", "NEJM, JAMA, Lancet, BMJ + trauma filter"),
    ]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off")

    cell_text = [[q[0], q[1], q[2]] for q in queries]

    table = ax.table(
        cellText=cell_text,
        colLabels=["Search Type", "Category", "Search Strategy (Scopus)"],
        cellLoc="left",
        loc="center",
        colWidths=[0.12, 0.20, 0.59],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)

    for j in range(3):
        table[0, j].set_facecolor(COLORS["primary"])
        table[0, j].set_text_props(color="white", fontweight="bold")

    layer_colors = {"Journal Sweep": "#E8F4FD", "Broad Topic": "#FFF8E1", "Targeted Niche": "#F3E5F5"}
    for i, q in enumerate(queries, 1):
        color = layer_colors.get(q[0], "white")
        for j in range(3):
            table[i, j].set_facecolor(color)

    ax.set_title(f"3-Layer Scopus Search Strategy ({YEAR_MIN}-{YEAR_MAX}, {GEO_LABEL})",
                 fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "15_search_queries_table.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 15_search_queries_table.png")


def generate_summary_stats(df):
    """Print summary statistics for presentation narration."""
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS FOR PRESENTATION")
    print(f"{'='*70}")
    print(f"Total unique papers in curated corpus: {len(df)}")
    print(f"Date range: {df['year'].min()}–{df['year'].max()}")

    na_mask = df["affiliation_country"].fillna("").str.contains(GEO_PRIMARY_REGEX, case=False)
    print(f"{GEO_PRIMARY_LABEL} papers: {na_mask.sum()} ({na_mask.mean()*100:.1f}%)")
    if GEO_SECONDARY_REGEX:
        ca_mask = df["affiliation_country"].fillna("").str.contains(GEO_SECONDARY_REGEX, case=False)
        print(f"{GEO_SECONDARY_LABEL} papers: {ca_mask.sum()} ({ca_mask.mean()*100:.1f}%)")

    high_impact = (df["citations_count"] >= 20).sum()
    print(f"High-impact papers (≥20 cites): {high_impact}")

    recent_hot = ((df["year"] >= 2023) & (df["citations_count"] >= 5)).sum()
    print(f"Hot topics (2023+, ≥5 cites): {recent_hot}")

    cutting_edge = (df["year"] >= 2025).sum()
    print(f"Cutting-edge (2025–2026): {cutting_edge}")

    print(f"\nTop 5 journals:")
    for j, c in df["publication"].value_counts().head(5).items():
        print(f"  {c:>3} papers — {j}")

    concepts = extract_clinical_concepts(df["title"].tolist())
    print(f"\nTop 10 clinical concepts (by title frequency):")
    for concept, count in list(concepts.items())[:10]:
        pct = count / len(df) * 100
        print(f"  {count:>4} ({pct:.1f}%) — {concept}")

    print(f"\n{'─'*70}")
    print("METHODOLOGICAL LIMITATIONS (for methods section / reviewer response)")
    print(f"{'─'*70}")
    print(f"  1. Single database (Scopus). PubMed/WoS cross-validation recommended.")
    print(f"  2. Clinical concepts detected via title-based regex matching.")
    print(f"     Sensitivity analysis (title vs title+abstract) provided in stats module.")
    print(f"  3. {YEAR_MAX} data is partial (incomplete year). Excluded from")
    print(f"     statistical inference; shown in descriptive figures only.")
    print(f"  4. Concepts are not mutually exclusive (papers may match >1 concept).")
    print(f"  5. Exclusion/inclusion keywords are field-specific and empirically derived.")
    print(f"  6. No PROSPERO pre-registration. Code available for reproducibility.")
    print(f"  7. Citation counts not field-normalized (Field Citation Ratio not computed).")
    print(f"  8. No co-authorship network or bibliographic coupling analysis.")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    if not INPUT.exists():
        print(f"ERROR: Input file not found: {INPUT}")
        print(f"Run 'python filter_results.py' first to generate filtered results.")
        return
    if not RAW_INPUT.exists():
        print(f"WARNING: Raw input not found: {RAW_INPUT}")
        print(f"Flowchart numbers may be incomplete. Run search script first.")

    print("Loading data...")
    df = pd.read_csv(INPUT)
    df["title"] = df["title"].fillna("")
    df["publication"] = df["publication"].fillna("")
    df["affiliation_country"] = df["affiliation_country"].fillna("")
    df["citations_count"] = pd.to_numeric(df["citations_count"], errors="coerce").fillna(0).astype(int)

    print(f"Corpus: {len(df)} papers\n")
    print("Generating figures...")

    fig1_methodology_flowchart(df)
    fig2_query_categories_bar(df)
    fig3_publications_by_year(df)
    fig4_top_journals(df)
    fig5_geographic_distribution(df)
    fig6_wordcloud_all(df)
    fig7_clinical_concepts_heatmap(df)
    fig8_trending_concepts(df)
    fig9_top_bigrams(df)
    fig10_citation_vs_recency(df)
    fig11_concept_timeseries(df)
    fig12_na_vs_world(df)
    fig13_top_cited_table(df)
    fig14_wordcloud_by_theme(df)
    fig15_search_queries_table()

    generate_summary_stats(df)

    print(f"\n✅ All 15 figures saved to: {FIG_DIR}/")

    # Run statistical tests module (figures 19-23)
    print(f"\n{'='*70}")
    print("Running statistical analysis module...")
    print(f"{'='*70}")
    try:
        from statistical_tests import main as stats_main
        stats_main()
    except ImportError:
        print("  ⚠️ statistical_tests.py not found — skipping statistical analysis.")
        print("     Figures 19-23 will not be generated.")
    except Exception as e:
        print(f"  ⚠️ Statistical analysis failed: {e}")

    print(f"\nReady to drop into your presentation!")


if __name__ == "__main__":
    main()
