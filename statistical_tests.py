#!/usr/bin/env python3
"""
STATISTICAL ANALYSIS MODULE
============================
Adds statistical rigor to the bibliometric trend analysis.

Tests performed:
  1. Chi-squared / Fisher's exact tests for concept trend significance
  2. Benjamini-Hochberg FDR correction for multiple comparisons
  3. 95% confidence intervals for concept proportions (Wilson score)
  4. Effect sizes: odds ratios with 95% CI, Cramér's V
  5. Citation rate normalization (citations per year since publication)
  6. Mann-Whitney U tests for citation distribution shifts
  7. Cochran-Armitage trend test for monotonic year-over-year trends

Generates:
  - Fig 19: Forest plot (odds ratios with 95% CI for early vs late)
  - Fig 20: Volcano plot (effect size vs significance)
  - Fig 21: Normalized citation rate analysis by period
  - Fig 22: Statistical summary table
  - stats_report.csv: Full statistical results for all concepts

Usage:
  python statistical_tests.py
"""

import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from concept_definitions import CLINICAL_CONCEPTS
from config import (
    FIELD_NAME, FIELD_SHORT, YEAR_MIN, YEAR_MAX, YEAR_STATS_MAX,
    TREND_EARLY, TREND_LATE, GEO_PRIMARY_REGEX,
    GEO_PRIMARY_LABEL, FIG_DIR_NAME,
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────
INPUT = Path(__file__).parent / "results_curated" / "all_filtered.csv"
FIG_DIR = Path(__file__).parent / FIG_DIR_NAME
FIG_DIR.mkdir(exist_ok=True)
STATS_DIR = Path(__file__).parent / "stats_output"
STATS_DIR.mkdir(exist_ok=True)

# ── Style (matches analysis.py) ─────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.grid": True, "grid.alpha": 0.3,
})

COLORS = {
    "primary": "#1B4F72", "secondary": "#2E86C1", "accent": "#E74C3C",
    "warm": "#E67E22", "green": "#27AE60", "purple": "#8E44AD", "gray": "#7F8C8D",
}


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def wilson_ci(count, total, alpha=0.05):
    """Wilson score confidence interval for a binomial proportion.

    More accurate than the normal approximation for small samples
    and proportions near 0 or 1. Recommended by Agresti & Coull (1998).

    Returns (lower, upper) as proportions (0-1 scale).
    """
    if total == 0:
        return (0.0, 0.0)
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    p_hat = count / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return (max(0, center - spread), min(1, center + spread))


def odds_ratio_ci(a, b, c, d, alpha=0.05):
    """Odds ratio with 95% CI using the log method.

    2x2 table:
                  Late    Early
    Concept+   [  a        b  ]
    Concept-   [  c        d  ]

    Returns (OR, lower, upper). Adds 0.5 Haldane correction if any cell is 0.
    """
    # Haldane correction for zero cells
    if any(x == 0 for x in [a, b, c, d]):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    OR = (a * d) / (b * c)
    log_OR = np.log(OR)
    se_log_OR = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    lower = np.exp(log_OR - z * se_log_OR)
    upper = np.exp(log_OR + z * se_log_OR)
    return (OR, lower, upper)


def cramers_v(chi2, n, k=2, r=2):
    """Cramér's V effect size for chi-squared test.

    V ∈ [0, 1]. For 2x2 tables, V = phi coefficient.
    Guidelines (Cohen): small=0.1, medium=0.3, large=0.5
    """
    return np.sqrt(chi2 / (n * (min(k, r) - 1)))


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Returns:
        adjusted_p: array of BH-adjusted p-values
        significant: boolean array (True if significant after correction)
    """
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]

    # BH adjusted p-values
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / rank)
    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    result = np.zeros(n)
    result[sorted_idx] = adjusted

    return result, result < alpha


def cochran_armitage_trend(counts_per_year, totals_per_year):
    """Cochran-Armitage test for monotonic trend in proportions.

    Tests whether concept proportion changes systematically over years
    (not just early vs late, but year-by-year direction).

    Parameters:
        counts_per_year: array of concept counts per year
        totals_per_year: array of total papers per year

    Returns:
        z_stat: z-statistic (positive = increasing trend)
        p_value: two-sided p-value
    """
    counts = np.array(counts_per_year, dtype=float)
    totals = np.array(totals_per_year, dtype=float)
    k = len(counts)

    if k < 3 or np.all(totals == 0):
        return 0.0, 1.0

    # Scores (use integer year indices as scores)
    scores = np.arange(k, dtype=float)

    N = totals.sum()
    n_pos = counts.sum()
    n_neg = N - n_pos
    p_bar = n_pos / N

    # Weighted score means
    t_bar = np.sum(scores * totals) / N

    # Numerator
    numerator = np.sum(scores * counts) - n_pos * t_bar

    # Denominator
    denominator_sq = p_bar * (1 - p_bar) * (np.sum(scores**2 * totals) - N * t_bar**2)

    if denominator_sq <= 0:
        return 0.0, 1.0

    z = numerator / np.sqrt(denominator_sq)
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

    return z, p_value


# ═══════════════════════════════════════════════════════════════════════
# MAIN STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def run_trend_analysis(df):
    """Run full statistical analysis on concept trends.

    For each concept:
    1. Build 2x2 contingency table (concept yes/no × early/late)
    2. Chi-squared test (or Fisher's exact for small expected counts)
    3. Odds ratio with 95% CI
    4. Cramér's V effect size
    5. Cochran-Armitage trend test across all years

    Returns DataFrame with all results.
    """
    early = df[df["year"].between(TREND_EARLY[0], TREND_EARLY[1])]
    late = df[df["year"].between(TREND_LATE[0], TREND_LATE[1])]
    n_early = len(early)
    n_late = len(late)

    # Use only complete years for statistical inference
    df_stats = df[df["year"] <= YEAR_STATS_MAX].copy()
    years = sorted(df_stats["year"].unique())
    year_totals = [len(df_stats[df_stats["year"] == y]) for y in years]

    # Recompute early/late on stats-only data
    early = df_stats[df_stats["year"].between(TREND_EARLY[0], TREND_EARLY[1])]
    late = df_stats[df_stats["year"].between(TREND_LATE[0], TREND_LATE[1])]
    n_early = len(early)
    n_late = len(late)

    print(f"\n  Early period ({TREND_EARLY[0]}-{TREND_EARLY[1]}): {n_early:,} papers")
    print(f"  Late  period ({TREND_LATE[0]}-{TREND_LATE[1]}):  {n_late:,} papers")
    if YEAR_STATS_MAX < YEAR_MAX:
        n_excluded_partial = len(df[df["year"] > YEAR_STATS_MAX])
        print(f"  ⚠ {YEAR_MAX} excluded from inference ({n_excluded_partial:,} papers, partial year)")
    print(f"  Total concepts tested: {len(CLINICAL_CONCEPTS)}")
    print(f"  Significance threshold: α = 0.05 (BH-corrected for {len(CLINICAL_CONCEPTS)} tests)\n")

    results = []

    for concept, pattern in CLINICAL_CONCEPTS.items():
        try:
            early_match = early["title"].str.lower().str.contains(pattern, regex=True, na=False).sum()
            late_match = late["title"].str.lower().str.contains(pattern, regex=True, na=False).sum()
        except re.error:
            continue

        total_match = early_match + late_match

        # Skip concepts with too few papers for meaningful inference.
        # Threshold of 15 ensures stable odds ratios and adequate power
        # for detecting meaningful effect sizes (OR >= 2.0) after BH correction.
        if total_match < 15:
            results.append({
                "concept": concept,
                "early_count": early_match,
                "late_count": late_match,
                "total": total_match,
                "early_pct": early_match / n_early * 100 if n_early else 0,
                "late_pct": late_match / n_late * 100 if n_late else 0,
                "pct_change": 0,
                "test_used": "insufficient data",
                "chi2": np.nan, "p_value": 1.0, "p_adjusted": 1.0,
                "significant": False,
                "OR": np.nan, "OR_lower": np.nan, "OR_upper": np.nan,
                "cramers_v": np.nan,
                "trend_z": np.nan, "trend_p": np.nan, "trend_direction": "—",
            })
            continue

        # ── 2×2 contingency table ─────────────────────────────
        #                Late    Early
        # Concept+     [  a       b  ]
        # Concept−     [  c       d  ]
        a = late_match
        b = early_match
        c = n_late - late_match
        d = n_early - early_match

        table = np.array([[a, b], [c, d]])

        # ── Choose test: chi-squared or Fisher's exact ────────
        # Fisher's exact when any expected cell count < 5 (Cochran's rule)
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
        use_fisher = np.any(expected < 5)

        if use_fisher:
            # Fisher's exact test (2-sided)
            _, p_val = scipy_stats.fisher_exact(table, alternative="two-sided")
            chi2_val = np.nan
            test_name = "Fisher's exact"
        else:
            # Chi-squared test with Yates' continuity correction
            chi2_val, p_val, _, _ = scipy_stats.chi2_contingency(table, correction=True)
            test_name = "Chi-squared (Yates)"

        # ── Odds ratio with 95% CI ────────────────────────────
        OR, OR_lo, OR_hi = odds_ratio_ci(a, b, c, d)

        # ── Effect size ───────────────────────────────────────
        if not np.isnan(chi2_val):
            v = cramers_v(chi2_val, table.sum())
        else:
            # For Fisher's exact: compute phi coefficient directly from 2×2 table.
            # phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
            # This is equivalent to Cramér's V for 2×2 tables.
            n_total_table = a + b + c + d
            denom = np.sqrt((a+b) * (c+d) * (a+c) * (b+d))
            if denom > 0:
                v = abs((a*d - b*c)) / denom
            else:
                v = np.nan

        # ── Percent change (normalized) ───────────────────────
        early_pct = early_match / n_early * 100 if n_early else 0
        late_pct = late_match / n_late * 100 if n_late else 0
        if early_pct > 0:
            pct_change = ((late_pct - early_pct) / early_pct) * 100
        elif late_pct > 0:
            pct_change = 200.0  # new topic cap
        else:
            pct_change = 0.0

        # ── Cochran-Armitage trend test (year-over-year) ──────
        # IMPORTANT: Use df_stats (not df) to match year_totals denominator.
        # Both must use the same dataset (complete years only) to avoid
        # numerator/denominator mismatch on partial-year data.
        year_counts = [
            df_stats[df_stats["year"] == y]["title"].str.lower().str.contains(
                pattern, regex=True, na=False
            ).sum() for y in years
        ]
        trend_z, trend_p = cochran_armitage_trend(year_counts, year_totals)
        if trend_z > 0:
            trend_dir = "↑ increasing"
        elif trend_z < 0:
            trend_dir = "↓ decreasing"
        else:
            trend_dir = "— flat"

        results.append({
            "concept": concept,
            "early_count": early_match,
            "late_count": late_match,
            "total": total_match,
            "early_pct": round(early_pct, 2),
            "late_pct": round(late_pct, 2),
            "pct_change": round(pct_change, 1),
            "test_used": test_name,
            "chi2": round(chi2_val, 3) if not np.isnan(chi2_val) else np.nan,
            "p_value": p_val,
            "p_adjusted": np.nan,  # filled after BH correction
            "significant": False,  # filled after BH correction
            "OR": round(OR, 3),
            "OR_lower": round(OR_lo, 3),
            "OR_upper": round(OR_hi, 3),
            "cramers_v": round(v, 4) if not np.isnan(v) else np.nan,
            "trend_z": round(trend_z, 3),
            "trend_p": trend_p,
            "trend_direction": trend_dir,
        })

    results_df = pd.DataFrame(results)

    # ── Unified Benjamini-Hochberg FDR correction ─────────────
    # We apply a SINGLE FDR correction across ALL p-values from both
    # the chi-squared/Fisher's test AND the Cochran-Armitage trend test.
    # This is the conservative approach — treating both test families
    # as a single hypothesis family (2 × N_concepts tests total).
    # Rationale: Both test the same conceptual question ("is this concept
    # changing?") from different angles, so unified FDR is appropriate.
    valid_mask = results_df["test_used"] != "insufficient data"

    results_df["trend_p_adjusted"] = np.nan
    results_df["trend_significant"] = False

    if valid_mask.any():
        # Collect all p-values into one pool
        chi_p = results_df.loc[valid_mask, "p_value"].values
        trend_p = results_df.loc[valid_mask, "trend_p"].values

        all_p = np.concatenate([chi_p, trend_p])
        all_adjusted, all_sig = benjamini_hochberg(all_p, alpha=0.05)

        n_valid = valid_mask.sum()
        # Split back into chi-squared and trend results
        results_df.loc[valid_mask, "p_adjusted"] = all_adjusted[:n_valid]
        results_df.loc[valid_mask, "significant"] = all_sig[:n_valid]
        results_df.loc[valid_mask, "trend_p_adjusted"] = all_adjusted[n_valid:]
        results_df.loc[valid_mask, "trend_significant"] = all_sig[n_valid:]

    # Sort by effect size (odds ratio distance from 1)
    results_df["abs_log_OR"] = np.abs(np.log(results_df["OR"].replace(0, np.nan)))
    results_df = results_df.sort_values("abs_log_OR", ascending=False)

    return results_df


def run_citation_analysis(df):
    """Normalized citation analysis.

    Computes citation rate = citations / years_since_publication
    to remove age bias from citation comparisons.

    Also runs Mann-Whitney U test comparing early vs late period
    citation rates.
    """
    # Use YEAR_STATS_MAX as reference point for citation rate normalization.
    # Citation rate = citations / (YEAR_STATS_MAX - pub_year + 1)
    # +1 makes the denominator inclusive (2025 paper published in 2025 → 1 year of exposure).
    # Papers from partial years (>YEAR_STATS_MAX) are excluded from inference.
    df = df.copy()
    df_stats = df[df["year"] <= YEAR_STATS_MAX].copy()
    df_stats["years_since_pub"] = YEAR_STATS_MAX - df_stats["year"] + 1
    df_stats["citation_rate"] = df_stats["citations_count"] / df_stats["years_since_pub"]

    # Also compute for full df (for figure only)
    df["years_since_pub"] = YEAR_STATS_MAX - df["year"] + 1
    df.loc[df["years_since_pub"] < 1, "years_since_pub"] = 0.5  # partial year fallback
    df["citation_rate"] = df["citations_count"] / df["years_since_pub"]

    early = df_stats[df_stats["year"].between(TREND_EARLY[0], TREND_EARLY[1])]
    late = df_stats[df_stats["year"].between(TREND_LATE[0], TREND_LATE[1])]

    # Mann-Whitney U test: are citation rates different between periods?
    stat, p_val = scipy_stats.mannwhitneyu(
        early["citation_rate"].values,
        late["citation_rate"].values,
        alternative="two-sided"
    )

    # Effect size: rank-biserial correlation
    n1, n2 = len(early), len(late)
    r = 1 - (2 * stat) / (n1 * n2)  # rank-biserial

    summary = {
        "early_median_raw": early["citations_count"].median(),
        "late_median_raw": late["citations_count"].median(),
        "early_median_rate": round(early["citation_rate"].median(), 2),
        "late_median_rate": round(late["citation_rate"].median(), 2),
        "early_mean_rate": round(early["citation_rate"].mean(), 2),
        "late_mean_rate": round(late["citation_rate"].mean(), 2),
        "mann_whitney_U": stat,
        "mann_whitney_p": p_val,
        "rank_biserial_r": round(r, 4),
        "interpretation": (
            "Significant difference" if p_val < 0.05
            else "No significant difference"
        ),
    }

    return df, summary


# ═══════════════════════════════════════════════════════════════════════
# FIGURE GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def fig19_forest_plot(results_df):
    """FIGURE 19: Forest plot — odds ratios with 95% CI.

    Shows which concepts genuinely shifted between periods,
    with multiple-comparison correction. This is the key figure
    for defensible trend claims.
    """
    # Filter to concepts with enough data and sort by OR
    plot_df = results_df[results_df["test_used"] != "insufficient data"].copy()
    plot_df = plot_df.sort_values("OR", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.35)))

    y_positions = range(len(plot_df))

    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = COLORS["green"] if row["significant"] else COLORS["gray"]
        marker_size = 8 if row["significant"] else 5

        # Horizontal error bar (CI)
        ax.errorbar(
            row["OR"], i,
            xerr=[[row["OR"] - row["OR_lower"]], [row["OR_upper"] - row["OR"]]],
            fmt="o", color=color, markersize=marker_size,
            capsize=3, capthick=1.5, elinewidth=1.5,
            markeredgecolor="white", markeredgewidth=0.5,
        )

        # Significance marker
        if row["significant"]:
            label = f'p={row["p_adjusted"]:.4f}' if row["p_adjusted"] >= 0.0001 else "p<0.0001"
            ax.text(
                max(row["OR_upper"] + 0.05, row["OR"] + 0.3), i,
                f'★ {label}',
                va="center", fontsize=7, color=COLORS["accent"], fontweight="bold",
            )

    # Reference line at OR=1 (no difference)
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1, alpha=0.7, zorder=0)
    ax.text(1.0, len(plot_df) + 0.5, "OR = 1.0\n(no change)", ha="center",
            fontsize=8, color="black", alpha=0.6)

    # Labels
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(
        [f'{row["concept"]} (n={row["total"]})' for _, row in plot_df.iterrows()],
        fontsize=9,
    )
    ax.set_xlabel("Odds Ratio (late vs early period)", fontsize=12)
    ax.set_title(
        f"Concept Trend Significance: {TREND_EARLY[0]}-{TREND_EARLY[1]} vs "
        f"{TREND_LATE[0]}-{TREND_LATE[1]}\n"
        f"(OR > 1 = more frequent in late period | "
        f"★ = significant after BH correction, α=0.05)",
        fontsize=13, fontweight="bold",
    )

    # Set x-axis to log scale for better OR visualization
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xlim(left=max(0.1, plot_df["OR_lower"].min() * 0.7),
                right=min(10, plot_df["OR_upper"].max() * 1.5))

    # Shade left/right halves
    ax.axvspan(ax.get_xlim()[0], 1.0, alpha=0.04, color=COLORS["accent"], zorder=0)
    ax.axvspan(1.0, ax.get_xlim()[1], alpha=0.04, color=COLORS["green"], zorder=0)
    ax.text(0.15, -1.5, "← More in early period", fontsize=9, color=COLORS["accent"],
            ha="left", style="italic", transform=ax.get_xaxis_transform())
    ax.text(0.85, -1.5, "More in late period →", fontsize=9, color=COLORS["green"],
            ha="right", style="italic", transform=ax.get_xaxis_transform())

    n_sig = plot_df["significant"].sum()
    n_total = len(plot_df)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["green"],
               markersize=8, label=f"Significant (n={n_sig})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["gray"],
               markersize=5, label=f"Not significant (n={n_total - n_sig})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              framealpha=0.9, edgecolor="gray")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "19_forest_plot_OR.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 19_forest_plot_OR.png")


def fig20_volcano_plot(results_df):
    """FIGURE 20: Volcano plot — effect size vs statistical significance.

    X-axis: log2(OR) — direction and magnitude of change
    Y-axis: -log10(adjusted p-value) — statistical significance
    Threshold line: BH-corrected α = 0.05
    """
    plot_df = results_df[results_df["test_used"] != "insufficient data"].copy()
    plot_df = plot_df[plot_df["OR"] > 0].copy()

    plot_df["log2_OR"] = np.log2(plot_df["OR"])
    plot_df["neg_log10_p"] = -np.log10(plot_df["p_adjusted"].clip(lower=1e-20))

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by significance and direction
    for _, row in plot_df.iterrows():
        if row["significant"] and row["log2_OR"] > 0:
            color = COLORS["green"]
            alpha = 0.9
        elif row["significant"] and row["log2_OR"] < 0:
            color = COLORS["accent"]
            alpha = 0.9
        else:
            color = COLORS["gray"]
            alpha = 0.5

        size = max(30, min(200, row["total"] * 2))
        ax.scatter(row["log2_OR"], row["neg_log10_p"],
                   s=size, c=color, alpha=alpha, edgecolors="white", linewidth=0.5)

        # Label significant concepts
        if row["significant"]:
            ax.annotate(
                row["concept"],
                xy=(row["log2_OR"], row["neg_log10_p"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.8, edgecolor="none"),
            )

    # Significance threshold line
    sig_threshold = -np.log10(0.05)
    ax.axhline(y=sig_threshold, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.9, sig_threshold + 0.15,
            "BH-adjusted α = 0.05", fontsize=8, ha="right", alpha=0.6)

    # Center line
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("log₂(Odds Ratio) — Effect Size", fontsize=12)
    ax.set_ylabel("-log₁₀(adjusted p-value) — Significance", fontsize=12)
    ax.set_title(
        f"Volcano Plot: Concept Trend Changes\n"
        f"({TREND_EARLY[0]}-{TREND_EARLY[1]} vs {TREND_LATE[0]}-{TREND_LATE[1]} | "
        f"bubble size ∝ total papers)",
        fontsize=13, fontweight="bold",
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["green"], label="Significant increase"),
        Patch(facecolor=COLORS["accent"], label="Significant decrease"),
        Patch(facecolor=COLORS["gray"], label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "20_volcano_plot.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 20_volcano_plot.png")


def fig21_citation_rate(df, cite_summary):
    """FIGURE 21: Citation rate analysis (normalized for publication age).

    Shows citation rate (cites/year) distributions by publication period,
    removing the systematic bias where older papers accumulate more citations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Box plot of citation rate by period
    early = df[df["year"].between(TREND_EARLY[0], TREND_EARLY[1])].copy()
    late = df[df["year"].between(TREND_LATE[0], TREND_LATE[1])].copy()

    # Cap extreme outliers for visualization (keep in analysis)
    cap = df["citation_rate"].quantile(0.99)

    data_to_plot = [
        early["citation_rate"].clip(upper=cap).values,
        late["citation_rate"].clip(upper=cap).values,
    ]
    labels = [
        f"Early ({TREND_EARLY[0]}-{TREND_EARLY[1]})\n"
        f"n={len(early):,}, median={cite_summary['early_median_rate']}/yr",
        f"Late ({TREND_LATE[0]}-{TREND_LATE[1]})\n"
        f"n={len(late):,}, median={cite_summary['late_median_rate']}/yr",
    ]

    bp = axes[0].boxplot(data_to_plot, labels=labels, patch_artist=True,
                          showfliers=False, widths=0.5)
    bp["boxes"][0].set_facecolor(COLORS["secondary"])
    bp["boxes"][1].set_facecolor(COLORS["accent"])
    for box in bp["boxes"]:
        box.set_alpha(0.7)

    axes[0].set_ylabel("Citation Rate (citations / year since publication)", fontsize=11)
    axes[0].set_title("A. Citation Rate by Period", fontsize=12, fontweight="bold")

    # Add Mann-Whitney result
    mw_text = (
        f"Mann-Whitney U = {cite_summary['mann_whitney_U']:,.0f}\n"
        f"p = {cite_summary['mann_whitney_p']:.4f}\n"
        f"r = {cite_summary['rank_biserial_r']:.3f}"
    )
    axes[0].text(0.95, 0.95, mw_text, transform=axes[0].transAxes,
                  fontsize=9, verticalalignment="top", horizontalalignment="right",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel B: Citation rate by year (median + IQR)
    yearly_stats = df.groupby("year")["citation_rate"].agg(["median", "mean",
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
    ])
    yearly_stats.columns = ["median", "mean", "q25", "q75"]

    years = yearly_stats.index.values
    axes[1].fill_between(years, yearly_stats["q25"], yearly_stats["q75"],
                          alpha=0.2, color=COLORS["secondary"], label="IQR")
    axes[1].plot(years, yearly_stats["median"], "o-",
                  color=COLORS["primary"], linewidth=2.5, markersize=7, label="Median")
    axes[1].plot(years, yearly_stats["mean"], "s--",
                  color=COLORS["accent"], linewidth=1.5, markersize=5, alpha=0.7, label="Mean")

    axes[1].set_xlabel("Publication Year", fontsize=11)
    axes[1].set_ylabel("Citation Rate (citations / year)", fontsize=11)
    axes[1].set_title("B. Citation Rate by Publication Year", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].set_xticks(years)
    axes[1].set_xticklabels([str(y) for y in years])

    # Annotate 2026 as partial
    if 2026 in years:
        axes[1].axvspan(2025.5, 2026.5, alpha=0.1, color="orange")
        axes[1].text(2026, axes[1].get_ylim()[1] * 0.9, "partial\nyear",
                      ha="center", fontsize=8, alpha=0.6)

    fig.suptitle(
        f"Citation Analysis (Age-Normalized) — {FIELD_NAME}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "21_citation_rate_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 21_citation_rate_analysis.png")


def fig22_stats_summary_table(results_df, cite_summary):
    """FIGURE 22: Summary table of statistical results for presentation."""
    # Select significant concepts + top non-significant for context
    sig = results_df[results_df["significant"]].copy()
    nonsig = results_df[~results_df["significant"] & (results_df["test_used"] != "insufficient data")]
    nonsig = nonsig.head(5)
    display_df = pd.concat([sig, nonsig]).copy()
    display_df = display_df.sort_values("OR", ascending=False)

    # Format for table
    rows = []
    for _, r in display_df.iterrows():
        p_str = f'{r["p_adjusted"]:.4f}' if r["p_adjusted"] >= 0.0001 else "<0.0001"
        or_str = f'{r["OR"]:.2f} ({r["OR_lower"]:.2f}-{r["OR_upper"]:.2f})'
        sig_str = "★ Yes" if r["significant"] else "No"
        trend_str = r["trend_direction"].split(" ")[0]  # just the arrow

        rows.append([
            r["concept"],
            f'{r["early_count"]}/{r["late_count"]}',
            f'{r["early_pct"]:.1f}% → {r["late_pct"]:.1f}%',
            or_str,
            p_str,
            sig_str,
            trend_str,
        ])

    col_labels = ["Concept", "Early/Late\nCounts", "Share\n(% of period)",
                   "OR (95% CI)", "p-adj\n(BH)", "Sig?", "Trend"]

    fig, ax = plt.subplots(figsize=(16, max(5, len(rows) * 0.5 + 2)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.17, 0.09, 0.13, 0.18, 0.09, 0.06, 0.06],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor(COLORS["primary"])
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8)

    # Color rows by significance
    for i in range(1, len(rows) + 1):
        if rows[i-1][5] == "★ Yes":
            bg_color = "#E8F8E8"  # light green
        else:
            bg_color = "#F5F5F5"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg_color)

    # Title with methodology note
    n_sig = (results_df["significant"]).sum()
    n_tested = len(results_df[results_df["test_used"] != "insufficient data"])
    ax.set_title(
        f"Statistical Summary: Concept Trend Analysis\n"
        f"({n_sig}/{n_tested} concepts significant after Benjamini-Hochberg correction | "
        f"Chi-squared/Fisher's exact | α=0.05)",
        fontsize=12, fontweight="bold", pad=20,
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "22_stats_summary_table.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 22_stats_summary_table.png")


def fig23_concept_ci_plot(results_df, df):
    """FIGURE 23: Concept proportions with Wilson 95% CIs.

    For each concept, shows the proportion of papers matching that
    concept in the full corpus, with Wilson score confidence intervals.
    This directly shows the precision of our estimates.
    """
    n_total = len(df)

    # Get all concepts with data, sorted by proportion
    plot_df = results_df[results_df["test_used"] != "insufficient data"].copy()
    plot_df["full_pct"] = plot_df["total"] / n_total * 100
    plot_df = plot_df.sort_values("full_pct", ascending=True)

    # Compute Wilson CIs
    cis = [wilson_ci(int(row["total"]), n_total) for _, row in plot_df.iterrows()]
    plot_df["ci_lower"] = [c[0] * 100 for c in cis]
    plot_df["ci_upper"] = [c[1] * 100 for c in cis]

    fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.32)))

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.errorbar(
            row["full_pct"], i,
            xerr=[[row["full_pct"] - row["ci_lower"]],
                  [row["ci_upper"] - row["full_pct"]]],
            fmt="o", color=COLORS["primary"], markersize=6,
            capsize=3, capthick=1, elinewidth=1.2,
            markeredgecolor="white", markeredgewidth=0.5,
        )
        # Label count
        ax.text(row["ci_upper"] + 0.15, i,
                f'n={int(row["total"])}',
                va="center", fontsize=7, color=COLORS["gray"])

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["concept"].values, fontsize=9)
    ax.set_xlabel("Proportion of Corpus (%) with 95% Wilson CI", fontsize=12)
    ax.set_title(
        f"Concept Prevalence with 95% Confidence Intervals\n"
        f"(n = {n_total:,} papers, Wilson score intervals)",
        fontsize=13, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "23_concept_confidence_intervals.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 23_concept_confidence_intervals.png")


def run_sensitivity_analysis(df):
    """Sensitivity analysis: compare title-only vs title+abstract matching.

    For each concept, reports both detection rates and the ratio.
    This directly addresses reviewer concern #1: "title-only detection
    underestimates concept prevalence."

    Prints a comparison table and saves to CSV.
    """
    titles = df["title"].fillna("").str.lower()
    abstracts = df["abstract"].fillna("").str.lower()
    combined = titles + " " + abstracts

    rows = []
    for concept, pattern in CLINICAL_CONCEPTS.items():
        try:
            n_title = titles.str.contains(pattern, regex=True, na=False).sum()
            n_combined = combined.str.contains(pattern, regex=True, na=False).sum()
        except re.error:
            continue

        ratio = n_combined / n_title if n_title > 0 else float("inf")
        rows.append({
            "concept": concept,
            "title_only": n_title,
            "title_abstract": n_combined,
            "ratio": round(ratio, 2),
            "additional_from_abstract": n_combined - n_title,
        })

    sens_df = pd.DataFrame(rows).sort_values("ratio", ascending=False)

    # Print summary
    mean_ratio = sens_df["ratio"].replace([float("inf")], np.nan).mean()
    print(f"\n  Mean detection ratio (title+abstract / title-only): {mean_ratio:.2f}x")
    print(f"  Concepts where abstract adds >50% more papers:")
    high = sens_df[sens_df["ratio"] > 1.5]
    for _, r in high.iterrows():
        print(f'    {r["concept"]:30s}  title={r["title_only"]:>4}  '
              f'title+abs={r["title_abstract"]:>4}  '
              f'ratio={r["ratio"]:.1f}x (+{r["additional_from_abstract"]})')

    if len(high) == 0:
        print("    (none — title-only detection is adequate for all concepts)")

    # Save
    sens_df.to_csv(STATS_DIR / "sensitivity_title_vs_abstract.csv", index=False)
    print(f"  📊 Saved to: {STATS_DIR / 'sensitivity_title_vs_abstract.csv'}")

    # Generate figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(sens_df) * 0.32)))
    y = range(len(sens_df))

    ax.barh(y, sens_df["title_only"].values, height=0.4, label="Title only",
            color=COLORS["primary"], alpha=0.8)
    ax.barh([i + 0.4 for i in y], sens_df["title_abstract"].values, height=0.4,
            label="Title + Abstract", color=COLORS["warm"], alpha=0.8)

    ax.set_yticks([i + 0.2 for i in y])
    ax.set_yticklabels(sens_df["concept"].values, fontsize=8)
    ax.set_xlabel("Number of Papers Matching", fontsize=12)
    ax.set_title(
        "Sensitivity Analysis: Title-Only vs Title+Abstract Concept Detection\n"
        f"(n = {len(df):,} papers | mean detection ratio: {mean_ratio:.2f}x)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "25_sensitivity_title_vs_abstract.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✅ 25_sensitivity_title_vs_abstract.png")

    return sens_df


def fig24_concept_cooccurrence(df):
    """FIGURE 24: Concept co-occurrence matrix.

    Shows which concepts frequently appear together in the same paper.
    This directly addresses the concept overlap issue — reviewers need
    to know which concepts are correlated, because overlapping concepts
    violate the independence assumption of per-concept testing.
    """
    titles = df["title"].str.lower()
    top_concepts = []
    concept_vectors = {}

    # Get top 15 concepts by frequency for readability
    for concept, pattern in CLINICAL_CONCEPTS.items():
        try:
            vec = titles.str.contains(pattern, regex=True, na=False)
            concept_vectors[concept] = vec
        except re.error:
            continue

    # Sort by frequency, take top 15
    sorted_concepts = sorted(concept_vectors.keys(),
                             key=lambda c: concept_vectors[c].sum(), reverse=True)[:15]

    # Build co-occurrence matrix
    n = len(sorted_concepts)
    matrix = np.zeros((n, n), dtype=int)
    for i, c1 in enumerate(sorted_concepts):
        for j, c2 in enumerate(sorted_concepts):
            if i == j:
                matrix[i][j] = concept_vectors[c1].sum()
            else:
                matrix[i][j] = (concept_vectors[c1] & concept_vectors[c2]).sum()

    # Convert to percentages (Jaccard-like: overlap / union)
    pct_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                pct_matrix[i][j] = 100.0  # diagonal = self-match
            else:
                union = (concept_vectors[sorted_concepts[i]] |
                         concept_vectors[sorted_concepts[j]]).sum()
                if union > 0:
                    pct_matrix[i][j] = matrix[i][j] / union * 100

    fig, ax = plt.subplots(figsize=(14, 12))

    # Use raw counts for annotation but percentage for color
    import seaborn as sns
    mask = np.zeros_like(pct_matrix, dtype=bool)
    np.fill_diagonal(mask, True)  # mask diagonal

    sns.heatmap(
        pct_matrix, annot=matrix, fmt="d", cmap="YlOrRd",
        xticklabels=sorted_concepts, yticklabels=sorted_concepts,
        linewidths=0.5, linecolor="white", ax=ax, mask=mask,
        cbar_kws={"label": "Jaccard Overlap (%)"},
        vmin=0, vmax=20,
    )

    # Add diagonal counts
    for i in range(n):
        ax.text(i + 0.5, i + 0.5, str(matrix[i][i]),
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round", facecolor=COLORS["primary"], alpha=0.8))

    ax.set_title(
        "Concept Co-Occurrence Matrix (Top 15 by Frequency)\n"
        "Diagonal = concept total | Off-diagonal = papers matching both | "
        "Color = Jaccard overlap %",
        fontsize=12, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "24_concept_cooccurrence.png", bbox_inches="tight")
    plt.close(fig)

    # Report overlap stats
    overlap_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] > 0:
                overlap_pairs.append((sorted_concepts[i], sorted_concepts[j],
                                       matrix[i][j], round(pct_matrix[i][j], 1)))

    overlap_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Top concept overlaps (papers matching both):")
    for c1, c2, count, pct in overlap_pairs[:10]:
        print(f"    {c1:25s} × {c2:25s} = {count:>3} papers ({pct:.1f}% Jaccard)")

    print("  ✅ 24_concept_cooccurrence.png")


# ═══════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════

def print_report(results_df, cite_summary):
    """Print a human-readable statistical report to console."""
    print(f"\n{'=' * 70}")
    print("STATISTICAL REPORT")
    print(f"{'=' * 70}")

    n_tested = len(results_df[results_df["test_used"] != "insufficient data"])
    n_sig = results_df["significant"].sum()
    n_insuf = (results_df["test_used"] == "insufficient data").sum()

    print(f"\n  Concepts analyzed: {len(results_df)}")
    print(f"  Concepts with sufficient data: {n_tested}")
    print(f"  Concepts with insufficient data (<15 papers): {n_insuf}")
    print(f"  Partial year ({YEAR_MAX}) excluded from inference: Yes")
    print(f"  Statistical tests on complete years only: {TREND_EARLY[0]}-{TREND_LATE[1]}")
    n_total_tests = n_tested * 2  # chi-squared + Cochran-Armitage per concept
    print(f"  Multiple comparison correction: Unified BH FDR (α=0.05, {n_total_tests} tests)")
    print(f"  Significant trend changes: {n_sig}/{n_tested}")

    # ── Significant results ────────────────────────────────────────
    sig = results_df[results_df["significant"]].copy()
    if len(sig) > 0:
        print(f"\n  {'─' * 60}")
        print(f"  SIGNIFICANT TREND CHANGES (BH-corrected p < 0.05)")
        print(f"  {'─' * 60}")

        increasing = sig[sig["OR"] > 1].sort_values("OR", ascending=False)
        decreasing = sig[sig["OR"] < 1].sort_values("OR", ascending=True)

        if len(increasing) > 0:
            print(f"\n  ▲ INCREASING in late period:")
            for _, r in increasing.iterrows():
                p_str = f'{r["p_adjusted"]:.4f}' if r["p_adjusted"] >= 0.0001 else "<0.0001"
                print(f'    {r["concept"]:30s}  OR={r["OR"]:.2f} '
                      f'({r["OR_lower"]:.2f}-{r["OR_upper"]:.2f})  '
                      f'p_adj={p_str}  '
                      f'({r["early_pct"]:.1f}% → {r["late_pct"]:.1f}%)')

        if len(decreasing) > 0:
            print(f"\n  ▼ DECREASING in late period:")
            for _, r in decreasing.iterrows():
                p_str = f'{r["p_adjusted"]:.4f}' if r["p_adjusted"] >= 0.0001 else "<0.0001"
                print(f'    {r["concept"]:30s}  OR={r["OR"]:.2f} '
                      f'({r["OR_lower"]:.2f}-{r["OR_upper"]:.2f})  '
                      f'p_adj={p_str}  '
                      f'({r["early_pct"]:.1f}% → {r["late_pct"]:.1f}%)')

    # ── Cochran-Armitage results ───────────────────────────────────
    if "trend_significant" in results_df.columns:
        trend_sig = results_df[results_df["trend_significant"] == True]
        if len(trend_sig) > 0:
            print(f"\n  {'─' * 60}")
            print(f"  MONOTONIC YEAR-OVER-YEAR TRENDS (Cochran-Armitage, BH-corrected)")
            print(f"  {'─' * 60}")
            for _, r in trend_sig.iterrows():
                p_str = f'{r["trend_p_adjusted"]:.4f}' if r["trend_p_adjusted"] >= 0.0001 else "<0.0001"
                print(f'    {r["concept"]:30s}  z={r["trend_z"]:+.3f}  '
                      f'p_adj={p_str}  {r["trend_direction"]}')

    # ── Citation analysis ──────────────────────────────────────────
    print(f"\n  {'─' * 60}")
    print(f"  CITATION RATE ANALYSIS (age-normalized)")
    print(f"  {'─' * 60}")
    print(f"  Early period median rate: {cite_summary['early_median_rate']} cites/year")
    print(f"  Late  period median rate: {cite_summary['late_median_rate']} cites/year")
    print(f"  Early period raw median:  {cite_summary['early_median_raw']} citations")
    print(f"  Late  period raw median:  {cite_summary['late_median_raw']} citations")
    print(f"  Mann-Whitney U: {cite_summary['mann_whitney_U']:,.0f}")
    print(f"  p-value: {cite_summary['mann_whitney_p']:.6f}")
    print(f"  Rank-biserial r: {cite_summary['rank_biserial_r']}")
    print(f"  → {cite_summary['interpretation']}")

    # ── What you can now claim ─────────────────────────────────────
    print(f"\n  {'─' * 60}")
    print(f"  DEFENSIBLE CLAIMS (based on statistical evidence)")
    print(f"  {'─' * 60}")
    if len(sig) > 0:
        for _, r in sig.iterrows():
            direction = "increased" if r["OR"] > 1 else "decreased"
            p_str = f'{r["p_adjusted"]:.4f}' if r["p_adjusted"] >= 0.0001 else "<0.0001"
            print(f'  ✅ "{r["concept"]} significantly {direction} from '
                  f'{r["early_pct"]:.1f}% to {r["late_pct"]:.1f}% of corpus '
                  f'(OR={r["OR"]:.2f}, p={p_str}, BH-corrected)"')
    else:
        print("  ⚠️ No statistically significant trends after multiple comparison correction.")
        print("     Any apparent changes may be due to random variation.")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    if not INPUT.exists():
        print(f"ERROR: Input file not found: {INPUT}")
        print(f"Run 'python filter_results.py' first.")
        return

    print("=" * 70)
    print("STATISTICAL ANALYSIS MODULE")
    print("=" * 70)
    print(f"  Field: {FIELD_NAME}")
    print(f"  Period comparison: {TREND_EARLY[0]}-{TREND_EARLY[1]} vs {TREND_LATE[0]}-{TREND_LATE[1]}")

    df = pd.read_csv(INPUT)
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["citations_count"] = pd.to_numeric(df["citations_count"], errors="coerce").fillna(0).astype(int)
    df["year"] = df["year"].astype(int)
    print(f"  Corpus: {len(df):,} papers")

    # ── 1. Trend analysis ─────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("RUNNING TREND ANALYSIS...")
    print("─" * 70)
    results_df = run_trend_analysis(df)

    # ── 2. Citation analysis ──────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("RUNNING CITATION ANALYSIS...")
    print("─" * 70)
    df_with_rates, cite_summary = run_citation_analysis(df)

    # ── 3. Generate figures ───────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("GENERATING STATISTICAL FIGURES...")
    print("─" * 70)
    fig19_forest_plot(results_df)
    fig20_volcano_plot(results_df)
    fig21_citation_rate(df_with_rates, cite_summary)
    fig22_stats_summary_table(results_df, cite_summary)
    fig23_concept_ci_plot(results_df, df)

    # ── 3b. Sensitivity analysis: title vs title+abstract ────────
    print(f"\n{'─' * 70}")
    print("SENSITIVITY ANALYSIS: Title-only vs Title+Abstract matching...")
    print("─" * 70)
    sensitivity_df = run_sensitivity_analysis(df)

    # ── 3c. Concept co-occurrence matrix ──────────────────────────
    print(f"\n{'─' * 70}")
    print("CONCEPT CO-OCCURRENCE ANALYSIS...")
    print("─" * 70)
    fig24_concept_cooccurrence(df)

    # ── 4. Save full results ──────────────────────────────────────
    output_cols = [
        "concept", "early_count", "late_count", "total",
        "early_pct", "late_pct", "pct_change",
        "test_used", "chi2", "p_value", "p_adjusted", "significant",
        "OR", "OR_lower", "OR_upper", "cramers_v",
        "trend_z", "trend_p", "trend_direction",
    ]
    if "trend_p_adjusted" in results_df.columns:
        output_cols.extend(["trend_p_adjusted", "trend_significant"])

    save_df = results_df[[c for c in output_cols if c in results_df.columns]].copy()
    save_df.to_csv(STATS_DIR / "stats_report.csv", index=False)
    print(f"\n  📊 Full results saved to: {STATS_DIR / 'stats_report.csv'}")

    # ── 5. Print report ───────────────────────────────────────────
    print_report(results_df, cite_summary)

    print(f"\n{'=' * 70}")
    print("STATISTICAL ANALYSIS COMPLETE")
    print(f"  Figures: 19-23 saved to {FIG_DIR}/")
    print(f"  Data:    {STATS_DIR / 'stats_report.csv'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
