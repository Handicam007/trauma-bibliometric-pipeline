#!/usr/bin/env python3
"""
Comprehensive line chart: all major clinical concepts over time (2020–2026).
Uses canonical patterns from concept_definitions.py (single source of truth).
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from concept_definitions import CLINICAL_CONCEPTS, DOMAIN_GROUPS
from config import FIELD_NAME, FIELD_SHORT, GEO_LABEL, FIG_DIR_NAME, YEAR_MIN, YEAR_MAX

INPUT = Path(__file__).parent / "results_curated" / "all_filtered.csv"
FIG_DIR = Path(__file__).parent / FIG_DIR_NAME

# Use canonical patterns
CONCEPTS = CLINICAL_CONCEPTS

# ── Color assignments (distinct, colorblind-friendly-ish) ───────
COLORS = [
    "#E74C3C", "#2E86C1", "#E67E22", "#27AE60", "#8E44AD",
    "#1ABC9C", "#D35400", "#2C3E50", "#F39C12", "#7D3C98",
    "#16A085", "#C0392B", "#2980B9", "#D4AC0D", "#1B4F72",
    "#A93226", "#117A65", "#6C3483", "#CA6F1E", "#1A5276",
    "#28B463", "#CB4335", "#884EA0", "#229954", "#5B2C6F",
    "#148F77", "#B03A2E", "#1F618D", "#B7950B", "#0E6655",
    "#633974", "#D68910", "#154360", "#7B241C", "#0B5345",
    "#76448A", "#C27C0E", "#1A5276", "#922B21", "#0E6251",
    "#6C3483", "#D4AC0D",
]

LINE_STYLES = ["-", "--", "-.", ":"]


def annotate_partial_year(ax, years):
    """Add a shaded region and note for the partial 2026 data."""
    if 2026 in years:
        ax.axvspan(2025.5, 2026.5, alpha=0.08, color="gray", zorder=0)
        ax.annotate("2026\n(partial year)",
                     xy=(2026, 0), xytext=(2026, ax.get_ylim()[1] * 0.92),
                     fontsize=8, color="gray", ha="center", fontstyle="italic")


def main():
    df = pd.read_csv(INPUT)
    df["title"] = df["title"].fillna("").str.lower()
    years = sorted(df["year"].unique())

    # ── Count each concept per year ─────────────────────────────
    data = {}
    totals = {}
    for concept, pattern in CONCEPTS.items():
        counts = []
        total = 0
        for y in years:
            titles_y = df[df["year"] == y]["title"]
            c = titles_y.str.contains(pattern, regex=True, na=False).sum()
            counts.append(c)
            total += c
        data[concept] = counts
        totals[concept] = total

    # Sort by total frequency descending
    sorted_concepts = sorted(totals, key=totals.get, reverse=True)

    # ═══════════════════════════════════════════════════════════
    # FIGURE A: ALL concepts, single large chart
    # ═══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(16, 10))

    for i, concept in enumerate(sorted_concepts):
        counts = data[concept]
        color = COLORS[i % len(COLORS)]
        ls = LINE_STYLES[i // len(COLORS) % len(LINE_STYLES)]
        lw = 2.8 if totals[concept] >= 50 else 1.8
        alpha = 1.0 if totals[concept] >= 30 else 0.65
        marker = "o" if totals[concept] >= 50 else "s" if totals[concept] >= 20 else ""
        ms = 6 if marker else 0

        ax.plot(years, counts, marker=marker, linewidth=lw, linestyle=ls,
                label=f"{concept} (n={totals[concept]})", color=color,
                alpha=alpha, markersize=ms, markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Papers", fontsize=13, fontweight="bold")
    ax.set_title(f"All Major Clinical Concepts in {FIELD_NAME} Over Time\n(Title-based lexical analysis, n = {len(df):,} papers, {GEO_LABEL})",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    annotate_partial_year(ax, years)

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=9.5, frameon=True, framealpha=0.95,
        edgecolor="#CCCCCC", title="Concept (total papers)",
        title_fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "16_all_concepts_line_chart.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("✅ 16_all_concepts_line_chart.png")

    # ═══════════════════════════════════════════════════════════
    # FIGURE B: Top 10 only — cleaner for main presentation slide
    # ═══════════════════════════════════════════════════════════
    top10 = sorted_concepts[:10]

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, concept in enumerate(top10):
        counts = data[concept]
        color = COLORS[i]
        ax.plot(years, counts, marker="o", linewidth=3, label=f"{concept} (n={totals[concept]})",
                color=color, markersize=7, markeredgecolor="white", markeredgewidth=1)

    ax.set_xlabel("Year", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Papers", fontsize=13, fontweight="bold")
    ax.set_title(f"Top 10 {FIELD_SHORT} Topics Over Time ({YEAR_MIN}-{YEAR_MAX})",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    annotate_partial_year(ax, years)

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=11, frameon=True, framealpha=0.95,
        edgecolor="#CCCCCC",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "17_top10_concepts_line_chart.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("✅ 17_top10_concepts_line_chart.png")

    # ═══════════════════════════════════════════════════════════
    # FIGURE C: Grouped by domain (3×2 subplots for 6 domains)
    # ═══════════════════════════════════════════════════════════
    n_groups = len(DOMAIN_GROUPS)
    n_cols = 2
    n_rows = (n_groups + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows), sharex=True)
    axes_flat = axes.flatten()

    for idx, (group_name, concepts_in_group) in enumerate(DOMAIN_GROUPS.items()):
        ax = axes_flat[idx]
        for j, concept in enumerate(concepts_in_group):
            if concept not in data:
                continue
            counts = data[concept]
            color = COLORS[j % len(COLORS)]
            ax.plot(years, counts, marker="o", linewidth=2.5,
                    label=f"{concept} ({totals[concept]})",
                    color=color, markersize=6, markeredgecolor="white", markeredgewidth=0.5)

        ax.set_title(group_name, fontsize=13, fontweight="bold")
        ax.set_ylabel("Papers", fontsize=11)
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    # Hide any unused subplots
    for idx in range(n_groups, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add x-axis labels to the bottom row
    for idx in range(n_cols):
        bottom_idx = (n_rows - 1) * n_cols + idx
        if bottom_idx < len(axes_flat) and axes_flat[bottom_idx].get_visible():
            axes_flat[bottom_idx].set_xlabel("Year", fontsize=12, fontweight="bold")

    fig.suptitle(f"{FIELD_NAME} Research Trends by Domain ({YEAR_MIN}-{YEAR_MAX})",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "18_concepts_by_domain_grid.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("✅ 18_concepts_by_domain_grid.png")


if __name__ == "__main__":
    main()
