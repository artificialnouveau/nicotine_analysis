#!/usr/bin/env python3
"""
plots.py

Generate static and interactive visualizations for the tobacco/nicotine
industry influence analysis.

Uses three-category classification:
  - Tobacco Company: actual tobacco/nicotine industry ties
  - COI Declared: declared conflict of interest (non-tobacco)
  - Independent: no conflicts declared

Outputs (to --output_dir/figures/):
  Static (matplotlib/seaborn):
    - outcome_by_industry.png
    - outcome_proportions.png
    - centrality_distribution.png
    - community_industry_scatter.png
    - timeline_industry_papers.png
    - odds_ratio_forest.png

  Interactive (plotly HTML):
    - sankey_funding_outcome.html
    - outcome_heatmap.html
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Color scheme — three categories
CATEGORY_ORDER = ["Tobacco Company", "COI Declared", "Independent"]
CATEGORY_COLORS = {
    "Tobacco Company": "#e74c3c",   # red
    "COI Declared": "#f39c12",      # amber/orange
    "Independent": "#3498db",       # blue
}

OUTCOME_ORDER = ["Positive", "Negative", "Neutral", "Mixed", "Not coded"]
OUTCOME_COLORS = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral": "#95a5a6",
    "Mixed": "#f39c12",
    "Not coded": "#bdc3c7",
}


def _hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    """Convert a hex color to rgba string."""
    if hex_color.startswith("#") and len(hex_color) == 7:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color


def plot_outcome_by_industry(papers_df: pd.DataFrame, output_dir: str):
    """Grouped bar chart: outcome counts by three categories."""
    ctab = pd.crosstab(papers_df["industry_category"], papers_df["outcome"])
    for col in OUTCOME_ORDER:
        if col not in ctab.columns:
            ctab[col] = 0
    ctab = ctab[OUTCOME_ORDER]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(OUTCOME_ORDER))
    width = 0.25
    offsets = [-width, 0, width]

    all_bars = []
    for i, cat in enumerate(CATEGORY_ORDER):
        vals = ctab.loc[cat].values if cat in ctab.index else np.zeros(len(OUTCOME_ORDER))
        bars = ax.bar(x + offsets[i], vals, width, label=cat,
                      color=CATEGORY_COLORS[cat], alpha=0.85, edgecolor="white")
        all_bars.append(bars)

    ax.set_xlabel("Outcome Direction", fontsize=12)
    ax.set_ylabel("Number of Papers", fontsize=12)
    ax.set_title("Paper Outcomes by Author Category", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(OUTCOME_ORDER, fontsize=11)
    ax.legend(fontsize=10)

    for bars in all_bars:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outcome_by_industry.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_outcome_proportions(papers_df: pd.DataFrame, output_dir: str):
    """Stacked bar chart showing proportions for three categories."""
    ctab = pd.crosstab(papers_df["industry_category"], papers_df["outcome"], normalize="index") * 100
    for col in OUTCOME_ORDER:
        if col not in ctab.columns:
            ctab[col] = 0
    ctab = ctab[OUTCOME_ORDER]
    # Reorder rows
    ctab = ctab.reindex([c for c in CATEGORY_ORDER if c in ctab.index])

    fig, ax = plt.subplots(figsize=(10, 6))
    ctab.plot(kind="barh", stacked=True, ax=ax,
              color=[OUTCOME_COLORS.get(o, "#999") for o in OUTCOME_ORDER], edgecolor="white")

    ax.set_xlabel("Percentage of Papers (%)", fontsize=12)
    ax.set_ylabel("")
    ax.set_yticklabels([c for c in CATEGORY_ORDER if c in ctab.index], fontsize=11)
    ax.set_title("Outcome Distribution by Author Category", fontsize=14, fontweight="bold")
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outcome_proportions.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_centrality_distribution(centrality_df: pd.DataFrame, output_dir: str):
    """Box plot comparing centrality metrics across three author categories."""
    if centrality_df.empty or "author_category" not in centrality_df.columns:
        return

    metrics = ["degree_centrality", "betweenness_centrality", "eigenvector_centrality"]
    available = [m for m in metrics if m in centrality_df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        data_groups = []
        labels = []
        colors = []
        for cat in CATEGORY_ORDER:
            vals = centrality_df[centrality_df["author_category"] == cat][metric].dropna()
            if len(vals) > 0:
                data_groups.append(vals)
                labels.append(cat.replace("Tobacco Company", "Tobacco\nCompany").replace("COI Declared", "COI\nDeclared"))
                colors.append(CATEGORY_COLORS[cat])

        if not data_groups:
            continue

        bp = ax.boxplot(
            data_groups,
            tick_labels=labels,
            patch_artist=True,
            widths=0.5,
        )
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)

        label = metric.replace("_", " ").title()
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel(label, fontsize=10)

    fig.suptitle("Network Centrality by Author Category", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "centrality_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_community_scatter(comm_df: pd.DataFrame, output_dir: str):
    """Scatter plot: community % tobacco authors vs % positive outcomes."""
    if comm_df.empty:
        return

    # Use pct_tobacco if available, fall back to pct_industry
    x_col = "pct_tobacco" if "pct_tobacco" in comm_df.columns else "pct_industry"
    x_label = "% Tobacco Company Authors in Community"

    fig, ax = plt.subplots(figsize=(10, 7))

    sizes = comm_df["n_authors"] * 5
    sizes = sizes.clip(lower=20)

    scatter = ax.scatter(
        comm_df[x_col],
        comm_df["pct_positive"],
        s=sizes,
        c=comm_df[x_col],
        cmap="RdYlBu_r",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    for _, row in comm_df.iterrows():
        if row["n_authors"] >= 5:
            ax.annotate(
                f"C{row['community_id']} (n={row['n_authors']})",
                (row[x_col], row["pct_positive"]),
                fontsize=8, ha="center", va="bottom",
            )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("% Positive Outcomes in Community", fontsize=12)
    fig.suptitle("Community-Level: Industry Concentration vs Positive Outcomes", fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(
        "Each bubble is a Louvain community of co-authors. Size = number of authors.\n"
        "X-axis shows how concentrated tobacco company affiliations are; Y-axis shows the share of positive findings.",
        fontsize=9.5, color="#555", fontstyle="italic", pad=12,
    )

    plt.colorbar(scatter, ax=ax, label="% Tobacco Company", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "community_industry_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_timeline(papers_df: pd.DataFrame, output_dir: str):
    """Timeline: number of papers per year by three categories."""
    if "year" not in papers_df.columns:
        return

    df = papers_df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1960) & (df["year"] <= 2026)]

    yearly = df.groupby(["year", "industry_category"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 6))

    for cat in CATEGORY_ORDER:
        if cat in yearly.columns:
            color = CATEGORY_COLORS[cat]
            ax.fill_between(yearly.index, yearly[cat], alpha=0.2, color=color)
            ax.plot(yearly.index, yearly[cat], color=color, linewidth=2, label=cat)

    ax.set_xlabel("Publication Year", fontsize=12)
    ax.set_ylabel("Number of Papers", fontsize=12)
    ax.set_title("Publication Timeline by Author Category", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timeline_industry_papers.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_odds_ratio_forest(stats_path: str, output_dir: str):
    """Forest plot showing odds ratios for Tobacco Company and COI Declared vs Independent."""
    if not os.path.exists(stats_path):
        return

    with open(stats_path) as f:
        stats = json.load(f)

    # Collect OR data for both comparisons
    comparisons = []
    for key, label, color in [
        ("odds_ratio_tobacco_company_vs_independent", "Tobacco Company\nvs Independent", CATEGORY_COLORS["Tobacco Company"]),
        ("odds_ratio_coi_declared_vs_independent", "COI Declared\nvs Independent", CATEGORY_COLORS["COI Declared"]),
    ]:
        or_data = stats.get(key, {})
        or_val = or_data.get("odds_ratio")
        ci_lo = or_data.get("ci_95_lower")
        ci_hi = or_data.get("ci_95_upper")
        if or_val is not None and ci_lo is not None and ci_hi is not None:
            comparisons.append((label, or_val, ci_lo, ci_hi, color))

    if not comparisons:
        return

    fig, ax = plt.subplots(figsize=(10, 4 + len(comparisons)))

    y_positions = list(range(len(comparisons)))

    for i, (label, or_val, ci_lo, ci_hi, color) in enumerate(comparisons):
        ax.errorbar(
            or_val, i, xerr=[[or_val - ci_lo], [ci_hi - or_val]],
            fmt="o", color=color, markersize=12, capsize=6, linewidth=2.5,
        )

    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([c[0] for c in comparisons], fontsize=11)
    ax.set_xlabel("Odds Ratio (Positive Outcome)", fontsize=12)
    ax.set_title("Odds Ratios vs Independent Authors (with 95% CI)", fontsize=13, fontweight="bold")

    # Directional labels
    y_min = -0.7
    ax.text(0.82, y_min + 0.1, "Favors independent", ha="center", fontsize=10,
            color=CATEGORY_COLORS["Independent"], fontstyle="italic")
    ax.text(1.18, y_min + 0.1, "Favors group", ha="center", fontsize=10,
            color="#666", fontstyle="italic")

    ax.annotate("", xy=(0.72, y_min + 0.25), xytext=(0.95, y_min + 0.25),
                arrowprops=dict(arrowstyle="->", color=CATEGORY_COLORS["Independent"], lw=1.5))
    ax.annotate("", xy=(1.28, y_min + 0.25), xytext=(1.05, y_min + 0.25),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    ax.set_ylim(y_min, len(comparisons) - 0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=CATEGORY_COLORS["Tobacco Company"], edgecolor="black", label="Tobacco Company"),
        mpatches.Patch(facecolor=CATEGORY_COLORS["COI Declared"], edgecolor="black", label="COI Declared"),
        mpatches.Patch(facecolor=CATEGORY_COLORS["Independent"], edgecolor="black", label="Independent (reference)"),
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Null (OR = 1.0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "odds_ratio_forest.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_sankey_interactive(papers_df: pd.DataFrame, output_dir: str):
    """Interactive Sankey diagram: three categories -> Outcome."""
    if not HAS_PLOTLY:
        print("[WARN] plotly not installed, skipping Sankey diagram")
        return

    node_labels = list(CATEGORY_ORDER) + OUTCOME_ORDER
    node_colors = [CATEGORY_COLORS[c] for c in CATEGORY_ORDER] + [OUTCOME_COLORS[o] for o in OUTCOME_ORDER]

    sources = []
    targets = []
    values = []
    link_colors = []

    n_cats = len(CATEGORY_ORDER)
    for i, cat in enumerate(CATEGORY_ORDER):
        subset = papers_df[papers_df["industry_category"] == cat]
        for j, outcome in enumerate(OUTCOME_ORDER):
            count = (subset["outcome"] == outcome).sum()
            if count > 0:
                sources.append(i)
                targets.append(n_cats + j)
                values.append(count)
                link_colors.append(_hex_to_rgba(node_colors[i], 0.4))

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    ))

    fig.update_layout(
        title_text="Flow: Author Category to Paper Outcomes",
        font_size=12,
        width=1000,
        height=550,
    )

    fig.write_html(os.path.join(output_dir, "sankey_funding_outcome.html"))


def plot_outcome_heatmap_interactive(papers_df: pd.DataFrame, output_dir: str):
    """Interactive heatmap: outcome by year and three categories."""
    if not HAS_PLOTLY:
        return

    df = papers_df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1990) & (df["year"] <= 2026)]

    rows = []
    for year in sorted(df["year"].unique()):
        for cat in CATEGORY_ORDER:
            subset = df[(df["year"] == year) & (df["industry_category"] == cat)]
            total = len(subset)
            n_pos = (subset["outcome"] == "Positive").sum()
            pct = (n_pos / total * 100) if total > 0 else 0
            rows.append({"year": year, "group": cat, "pct_positive": round(pct, 1), "n": total})

    plot_df = pd.DataFrame(rows)
    pivot = plot_df.pivot(index="group", columns="year", values="pct_positive").fillna(0)
    # Reorder rows
    pivot = pivot.reindex([c for c in CATEGORY_ORDER if c in pivot.index])

    fig = px.imshow(
        pivot.values,
        labels=dict(x="Year", y="Group", color="% Positive"),
        x=list(pivot.columns),
        y=list(pivot.index),
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig.update_layout(
        title="% Positive Outcomes Over Time by Author Category",
        width=1200,
        height=350,
    )
    fig.write_html(os.path.join(output_dir, "outcome_heatmap.html"))


def main():
    ap = argparse.ArgumentParser(description="Generate visualizations")
    ap.add_argument("--data_dir", required=True, help="Dir with papers.csv, authors.csv")
    ap.add_argument("--network_dir", required=True, help="Dir with centrality/community CSVs")
    ap.add_argument("--stats_dir", required=True, help="Dir with full_statistics.json")
    ap.add_argument("--output_dir", required=True, help="Where to write figures/")
    args = ap.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    papers_df = pd.read_csv(os.path.join(args.data_dir, "papers.csv"))

    centrality_path = os.path.join(args.network_dir, "top_authors_centrality.csv")
    centrality_df = pd.read_csv(centrality_path) if os.path.exists(centrality_path) else pd.DataFrame()

    comm_path = os.path.join(args.network_dir, "community_summary.csv")
    comm_df = pd.read_csv(comm_path) if os.path.exists(comm_path) else pd.DataFrame()

    stats_path = os.path.join(args.stats_dir, "full_statistics.json")

    print("Generating static plots...")
    plot_outcome_by_industry(papers_df, fig_dir)
    plot_outcome_proportions(papers_df, fig_dir)
    plot_centrality_distribution(centrality_df, fig_dir)
    plot_community_scatter(comm_df, fig_dir)
    plot_timeline(papers_df, fig_dir)
    plot_odds_ratio_forest(stats_path, fig_dir)

    print("Generating interactive plots...")
    plot_sankey_interactive(papers_df, fig_dir)
    plot_outcome_heatmap_interactive(papers_df, fig_dir)

    print(f"All figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
