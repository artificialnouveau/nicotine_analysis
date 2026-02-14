#!/usr/bin/env python3
"""
app.py

Streamlit interactive dashboard for the tobacco/nicotine industry influence analysis.

Run: streamlit run viz/app.py -- --data_dir output/data --network_dir output/network --stats_dir output/stats

Features:
  - Overview statistics and key findings
  - Interactive co-authorship network (embedded plotly)
  - Outcome comparison charts (industry vs independent)
  - Community breakdown
  - Author explorer with search
  - Paper browser with filters
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("Streamlit and plotly are required: pip install streamlit plotly")
    sys.exit(1)

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTCOME_ORDER = ["Positive", "Negative", "Neutral", "Mixed", "Not coded"]
COLORS = {
    "industry": "#e74c3c",
    "independent": "#3498db",
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral": "#95a5a6",
    "Mixed": "#f39c12",
    "Not coded": "#bdc3c7",
}


def load_data(data_dir: str, network_dir: str, stats_dir: str):
    """Load all datasets."""
    data = {}

    papers_path = os.path.join(data_dir, "papers.csv")
    authors_path = os.path.join(data_dir, "authors.csv")
    edges_path = os.path.join(data_dir, "author_papers.csv")

    data["papers"] = pd.read_csv(papers_path) if os.path.exists(papers_path) else pd.DataFrame()
    data["authors"] = pd.read_csv(authors_path) if os.path.exists(authors_path) else pd.DataFrame()
    data["edges"] = pd.read_csv(edges_path) if os.path.exists(edges_path) else pd.DataFrame()

    centrality_path = os.path.join(network_dir, "top_authors_centrality.csv")
    comm_path = os.path.join(network_dir, "community_summary.csv")
    stats_path = os.path.join(stats_dir, "full_statistics.json")
    net_stats_path = os.path.join(network_dir, "network_stats.json")

    data["centrality"] = pd.read_csv(centrality_path) if os.path.exists(centrality_path) else pd.DataFrame()
    data["communities"] = pd.read_csv(comm_path) if os.path.exists(comm_path) else pd.DataFrame()

    if os.path.exists(stats_path):
        with open(stats_path) as f:
            data["stats"] = json.load(f)
    else:
        data["stats"] = {}

    if os.path.exists(net_stats_path):
        with open(net_stats_path) as f:
            data["net_stats"] = json.load(f)
    else:
        data["net_stats"] = {}

    return data


# ---------------------------------------------------------------------------
# Dashboard pages
# ---------------------------------------------------------------------------

def page_overview(data):
    st.header("Overview")

    papers = data["papers"]
    authors = data["authors"]
    stats = data["stats"]
    net_stats = data["net_stats"]

    if papers.empty:
        st.warning("No paper data loaded.")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", len(papers))
    with col2:
        n_ind = (papers["industry_involved"] == "Yes").sum()
        st.metric("Industry-Involved", n_ind)
    with col3:
        st.metric("Total Authors", len(authors) if not authors.empty else "N/A")
    with col4:
        n_ind_auth = authors["is_industry_affiliated"].sum() if not authors.empty and "is_industry_affiliated" in authors.columns else 0
        st.metric("Industry Authors", int(n_ind_auth))

    st.divider()

    # Key statistical findings
    st.subheader("Key Statistical Findings")

    or_data = stats.get("odds_ratio_positive", {})
    chi_data = stats.get("chi_square", {})
    prop_data = stats.get("proportion_ztest_positive", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        or_val = or_data.get("odds_ratio", "N/A")
        ci_lo = or_data.get("ci_95_lower", "")
        ci_hi = or_data.get("ci_95_upper", "")
        st.metric("Odds Ratio (Positive)", f"{or_val}")
        if ci_lo and ci_hi:
            st.caption(f"95% CI: [{ci_lo}, {ci_hi}]")

    with col2:
        p_val = chi_data.get("p_value", "N/A")
        st.metric("Chi-square p-value", f"{p_val}")

    with col3:
        fisher_p = stats.get("fisher_exact_p", "N/A")
        st.metric("Fisher exact p-value", f"{fisher_p}")

    # Interpretation
    if isinstance(or_val, (int, float)) and or_val > 1:
        st.info(
            f"Industry-involved papers are **{or_val:.2f}x** more likely to report "
            f"positive/beneficial outcomes compared to independent papers."
        )
    elif isinstance(or_val, (int, float)) and or_val < 1:
        st.info(
            f"Industry-involved papers are **{1/or_val:.2f}x less likely** to report "
            f"positive outcomes compared to independent papers."
        )

    # Network stats
    if net_stats:
        st.subheader("Network Statistics")
        ncol1, ncol2, ncol3, ncol4 = st.columns(4)
        with ncol1:
            st.metric("Nodes", net_stats.get("n_nodes", "N/A"))
        with ncol2:
            st.metric("Edges", net_stats.get("n_edges", "N/A"))
        with ncol3:
            st.metric("Communities", net_stats.get("communities", "N/A"))
        with ncol4:
            assort = net_stats.get("industry_assortativity")
            st.metric("Industry Assortativity", f"{assort:.3f}" if assort is not None else "N/A")
            if assort is not None:
                if assort > 0:
                    st.caption("Positive = industry authors cluster together")
                else:
                    st.caption("Negative = industry authors don't cluster")


def page_outcomes(data):
    st.header("Outcome Analysis")

    papers = data["papers"]
    if papers.empty:
        st.warning("No data loaded.")
        return

    # Grouped bar chart
    st.subheader("Outcome Counts by Industry Involvement")
    ctab = pd.crosstab(papers["industry_involved"], papers["outcome"])
    for col in OUTCOME_ORDER:
        if col not in ctab.columns:
            ctab[col] = 0
    ctab = ctab[OUTCOME_ORDER]

    fig = go.Figure()
    for outcome in OUTCOME_ORDER:
        vals = []
        for group in ["Yes", "No"]:
            vals.append(ctab.loc[group, outcome] if group in ctab.index else 0)
        fig.add_trace(go.Bar(
            name=outcome,
            x=["Industry-Involved", "Independent"],
            y=vals,
            marker_color=COLORS.get(outcome, "#999"),
        ))
    fig.update_layout(barmode="group", title="Paper Outcomes by Industry Involvement",
                      yaxis_title="Count", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Proportional view
    st.subheader("Outcome Proportions (%)")
    ctab_pct = pd.crosstab(papers["industry_involved"], papers["outcome"], normalize="index") * 100
    for col in OUTCOME_ORDER:
        if col not in ctab_pct.columns:
            ctab_pct[col] = 0
    ctab_pct = ctab_pct[OUTCOME_ORDER]

    fig2 = go.Figure()
    for outcome in OUTCOME_ORDER:
        vals = []
        for group in ["Yes", "No"]:
            vals.append(round(ctab_pct.loc[group, outcome], 1) if group in ctab_pct.index else 0)
        fig2.add_trace(go.Bar(
            name=outcome,
            x=["Industry-Involved", "Independent"],
            y=vals,
            marker_color=COLORS.get(outcome, "#999"),
            text=[f"{v:.1f}%" for v in vals],
            textposition="auto",
        ))
    fig2.update_layout(barmode="stack", title="Outcome Distribution (%)",
                       yaxis_title="Percentage", height=500)
    st.plotly_chart(fig2, use_container_width=True)

    # Timeline
    st.subheader("Positive Outcomes Over Time")
    df_time = papers.dropna(subset=["year"]).copy()
    if not df_time.empty:
        df_time["year"] = df_time["year"].astype(int)
        df_time = df_time[(df_time["year"] >= 1990) & (df_time["year"] <= 2026)]

        yearly = []
        for year in sorted(df_time["year"].unique()):
            for group, label in [("Yes", "Industry"), ("No", "Independent")]:
                sub = df_time[(df_time["year"] == year) & (df_time["industry_involved"] == group)]
                n_pos = (sub["outcome"] == "Positive").sum()
                total = len(sub)
                pct = (n_pos / total * 100) if total > 0 else 0
                yearly.append({"year": year, "group": label, "pct_positive": pct, "n": total})

        yearly_df = pd.DataFrame(yearly)
        fig3 = px.line(yearly_df, x="year", y="pct_positive", color="group",
                       color_discrete_map={"Industry": COLORS["industry"], "Independent": COLORS["independent"]},
                       title="% Positive Outcomes Over Time",
                       labels={"pct_positive": "% Positive", "year": "Year"})
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)


def page_network(data):
    st.header("Co-Authorship Network")

    centrality = data["centrality"]
    communities = data["communities"]

    if centrality.empty:
        st.warning("No centrality data. Run network.py first.")
        return

    # Top authors table
    st.subheader("Top Authors by Betweenness Centrality")
    top_n = st.slider("Show top N authors", 10, 100, 25)
    display_cols = ["name", "is_industry", "paper_count", "degree",
                    "betweenness_centrality", "eigenvector_centrality", "pct_positive"]
    available_cols = [c for c in display_cols if c in centrality.columns]
    st.dataframe(centrality.head(top_n)[available_cols], use_container_width=True)

    # Centrality comparison
    st.subheader("Centrality: Industry vs Independent")
    if "is_industry" in centrality.columns:
        metric = st.selectbox("Metric", ["betweenness_centrality", "degree_centrality",
                                          "eigenvector_centrality", "closeness_centrality"])
        if metric in centrality.columns:
            fig = px.box(
                centrality, x="is_industry", y=metric,
                color="is_industry",
                color_discrete_map={True: COLORS["industry"], False: COLORS["independent"]},
                labels={"is_industry": "Industry-Affiliated", metric: metric.replace("_", " ").title()},
                title=f"{metric.replace('_', ' ').title()} by Industry Status",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Community breakdown
    if not communities.empty:
        st.subheader("Community Breakdown")
        st.dataframe(communities, use_container_width=True)

        fig = px.scatter(
            communities, x="pct_industry", y="pct_positive",
            size="n_authors", color="pct_industry",
            color_continuous_scale="RdYlBu_r",
            hover_data=["community_id", "n_authors", "n_industry_authors"],
            title="Communities: % Industry vs % Positive Outcomes",
            labels={"pct_industry": "% Industry Authors", "pct_positive": "% Positive Outcomes"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Link to interactive network HTML
    st.subheader("Interactive Network Visualization")
    st.info("Open the interactive HTML files in your browser for full exploration:")
    st.code("open output/viz/coauthor_network_interactive.html")
    st.code("open output/viz/coauthor_network_plotly.html")


def page_papers(data):
    st.header("Paper Browser")

    papers = data["papers"]
    if papers.empty:
        st.warning("No data loaded.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        industry_filter = st.multiselect("Industry Involved", ["Yes", "No"], default=["Yes", "No"])
    with col2:
        outcome_filter = st.multiselect("Outcome", OUTCOME_ORDER, default=OUTCOME_ORDER)
    with col3:
        search = st.text_input("Search title")

    filtered = papers[
        papers["industry_involved"].isin(industry_filter) &
        papers["outcome"].isin(outcome_filter)
    ]
    if search:
        filtered = filtered[filtered["title"].str.contains(search, case=False, na=False)]

    st.write(f"Showing {len(filtered)} / {len(papers)} papers")

    display_cols = ["title", "year", "journal", "outcome", "industry_involved",
                    "declared_coi", "industry_orgs", "doi"]
    available = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[available].head(200), use_container_width=True, height=600)


def page_authors(data):
    st.header("Author Explorer")

    authors = data["authors"]
    centrality = data["centrality"]

    if authors.empty:
        st.warning("No author data loaded.")
        return

    # Merge centrality if available
    if not centrality.empty and "author_id" in centrality.columns:
        merged = authors.merge(centrality[["author_id", "degree", "betweenness_centrality",
                                            "eigenvector_centrality", "pct_positive"]],
                               on="author_id", how="left", suffixes=("", "_cent"))
    else:
        merged = authors

    # Filter
    col1, col2 = st.columns(2)
    with col1:
        show_industry = st.checkbox("Show only industry-affiliated", value=False)
    with col2:
        search = st.text_input("Search author name")

    if show_industry and "is_industry_affiliated" in merged.columns:
        merged = merged[merged["is_industry_affiliated"] == True]
    if search:
        merged = merged[merged["name"].str.contains(search, case=False, na=False)]

    display_cols = ["name", "is_industry_affiliated", "industry_orgs", "paper_count",
                    "degree", "betweenness_centrality", "pct_positive", "affiliations"]
    available = [c for c in display_cols if c in merged.columns]

    st.write(f"Showing {len(merged)} authors")
    st.dataframe(merged[available].head(200), use_container_width=True, height=600)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Tobacco/Nicotine Industry Influence Analysis",
        page_icon="🔬",
        layout="wide",
    )

    st.title("Tobacco/Nicotine Industry Influence Network Analysis")
    st.caption("Analyzing conflicts of interest and outcome bias in tobacco/nicotine research")

    # Parse args from CLI (streamlit passes them after --)
    # Default paths for convenience
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data = os.path.join(base, "output", "data")
    default_network = os.path.join(base, "output", "network")
    default_stats = os.path.join(base, "output", "stats")

    # Allow override via query params or sidebar
    with st.sidebar:
        st.header("Data Paths")
        data_dir = st.text_input("Data dir", value=default_data)
        network_dir = st.text_input("Network dir", value=default_network)
        stats_dir = st.text_input("Stats dir", value=default_stats)

    data = load_data(data_dir, network_dir, stats_dir)

    # Navigation
    with st.sidebar:
        st.divider()
        page = st.radio("Navigate", ["Overview", "Outcomes", "Network", "Papers", "Authors"])

    if page == "Overview":
        page_overview(data)
    elif page == "Outcomes":
        page_outcomes(data)
    elif page == "Network":
        page_network(data)
    elif page == "Papers":
        page_papers(data)
    elif page == "Authors":
        page_authors(data)


if __name__ == "__main__":
    main()
