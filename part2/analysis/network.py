#!/usr/bin/env python3
"""
network.py

Build co-authorship network from the author/paper tables produced by load_and_identify.py.

Network structure:
  - Nodes = authors
  - Edges = co-authorship (two authors who appear on the same paper)
  - Edge weight = number of shared papers

Node attributes:
  - is_industry_affiliated (bool)
  - industry_orgs (str)
  - paper_count (int)
  - pct_positive_outcomes (float) — % of that author's papers coded as Positive
  - community (int) — Louvain community ID

Outputs (to --output_dir):
  - coauthor_network.graphml       : full network in GraphML format (for Gephi)
  - coauthor_network.gexf          : GEXF format (for Gephi with dynamics)
  - network_stats.json             : global + per-node centrality stats
  - top_authors_centrality.csv     : top authors by betweenness centrality
  - community_summary.csv          : community-level industry/outcome breakdown
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Tuple

import networkx as nx
import pandas as pd

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


def build_coauthor_graph(
    authors_df: pd.DataFrame,
    papers_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> nx.Graph:
    """
    Build an undirected weighted co-authorship graph.
    """
    G = nx.Graph()

    # Index author info by author_id
    author_info = {}
    for _, row in authors_df.iterrows():
        aid = row["author_id"]
        author_info[aid] = {
            "name": row.get("name", ""),
            "is_industry": bool(row.get("is_industry_affiliated", False)),
            "industry_orgs": row.get("industry_orgs", ""),
            "paper_count": int(row.get("paper_count", 0)),
        }

    # Index paper outcomes
    paper_outcomes = {}
    paper_category = {}
    for _, row in papers_df.iterrows():
        pid = row["paper_id"]
        paper_outcomes[pid] = row.get("outcome", "Not coded")
        paper_category[pid] = row.get("industry_category", "Independent")

    # Map author -> papers and compute per-author outcome stats
    author_papers_map: Dict[str, List[str]] = defaultdict(list)
    for _, row in edges_df.iterrows():
        author_papers_map[row["author_id"]].append(row["paper_id"])

    # Add nodes
    for aid, info in author_info.items():
        papers = author_papers_map.get(aid, [])
        outcomes = [paper_outcomes.get(p, "Not coded") for p in papers]
        n_positive = sum(1 for o in outcomes if o == "Positive")
        n_negative = sum(1 for o in outcomes if o == "Negative")
        pct_positive = (n_positive / len(outcomes) * 100) if outcomes else 0.0
        pct_negative = (n_negative / len(outcomes) * 100) if outcomes else 0.0

        # Determine if author is ever on an industry-involved paper
        any_industry_paper = any(paper_category.get(p) == "Tobacco Company" for p in papers)
        # Determine if author is ever on a COI-declared paper (but not tobacco)
        any_coi_paper = any(paper_category.get(p) == "COI Declared" for p in papers)

        G.add_node(
            aid,
            label=info["name"],
            is_industry=info["is_industry"],
            industry_orgs=info["industry_orgs"],
            paper_count=info["paper_count"],
            pct_positive=round(pct_positive, 1),
            pct_negative=round(pct_negative, 1),
            n_positive=n_positive,
            n_negative=n_negative,
            any_industry_paper=any_industry_paper,
            any_coi_paper=any_coi_paper,
            author_category="Tobacco Company" if info["is_industry"] else ("COI Declared" if any_coi_paper else "Independent"),
        )

    # Build co-authorship edges
    # Group authors by paper
    paper_authors: Dict[str, List[str]] = defaultdict(list)
    for _, row in edges_df.iterrows():
        paper_authors[row["paper_id"]].append(row["author_id"])

    edge_weights: Counter = Counter()
    edge_papers: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for pid, auth_list in paper_authors.items():
        # Only create edges for unique author pairs on this paper
        unique_authors = list(set(auth_list))
        for a1, a2 in combinations(sorted(unique_authors), 2):
            edge_weights[(a1, a2)] += 1
            edge_papers[(a1, a2)].append(pid)

    for (a1, a2), weight in edge_weights.items():
        if a1 in G.nodes and a2 in G.nodes:
            cat1 = G.nodes[a1].get("author_category", "Independent")
            cat2 = G.nodes[a2].get("author_category", "Independent")
            cats = sorted([cat1, cat2])

            if cats == ["Tobacco Company", "Tobacco Company"]:
                edge_type = "both_tobacco"
            elif "Tobacco Company" in cats:
                edge_type = "tobacco_mixed"
            elif cats == ["COI Declared", "COI Declared"]:
                edge_type = "both_coi"
            elif "COI Declared" in cats:
                edge_type = "coi_mixed"
            else:
                edge_type = "independent"

            G.add_edge(
                a1, a2,
                weight=weight,
                shared_papers=len(edge_papers[(a1, a2)]),
                edge_type=edge_type,
            )

    return G


def compute_centrality(G: nx.Graph) -> pd.DataFrame:
    """Compute centrality metrics for all nodes."""
    if len(G) == 0:
        return pd.DataFrame()

    degree_cent = nx.degree_centrality(G)

    # For betweenness and closeness, NetworkX treats weight as distance/cost.
    # In a co-authorship network, weight = number of shared papers (strength).
    # We invert the weight so stronger ties = shorter paths.
    # Use a temporary copy to avoid leaking inv_weight into the exported graph.
    H = G.copy()
    inv_weight = {(u, v): 1.0 / d["weight"] for u, v, d in H.edges(data=True) if d.get("weight", 0) > 0}
    nx.set_edge_attributes(H, inv_weight, "inv_weight")

    betweenness = nx.betweenness_centrality(H, weight="inv_weight")
    closeness = nx.closeness_centrality(H, distance="inv_weight")

    try:
        # Eigenvector centrality correctly treats weight as strength in NetworkX
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G.nodes}

    rows = []
    for node in G.nodes:
        data = G.nodes[node]
        rows.append({
            "author_id": node,
            "name": data.get("label", ""),
            "is_industry": data.get("is_industry", False),
            "author_category": data.get("author_category", "Independent"),
            "paper_count": data.get("paper_count", 0),
            "pct_positive": data.get("pct_positive", 0),
            "degree": G.degree(node),
            "weighted_degree": G.degree(node, weight="weight"),
            "degree_centrality": round(degree_cent.get(node, 0), 6),
            "betweenness_centrality": round(betweenness.get(node, 0), 6),
            "closeness_centrality": round(closeness.get(node, 0), 6),
            "eigenvector_centrality": round(eigenvector.get(node, 0), 6),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("betweenness_centrality", ascending=False)
    return df


def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """Louvain community detection."""
    if not HAS_LOUVAIN:
        print("[WARN] python-louvain not installed. Skipping community detection.")
        return {n: 0 for n in G.nodes}

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {n: 0 for n in G.nodes}

    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    # Assign community to nodes
    for node, comm_id in partition.items():
        G.nodes[node]["community"] = comm_id
    return partition


def community_summary(
    G: nx.Graph,
    partition: Dict[str, int],
    edges_df: pd.DataFrame = None,
    papers_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Summarize each community by industry presence and outcome tendency.

    Bug fix: counts unique papers per community instead of summing per-author
    paper counts, which double-counted papers shared by co-authors.
    """
    # Map author -> community
    author_comm = partition

    # Build community -> unique papers mapping if edge/paper data available
    comm_papers: Dict[int, set] = defaultdict(set)
    paper_outcomes_map: Dict[str, str] = {}

    if edges_df is not None and papers_df is not None:
        # Index paper outcomes
        for _, row in papers_df.iterrows():
            paper_outcomes_map[str(row["paper_id"])] = row.get("outcome", "Not coded")

        # Map each paper to the community of its authors
        for _, row in edges_df.iterrows():
            aid = row["author_id"]
            pid = str(row["paper_id"])
            if aid in author_comm:
                comm_papers[author_comm[aid]].add(pid)

    comm_data: Dict[int, Dict] = defaultdict(lambda: {
        "n_authors": 0,
        "n_tobacco": 0,
        "n_coi": 0,
        "n_independent": 0,
    })

    for node, comm_id in partition.items():
        data = G.nodes[node]
        d = comm_data[comm_id]
        d["n_authors"] += 1
        cat = data.get("author_category", "Independent")
        if cat == "Tobacco Company":
            d["n_tobacco"] += 1
        elif cat == "COI Declared":
            d["n_coi"] += 1
        else:
            d["n_independent"] += 1

    rows = []
    for comm_id, d in sorted(comm_data.items()):
        pct_tobacco = (d["n_tobacco"] / d["n_authors"] * 100) if d["n_authors"] else 0
        pct_coi = (d["n_coi"] / d["n_authors"] * 100) if d["n_authors"] else 0

        # Count unique papers and their outcomes
        unique_papers = comm_papers.get(comm_id, set())
        n_positive = sum(1 for p in unique_papers if paper_outcomes_map.get(p) == "Positive")
        n_negative = sum(1 for p in unique_papers if paper_outcomes_map.get(p) == "Negative")
        total_coded = n_positive + n_negative
        pct_positive = (n_positive / total_coded * 100) if total_coded else 0

        rows.append({
            "community_id": comm_id,
            "n_authors": d["n_authors"],
            "n_tobacco_authors": d["n_tobacco"],
            "n_coi_authors": d["n_coi"],
            "n_independent_authors": d["n_independent"],
            "pct_tobacco": round(pct_tobacco, 1),
            "pct_coi": round(pct_coi, 1),
            "total_papers": len(unique_papers),
            "n_positive_outcomes": n_positive,
            "n_negative_outcomes": n_negative,
            "pct_positive": round(pct_positive, 1),
        })

    return pd.DataFrame(rows).sort_values("n_authors", ascending=False)


def global_stats(G: nx.Graph) -> Dict[str, Any]:
    """Compute global network statistics."""
    stats: Dict[str, Any] = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }

    if G.number_of_nodes() == 0:
        return stats

    stats["density"] = round(nx.density(G), 6)
    stats["avg_degree"] = round(sum(d for _, d in G.degree()) / G.number_of_nodes(), 2)
    stats["avg_weighted_degree"] = round(
        sum(d for _, d in G.degree(weight="weight")) / G.number_of_nodes(), 2
    )

    # Connected components
    components = list(nx.connected_components(G))
    stats["n_connected_components"] = len(components)
    largest_cc = max(components, key=len) if components else set()
    stats["largest_component_size"] = len(largest_cc)
    stats["largest_component_pct"] = round(
        len(largest_cc) / G.number_of_nodes() * 100, 1
    ) if G.number_of_nodes() > 0 else 0

    # Clustering coefficient
    stats["avg_clustering_coefficient"] = round(nx.average_clustering(G, weight="weight"), 4)

    # Transitivity
    stats["transitivity"] = round(nx.transitivity(G), 4)

    # Node counts by category
    n_tobacco = sum(1 for n in G.nodes if G.nodes[n].get("author_category") == "Tobacco Company")
    n_coi = sum(1 for n in G.nodes if G.nodes[n].get("author_category") == "COI Declared")
    n_independent = sum(1 for n in G.nodes if G.nodes[n].get("author_category") == "Independent")
    stats["n_tobacco_nodes"] = n_tobacco
    stats["n_coi_nodes"] = n_coi
    stats["n_independent_nodes"] = n_independent

    # Assortativity by industry flag (tobacco vs non-tobacco)
    # Use a temporary copy to avoid leaking industry_int into the exported graph
    try:
        H = G.copy()
        nx.set_node_attributes(H, {n: 1 if H.nodes[n].get("is_industry") else 0 for n in H.nodes}, "industry_int")
        stats["industry_assortativity"] = round(
            nx.attribute_assortativity_coefficient(H, "industry_int"), 4
        )
    except Exception:
        stats["industry_assortativity"] = None

    return stats


def main():
    ap = argparse.ArgumentParser(description="Build co-authorship network")
    ap.add_argument("--input_dir", required=True, help="Dir with authors.csv, papers.csv, author_papers.csv")
    ap.add_argument("--output_dir", required=True, help="Where to write network outputs")
    ap.add_argument("--min_papers", type=int, default=1,
                    help="Only include authors with >= this many papers (reduces noise)")
    args = ap.parse_args()

    authors_df = pd.read_csv(os.path.join(args.input_dir, "authors.csv"))
    papers_df = pd.read_csv(os.path.join(args.input_dir, "papers.csv"))
    edges_df = pd.read_csv(os.path.join(args.input_dir, "author_papers.csv"))

    # Filter by min_papers
    if args.min_papers > 1:
        keep = set(authors_df[authors_df["paper_count"] >= args.min_papers]["author_id"])
        authors_df = authors_df[authors_df["author_id"].isin(keep)]
        edges_df = edges_df[edges_df["author_id"].isin(keep)]
        print(f"Filtered to {len(authors_df)} authors with >= {args.min_papers} papers")

    G = build_coauthor_graph(authors_df, papers_df, edges_df)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Community detection
    partition = detect_communities(G)

    # Centrality
    centrality_df = compute_centrality(G)

    # Community summary (pass edges_df and papers_df for accurate unique-paper counts)
    comm_df = community_summary(G, partition, edges_df=edges_df, papers_df=papers_df)

    # Global stats
    stats = global_stats(G)
    stats["communities"] = len(set(partition.values())) if partition else 0

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)

    nx.write_graphml(G, os.path.join(args.output_dir, "coauthor_network.graphml"))
    nx.write_gexf(G, os.path.join(args.output_dir, "coauthor_network.gexf"))

    centrality_df.to_csv(os.path.join(args.output_dir, "top_authors_centrality.csv"), index=False)
    comm_df.to_csv(os.path.join(args.output_dir, "community_summary.csv"), index=False)

    with open(os.path.join(args.output_dir, "network_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Global stats: {json.dumps(stats, indent=2)}")
    print(f"Top 10 by betweenness centrality:")
    if not centrality_df.empty:
        print(centrality_df.head(10)[["name", "is_industry", "paper_count", "degree",
                                       "betweenness_centrality", "pct_positive"]].to_string(index=False))
    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
