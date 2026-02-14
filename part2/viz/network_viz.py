#!/usr/bin/env python3
"""
network_viz.py

Generate interactive co-authorship network visualizations.

Outputs:
  - coauthor_network_interactive.html   (pyvis interactive network)
  - coauthor_network_plotly.html        (plotly-based with hover info)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import networkx as nx
import pandas as pd

try:
    from pyvis.network import Network as PyvisNetwork
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


INDUSTRY_COLOR = "#e74c3c"
INDEPENDENT_COLOR = "#3498db"
MIXED_EDGE_COLOR = "#f39c12"
IND_EDGE_COLOR = "#e74c3c"
NOIND_EDGE_COLOR = "#3498db"


def load_network(graphml_path: str) -> nx.Graph:
    return nx.read_graphml(graphml_path)


def filter_for_viz(G: nx.Graph, max_nodes: int = 500, min_degree: int = 2) -> nx.Graph:
    """
    Filter network for visualization: keep only nodes with degree >= min_degree,
    then take the largest connected component, capped at max_nodes by betweenness.
    """
    # Remove isolates and low-degree nodes
    remove = [n for n in G.nodes if G.degree(n) < min_degree]
    H = G.copy()
    H.remove_nodes_from(remove)

    if H.number_of_nodes() == 0:
        return H

    # Take largest connected component
    largest_cc = max(nx.connected_components(H), key=len)
    H = H.subgraph(largest_cc).copy()

    # If still too large, keep top nodes by betweenness
    if H.number_of_nodes() > max_nodes:
        betweenness = nx.betweenness_centrality(H)
        top_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:max_nodes]
        H = H.subgraph(top_nodes).copy()

    return H


def generate_pyvis(G: nx.Graph, output_path: str):
    """Generate interactive pyvis HTML network visualization."""
    if not HAS_PYVIS:
        print("[WARN] pyvis not installed, skipping interactive network")
        return

    net = PyvisNetwork(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        notebook=False,
        directed=False,
    )

    # Physics settings — slowed down to prevent fast spinning
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -15,
                "centralGravity": 0.002,
                "springLength": 150,
                "springConstant": 0.02,
                "damping": 0.95,
                "avoidOverlap": 0.5
            },
            "maxVelocity": 3,
            "minVelocity": 0.1,
            "solver": "forceAtlas2Based",
            "timestep": 0.1,
            "stabilization": {
                "enabled": true,
                "iterations": 500,
                "updateInterval": 25,
                "fit": true
            }
        },
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": {"size": 10, "face": "arial"}
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "color": {"inherit": false}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
        }
    }
    """)

    # Add nodes
    for node in G.nodes:
        data = G.nodes[node]
        is_industry = str(data.get("is_industry", "false")).lower() == "true"
        name = data.get("label", node)
        paper_count = int(float(data.get("paper_count", 1)))
        pct_pos = float(data.get("pct_positive", 0))
        community = data.get("community", 0)
        orgs = data.get("industry_orgs", "")

        color = INDUSTRY_COLOR if is_industry else INDEPENDENT_COLOR
        size = max(8, min(50, paper_count * 3))

        title = (
            f"<b>{name}</b><br>"
            f"Papers: {paper_count}<br>"
            f"Industry: {'Yes' if is_industry else 'No'}<br>"
            f"% Positive: {pct_pos:.1f}%<br>"
            f"Community: {community}<br>"
        )
        if orgs:
            title += f"Orgs: {orgs}<br>"

        shape = "diamond" if is_industry else "dot"

        net.add_node(
            node,
            label=name if paper_count >= 3 else "",
            title=title,
            color=color,
            size=size,
            shape=shape,
            borderWidth=3 if is_industry else 1,
        )

    # Add edges
    for u, v, data in G.edges(data=True):
        weight = float(data.get("weight", 1))
        edge_type = data.get("edge_type", "independent")

        if edge_type == "both_industry":
            color = IND_EDGE_COLOR
        elif edge_type == "mixed":
            color = MIXED_EDGE_COLOR
        else:
            color = NOIND_EDGE_COLOR

        net.add_edge(
            u, v,
            value=weight,
            color={"color": color, "opacity": 0.4},
            title=f"Shared papers: {int(weight)}",
        )

    net.save_graph(output_path)

    # Inject a color legend into the saved HTML
    legend_html = """
    <div id="network-legend" style="
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(26, 26, 46, 0.92);
        border: 1px solid #444;
        border-radius: 10px;
        padding: 16px 20px;
        z-index: 9999;
        font-family: Arial, sans-serif;
        color: white;
        font-size: 13px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        min-width: 180px;
    ">
        <div style="font-weight: bold; font-size: 15px; margin-bottom: 12px; border-bottom: 1px solid #555; padding-bottom: 8px;">
            Legend
        </div>
        <div style="font-weight: bold; margin-bottom: 6px; color: #ccc;">Nodes (Authors)</div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="display:inline-block; width:16px; height:16px; background:#e74c3c; border-radius:2px; margin-right:8px; clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);"></span>
            Industry-Affiliated
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="display:inline-block; width:14px; height:14px; background:#3498db; border-radius:50%; margin-right:8px;"></span>
            Independent
        </div>
        <div style="font-weight: bold; margin-bottom: 6px; color: #ccc;">Edges (Co-authorship)</div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display:inline-block; width:24px; height:3px; background:#e74c3c; margin-right:8px;"></span>
            Both Industry
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display:inline-block; width:24px; height:3px; background:#f39c12; margin-right:8px;"></span>
            Mixed (Industry + Independent)
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="display:inline-block; width:24px; height:3px; background:#3498db; margin-right:8px;"></span>
            Both Independent
        </div>
        <div style="font-size: 11px; color: #888; border-top: 1px solid #555; padding-top: 8px;">
            Node size = paper count<br>
            Edge thickness = shared papers<br>
            Hover for details
        </div>
    </div>
    """

    # Read the saved HTML and inject the legend before </body>
    with open(output_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace("</body>", legend_html + "\n</body>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Interactive network saved to: {output_path}")


def generate_plotly_network(G: nx.Graph, output_path: str):
    """Generate plotly-based network visualization with spring layout."""
    if not HAS_PLOTLY:
        print("[WARN] plotly not installed, skipping plotly network")
        return

    # Compute layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42, weight="weight")

    # Edge traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
        opacity=0.3,
    )

    # Node traces (separate for industry and independent)
    def make_node_trace(nodes, color, name, symbol):
        x = [pos[n][0] for n in nodes]
        y = [pos[n][1] for n in nodes]
        sizes = [max(6, min(40, int(float(G.nodes[n].get("paper_count", 1))) * 2)) for n in nodes]
        texts = []
        for n in nodes:
            d = G.nodes[n]
            texts.append(
                f"{d.get('label', n)}<br>"
                f"Papers: {d.get('paper_count', 0)}<br>"
                f"% Positive: {d.get('pct_positive', 0)}%<br>"
                f"Degree: {G.degree(n)}<br>"
                f"Industry: {d.get('is_industry', False)}"
            )
        return go.Scatter(
            x=x, y=y,
            mode="markers",
            hoverinfo="text",
            text=texts,
            name=name,
            marker=dict(
                size=sizes,
                color=color,
                symbol=symbol,
                line=dict(width=1, color="white"),
            ),
        )

    ind_nodes = [n for n in G.nodes if str(G.nodes[n].get("is_industry", "false")).lower() == "true"]
    noind_nodes = [n for n in G.nodes if n not in ind_nodes]

    traces = [edge_trace]
    if noind_nodes:
        traces.append(make_node_trace(noind_nodes, INDEPENDENT_COLOR, "Independent", "circle"))
    if ind_nodes:
        traces.append(make_node_trace(ind_nodes, INDUSTRY_COLOR, "Industry-Affiliated", "diamond"))

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title="Co-Authorship Network: Industry vs Independent Researchers",
            titlefont_size=16,
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1200,
            height=800,
            plot_bgcolor="#f8f9fa",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        ),
    )

    fig.write_html(output_path)
    print(f"Plotly network saved to: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate interactive network visualizations")
    ap.add_argument("--network_dir", required=True, help="Dir with coauthor_network.graphml")
    ap.add_argument("--output_dir", required=True, help="Where to save HTML visualizations")
    ap.add_argument("--max_nodes", type=int, default=400, help="Max nodes to render (keeps top by centrality)")
    ap.add_argument("--min_degree", type=int, default=2, help="Min degree to include a node")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    graphml_path = os.path.join(args.network_dir, "coauthor_network.graphml")
    if not os.path.exists(graphml_path):
        raise SystemExit(f"Network file not found: {graphml_path}")

    G = load_network(graphml_path)
    print(f"Loaded network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    H = filter_for_viz(G, max_nodes=args.max_nodes, min_degree=args.min_degree)
    print(f"Filtered for viz: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    if H.number_of_nodes() == 0:
        print("[WARN] No nodes after filtering. Try lowering --min_degree.")
        return

    generate_pyvis(H, os.path.join(args.output_dir, "coauthor_network_interactive.html"))
    generate_plotly_network(H, os.path.join(args.output_dir, "coauthor_network_plotly.html"))


if __name__ == "__main__":
    main()
