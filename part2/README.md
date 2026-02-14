# Part 2: Tobacco/Nicotine Industry Influence — Network Analysis & Visualization

This analysis builds a co-authorship network of scientific authors in the tobacco and nicotine research space, identifies those with conflicts of interest and/or industry ties, and statistically tests whether their findings skew more positive/beneficial compared to independent researchers.

---

## What This Analysis Includes

1. **Data Loading & COI Identification** — Ingests 2,175 PubMed records from Part 1, extracts 7,006 unique authors and their affiliations, and flags industry ties by matching against known tobacco/nicotine companies (Philip Morris, BAT, JUUL, R.J. Reynolds, Altria, etc.) and COI disclosure statements.

2. **Co-Authorship Network Construction** — Builds an undirected weighted graph where nodes are authors and edges represent co-publication. Computes degree, betweenness, closeness, and eigenvector centrality for every author. Detects research communities via Louvain clustering.

3. **Statistical Testing** — Runs five complementary tests comparing industry-involved vs independent papers: chi-square, Fisher exact, odds ratio with 95% CI, two-proportion z-test, and a 10,000-iteration permutation test.

4. **Visualizations** — Generates six static plots (PNG), four interactive charts (HTML), and an interactive co-authorship network with color-coded nodes and a legend.

5. **Streamlit Dashboard** — A full interactive dashboard for exploring all results, with filterable paper/author browsers, network views, and outcome charts.

---

## Dataset Summary

| Metric | Value |
|---|---|
| Total papers analyzed | 2,175 |
| Industry-involved papers | 699 (32.1%) |
| Independent papers | 1,476 (67.9%) |
| Total unique authors | 7,006 |
| Industry-affiliated authors | 241 (3.4%) |
| Authors in network (2+ papers) | 1,651 |
| Co-authorship edges | 13,405 |

---

## Key Results

### Outcome Distribution

| Outcome | Industry-Involved | Independent |
|---|---|---|
| Positive | 106 (15.2%) | 238 (16.1%) |
| Negative | 46 (6.6%) | 139 (9.4%) |
| Neutral | 13 (1.9%) | 14 (0.9%) |
| Mixed | 24 (3.4%) | 79 (5.4%) |
| Not coded | 510 (73.0%) | 1,006 (68.2%) |

### Statistical Tests

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| Chi-square (full table) | 13.20 | **0.010** | Overall outcome distribution differs significantly |
| Fisher exact (Positive vs Not) | — | 0.615 | No significant difference in positive rate specifically |
| Odds ratio (Positive) | 0.93 (95% CI: 0.72–1.19) | — | CI spans 1.0 — no significant association |
| Proportion z-test | z = −0.57 | 0.567 | Industry: 15.2% positive vs Independent: 16.1% — not significant |
| Permutation test (n=10,000) | — | 0.743 | Observed rate consistent with random assignment |

**Bottom line:** Industry-involved papers do **not** show significantly more positive outcomes. However, the overall outcome distribution does differ (p = 0.01) — industry papers report **fewer negative findings** (6.6% vs 9.4%) and **fewer mixed results** (3.4% vs 5.4%), suggesting a potential bias in how negative results are reported or framed rather than an inflation of positive ones.

### Network Structure

| Metric | Value |
|---|---|
| Connected components | 71 |
| Largest component | 1,470 authors (89.0%) |
| Average degree | 16.24 |
| Transitivity | 0.6258 |
| Louvain communities | 92 |
| **Industry assortativity** | **0.8086** |

The interactive co-authorship network reveals a highly clustered research landscape where 89% of authors belong to a single giant connected component, yet industry-affiliated researchers (red diamonds) form tightly insular subgroups — reflected in an industry assortativity coefficient of 0.81, meaning tobacco/nicotine industry-tied authors overwhelmingly co-publish with each other rather than with independent scientists. The most central "bridge" authors connecting disparate research communities — such as Neal Benowitz, Thomas Eissenberg, and Maciej Goniewicz — are all independent researchers, while industry-affiliated authors sit at the network's periphery with significantly lower eigenvector centrality (p < 0.001). This structural segregation across 92 detected communities suggests two largely parallel research ecosystems operating within the same scientific field, with limited cross-pollination between industry-funded and independent groups.

### Centrality Comparison (Mann-Whitney U)

| Metric | Industry Mean | Independent Mean | p-value |
|---|---|---|---|
| Degree centrality | 0.0070 | 0.0100 | 0.416 |
| Betweenness centrality | 0.0017 | 0.0017 | 0.188 |
| Closeness centrality | 0.1707 | 0.2046 | **< 0.001** |
| Eigenvector centrality | 0.000004 | 0.0053 | **< 0.001** |

---

## Visualizations

### Static Plots

#### Outcome Counts by Industry Involvement
![Outcome by Industry](output/viz/figures/outcome_by_industry.png)

#### Outcome Proportions (%)
![Outcome Proportions](output/viz/figures/outcome_proportions.png)

#### Publication Timeline: Industry vs Independent
![Timeline](output/viz/figures/timeline_industry_papers.png)

#### Odds Ratio Forest Plot
![Odds Ratio](output/viz/figures/odds_ratio_forest.png)

#### Network Centrality: Industry vs Independent
![Centrality Distribution](output/viz/figures/centrality_distribution.png)

#### Community-Level: Industry Concentration vs Positive Outcomes
![Community Scatter](output/viz/figures/community_industry_scatter.png)

### Interactive Visualizations (HTML)

Open these files in a browser for full interactivity:

- **[Co-Authorship Network (pyvis)](output/viz/coauthor_network_interactive.html)** — Drag, zoom, and hover over the network. Industry authors are red diamonds, independent are blue circles. Includes a color legend. Node size reflects paper count; edge thickness reflects shared publications.

- **[Co-Authorship Network (plotly)](output/viz/coauthor_network_plotly.html)** — Spring-layout network with hover tooltips showing author name, paper count, % positive outcomes, and industry status.

- **[Sankey Diagram: Industry Status to Outcomes](output/viz/figures/sankey_funding_outcome.html)** — Flow diagram showing how papers from industry-involved vs independent groups distribute across outcome categories.

- **[Outcome Heatmap Over Time](output/viz/figures/outcome_heatmap.html)** — Year-by-year heatmap of % positive outcomes for industry vs independent papers from 1990–present.

### Streamlit Dashboard

A full interactive dashboard combining all analyses:

```bash
cd /Users/ahnjili_harmony/Documents/GitHub/nicotine_analysis/part2
streamlit run viz/app.py
```

Then open **http://localhost:8501** in your browser. The dashboard includes:
- Overview with key metrics and statistical findings
- Outcome comparison charts with interactive filters
- Network centrality explorer and community breakdown
- Searchable paper browser
- Searchable author explorer

---

## Project Structure

```
part2/
├── README.md
├── requirements.txt
├── run_pipeline.py                  # Master pipeline (runs all steps)
├── data/
│   └── load_and_identify.py         # Load part1 data, identify COI/industry ties
├── analysis/
│   ├── network.py                   # Build co-authorship network + communities
│   └── statistics.py                # Chi-square, Fisher, OR, permutation tests
├── viz/
│   ├── plots.py                     # Static PNG charts
│   ├── network_viz.py               # Interactive HTML network graphs
│   └── app.py                       # Streamlit dashboard
└── output/
    ├── data/                        # authors.csv, papers.csv, author_papers.csv
    ├── network/                     # GraphML, GEXF, centrality, communities
    ├── stats/                       # full_statistics.json, outcome_comparison.csv
    └── viz/                         # PNG plots + interactive HTML charts
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (reads from part1 output)
python run_pipeline.py --part1_dir ../part1/study_dump --output_dir ./output

# Launch the dashboard
streamlit run viz/app.py
```

---

## Methods

- **Industry identification**: Author affiliations are matched against 15 known tobacco/nicotine industry organizations using regex patterns. COI statements are classified using keyword matching against established disclosure phrases.
- **Outcome coding**: Conclusion-like sentences are extracted from structured abstracts (preferring labeled CONCLUSIONS sections), then classified as Positive/Negative/Neutral/Mixed using directional keyword patterns (e.g., "harm reduction", "cessation" = positive; "carcinogen", "increased risk" = negative).
- **Network analysis**: Co-authorship graph built with NetworkX; community detection via Louvain algorithm; centrality metrics include degree, betweenness, closeness, and eigenvector centrality.
- **Statistical tests**: Chi-square test on the full 2x5 contingency table; Fisher exact test on the collapsed 2x2 (Positive vs Not-Positive); odds ratio with Wald 95% CI; two-proportion z-test; permutation test with 10,000 iterations.
