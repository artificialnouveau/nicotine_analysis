# Part 2: Tobacco/Nicotine Industry Influence — Network Analysis & Visualization

In total, 2,175 tobacco and nicotine research papers published (and sourced on PubMed) between 1964 and 2025 were analyzed. In total, 7,006 authors were identified, and I mapped their co-authorship relationships into a network of 13,405 connections. Papers were classified into three categories based on the nature of their industry ties: **Tobacco Company** (111 papers, 5.1%) where at least one author had a direct affiliation with a known tobacco or nicotine company (e.g., Philip Morris, BAT, JUUL, R.J. Reynolds); **COI Declared** (425 papers, 19.5%) where authors disclosed financial conflicts of interest — such as consulting fees, advisory board membership, or industry funding — but were not directly employed by a tobacco company; and **Independent** (1,639 papers, 75.4%) with no disclosed industry ties. Industry affiliations were identified by matching author affiliations against known tobacco and nicotine companies using pattern-matching, and conflict-of-interest disclosure statements embedded in the publications were classified using keyword detection for phrases like "employee of," "funded by," "consultant," and "advisory board." The co-authorship network exhibits high industry assortativity (0.79), meaning tobacco-affiliated authors form tightly insular clusters, while the most central bridge scientists connecting disparate communities tend to be independent or COI-declaring researchers.

Outcome direction (positive, negative, neutral, or mixed) was determined using an enhanced keyword classifier that extracts conclusion-like sentences from full-text abstracts and classifies them using ~100 directional keyword patterns covering health outcomes, policy, epidemiology, marketing, cessation programs, and regulatory findings. Of the 1,075 papers with classifiable outcomes (49.4% of all papers — a substantial improvement over the initial 659 papers classified by the baseline keyword approach), the most striking finding was a split between the two industry-linked groups. Papers with **direct tobacco company ties showed no significant bias** — their rate of positive outcomes (36.5%) was comparable to independent papers (34.3%), with an odds ratio of 1.10 (p = 0.78). However, **COI-declared papers were significantly more likely to report positive outcomes** (43.2% vs 34.3%, OR = 1.46, p = 0.012). The overall 3-group chi-square test was also significant (chi2 = 14.03, p = 0.029), confirming that outcome distributions differ across the three categories. This suggests that the bias in tobacco and nicotine research may not originate from researchers directly employed by tobacco companies, but rather from the broader ecosystem of industry-funded consultants, grant recipients, and advisory board members whose financial relationships are disclosed but whose findings nonetheless skew favorably.

---

## What This Analysis Includes

1. **Data Loading & COI Identification** — Ingests 2,175 PubMed records from Part 1, extracts 7,006 unique authors and their affiliations, and classifies each paper into one of three categories: Tobacco Company ties, COI Declared (non-tobacco), or Independent.

2. **Enhanced Outcome Classification** — Uses an expanded keyword classifier with ~100 directional patterns applied to conclusion-like sentences extracted from full-text abstracts (not truncated). Covers health outcomes, policy/regulatory findings, cessation program effectiveness, industry criticism, and more. Classifies 1,075 papers (49.4%), up from 659 (30.3%) with the baseline approach.

3. **Co-Authorship Network Construction** — Builds an undirected weighted graph where nodes are authors and edges represent co-publication. Computes degree, betweenness, closeness, and eigenvector centrality for every author. Detects research communities via Louvain clustering.

4. **Statistical Testing** — Runs tests comparing all three groups: chi-square (3-group), pairwise Fisher exact tests, odds ratios with 95% CI, proportion z-tests, permutation tests, and Kruskal-Wallis + Mann-Whitney U for centrality metrics.

5. **Visualizations** — Generates six static plots (PNG), four interactive charts (HTML), and an interactive co-authorship network with color-coded nodes (three categories) and a legend.

6. **Streamlit Dashboard** — A full interactive dashboard for exploring all results, with filterable paper/author browsers, network views, and outcome charts.

---

## Three-Category Classification

Papers are classified into three mutually exclusive groups:

| Category | Definition | Papers |
|---|---|---|
| **Tobacco Company** | Author has verified affiliation with a known tobacco/nicotine company (Philip Morris, BAT, JUUL, R.J. Reynolds, Altria, etc.) OR the COI statement specifically names a tobacco company | 111 (5.1%) |
| **COI Declared** | Author declared a conflict of interest (consulting, honoraria, advisory boards, funding, etc.) but NOT tied to a tobacco company | 425 (19.5%) |
| **Independent** | No conflict of interest declared | 1,639 (75.4%) |

---

## Key Results

### Enhanced Outcome Classification

The enhanced classifier uses ~100 keyword patterns applied to conclusion-like sentences extracted from full abstracts. It classifies 1,075 of 2,175 papers (49.4%), compared to 659 (30.3%) with the original baseline approach. The remaining 1,100 papers either lack abstracts (296) or contain purely descriptive/methodological content without directional health or policy findings.

### Outcome Distribution (Enhanced Classifier)

| Outcome | Tobacco Company | COI Declared | Independent |
|---|---|---|---|
| Positive | 23 (20.7%) | 108 (25.4%) | 261 (15.9%) |
| Negative | 21 (18.9%) | 61 (14.4%) | 249 (15.2%) |
| Neutral | 3 (2.7%) | 21 (4.9%) | 37 (2.3%) |
| Mixed | 16 (14.4%) | 60 (14.1%) | 215 (13.1%) |
| Not applicable | 48 (43.2%) | 175 (41.2%) | 877 (53.5%) |

### Statistical Tests (Enhanced Classifier)

#### Tobacco Company vs Independent

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| Odds ratio (Positive) | 1.10 (95% CI: 0.65–1.88) | — | CI spans 1.0 — no significant association |
| Fisher exact | — | 0.783 | No significant difference |
| Proportion z-test | z = 0.36 | 0.717 | 36.5% vs 34.3% positive — not significant |
| Permutation test (two-sided) | — | 0.779 | Consistent with random assignment |

#### COI Declared vs Independent

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| Odds ratio (Positive) | **1.46** (95% CI: 1.09–1.95) | — | **Significant: CI does not span 1.0** |
| Fisher exact | — | **0.012** | **Significant difference** |
| Proportion z-test | z = 2.55 | **0.011** | 43.2% vs 34.3% positive — **significant** |
| Permutation test (two-sided) | — | **0.012** | **Not consistent with random assignment** |

#### Overall (3-group)

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| Chi-square (coded only, 3×4) | **14.03** | **0.029** | **Significant at α = 0.05** |

**Bottom line:** Papers with actual **tobacco company ties do not** show significantly more positive outcomes (OR = 1.10, p = 0.78). However, papers where authors **declared a non-tobacco COI** are **significantly more likely** to report positive findings (OR = 1.46, p = 0.012). With the enhanced classifier covering 1,075 papers (vs 659 previously), the overall 3-group chi-square is now significant (p = 0.029), confirming that outcome distributions genuinely differ across the three categories. This suggests the positive-outcome signal comes not from tobacco industry employment specifically, but from the broader category of researchers who declare financial conflicts of interest — potentially reflecting publication norms in industry-adjacent research or funding-related reporting biases that are not tobacco-specific.

### Network Structure

| Metric | Value |
|---|---|
| Connected components | 71 |
| Largest component | 1,470 authors (89.0%) |
| Average degree | 16.24 |
| Transitivity | 0.6258 |
| Louvain communities | 92 |
| **Industry assortativity** | **0.791** |

The interactive co-authorship network reveals a highly clustered research landscape where 89% of authors belong to a single giant connected component. Tobacco company-affiliated researchers (red diamonds) form tightly insular subgroups — reflected in an industry assortativity coefficient of 0.79 — while COI-declared authors (orange triangles) are more centrally embedded. The most central "bridge" authors connecting disparate research communities — such as Neal Benowitz, Thomas Eissenberg, and Maciej Goniewicz — tend to be COI-declared researchers with high betweenness centrality, while tobacco-affiliated authors sit at the network's periphery with significantly lower eigenvector centrality.

### Centrality Comparison (Kruskal-Wallis + Mann-Whitney U)

| Metric | Tobacco Co. Mean | COI Declared Mean | Independent Mean | KW p-value |
|---|---|---|---|---|
| Degree centrality | 0.0071 | 0.0151 | 0.0044 | **< 0.001** |
| Betweenness centrality | 0.0011 | 0.0036 | 0.0006 | **< 0.001** |
| Closeness centrality | 0.2375 | 0.4152 | 0.2685 | **< 0.001** |
| Eigenvector centrality | 0.000004 | 0.0097 | 0.0005 | **< 0.001** |

---

## Visualizations

### Static Plots

#### Outcome Counts by Author Category
![Outcome by Industry](output/viz/figures/outcome_by_industry.png)

#### Outcome Proportions (%)
![Outcome Proportions](output/viz/figures/outcome_proportions.png)

#### Publication Timeline by Author Category
![Timeline](output/viz/figures/timeline_industry_papers.png)

#### Odds Ratio Forest Plot
![Odds Ratio](output/viz/figures/odds_ratio_forest.png)

#### Network Centrality by Author Category
![Centrality Distribution](output/viz/figures/centrality_distribution.png)

#### Community-Level: Industry Concentration vs Positive Outcomes
![Community Scatter](output/viz/figures/community_industry_scatter.png)

### Interactive Visualizations (HTML)

Open these files in a browser for full interactivity:

- **[Co-Authorship Network (pyvis)](https://artificialnouveau.github.io/nicotine_analysis/part2/output/viz/coauthor_network_interactive.html)** — Drag, zoom, and hover over the network. Tobacco company authors are red diamonds, COI-declared are orange triangles, independent are blue circles. Node size reflects paper count; edge thickness reflects shared publications.

- **[Co-Authorship Network (plotly)](https://artificialnouveau.github.io/nicotine_analysis/part2/output/viz/coauthor_network_plotly.html)** — Spring-layout network with hover tooltips showing author name, paper count, % positive outcomes, and category.

- **[Sankey Diagram: Author Category to Outcomes](https://artificialnouveau.github.io/nicotine_analysis/part2/output/viz/figures/sankey_funding_outcome.html)** — Flow diagram showing how papers from each category distribute across outcome categories.

- **[Outcome Heatmap Over Time](https://artificialnouveau.github.io/nicotine_analysis/part2/output/viz/figures/outcome_heatmap.html)** — Year-by-year heatmap of % positive outcomes for all three categories from 1990–present.

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
│   ├── load_and_identify.py         # Load part1 data, identify COI/industry ties
│   └── enhanced_classifier.py       # Enhanced outcome classification (~100 patterns)
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
    ├── stats/                       # full_statistics.json, outcome_comparison.csv,
    │                                # enhanced_outcome_classifications.csv
    └── viz/                         # PNG plots + interactive HTML charts
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (reads from part1 output)
python run_pipeline.py --part1_dir ../part1/study_dump --output_dir ./output

# Run enhanced outcome classification
python data/enhanced_classifier.py \
  --papers_csv output/data/papers.csv \
  --jsonl_path ../part1/study_dump/data/pubmed_records.jsonl \
  --output_csv output/stats/enhanced_outcome_classifications.csv

# Launch the dashboard
streamlit run viz/app.py
```

---

## Methods

- **Three-category classification**: Papers are classified as "Tobacco Company" (author has verified affiliation with one of 15 known tobacco/nicotine organizations, or COI statement names a tobacco company), "COI Declared" (author disclosed a conflict of interest that is not tobacco-specific), or "Independent" (no COI declared).
- **Enhanced outcome classification**: An expanded keyword classifier extracts conclusion-like sentences from full-text abstracts using 30+ cue phrases (e.g., "we conclude," "our findings suggest," "results indicate"), then matches them against ~100 directional patterns covering health outcomes (harm reduction, cessation, toxicity, cancer), policy/regulatory findings, industry criticism, and epidemiological trends. Conclusion-only matching prevents background/introduction text from polluting results. Priority order: Mixed > Neutral > Negative > Positive > Not applicable.
- **Network analysis**: Co-authorship graph built with NetworkX; community detection via Louvain algorithm; centrality metrics include degree, betweenness (with inverted weights so stronger ties = shorter paths), closeness, and eigenvector centrality.
- **Statistical tests**: Chi-square test on the 3×4 table (coded outcomes only); pairwise Fisher exact, odds ratio with 95% CI, proportion z-test, and two-sided permutation test (10,000 iterations) for each group vs Independent; Kruskal-Wallis and Mann-Whitney U for centrality comparisons.
