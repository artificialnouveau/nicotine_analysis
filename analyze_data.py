#!/usr/bin/env python3
"""
analyze_data.py

Load JSONL records from fetch_data.py and perform:
  - Subject classification (e-cigarette / tobacco / both / neither)
  - COI proxy classification from coi_statement text
  - Abstract sentiment cluster classification (positive/negative/neutral + key sentences)
  - Organization/entity extraction from COI text (manual list; optional Transformers NER)
  - Co-authorship network degree centrality
  - Output tables + figures

Outputs (under --out_dir):
  analysis/
    merged_records.csv
    author_counts_top30.csv
    entity_counts.csv
    author_centrality.csv
    figures/
      figure1_top30_authors.png
      figure2_coi_entities.png
      figure3_sentiment_by_coi.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# -------------------------
# Load JSONL
# -------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s


# -------------------------
# COI proxy classifier
# -------------------------

NO_COI_PHRASES = [
    "no conflict of interest",
    "no conflicts of interest",
    "declare no conflict",
    "declares no conflict",
    "no competing interests",
    "no competing interest",
    "the authors declare no conflict",
    "the authors declare that they have no conflict",
    "the authors report no conflict",
    "nothing to disclose",
    "no disclosure",
]

YES_COI_PHRASES = [
    "received funding",
    "received grant",
    "supported by",
    "funded by",
    "consultant",
    "consulting",
    "honoraria",
    "speaker",
    "advisory board",
    "employee",
    "employment",
    "stock",
    "equity",
    "patent",
    "paid expert witness",
    "served as expert witness",
    "litigation",
    "financial relationship",
    "conflict of interest",
    "competing interests",
    "disclosure",
]


def classify_conflict(text: Optional[str]) -> str:
    """
    Returns:
      - 'Yes' if likely COI
      - 'No' if explicitly states none
    Heuristic mirrors your description: empty => 'No', unknown => default 'Yes'
    """
    t = clean_text(text).lower()
    if not t:
        return "No"
    for p in NO_COI_PHRASES:
        if p in t:
            return "No"
    for p in YES_COI_PHRASES:
        if p in t:
            return "Yes"
    # Default assumption per your description:
    return "Yes"


# -------------------------
# Subject classifier
# -------------------------

ECIG_KWS = [
    "e-cig", "ecig", "electronic cigarette", "electronic nicotine", "vape", "vaping",
    "ends", "electronic nicotine delivery", "e liquid", "e-liquid", "juul"
]
TOBACCO_KWS = [
    "tobacco", "cigarette", "smoking", "smoker", "nicotine replacement", "combustible",
    "cigar", "cigars", "cigarillo"
]

def classify_subject(abstract: Optional[str], title: Optional[str]) -> str:
    text = (clean_text(title) + " " + clean_text(abstract)).lower()
    has_ecig = any(k in text for k in ECIG_KWS)
    has_tob = any(k in text for k in TOBACCO_KWS)
    if has_ecig and has_tob:
        return "Both"
    if has_ecig:
        return "E-cigarette"
    if has_tob:
        return "Tobacco"
    return "Neither"


# -------------------------
# Abstract sentiment clustering (heuristic)
# -------------------------

KEYWORD_CLUSTERS = {
    "positive_cessation": [
        "cessation", "quit", "quitting", "abstinence", "reduced smoking", "harm reduction",
        "switching", "reduction in cigarettes", "reduced exposure", "reduced harm"
    ],
    "positive_biomarkers": [
        "lower levels", "reduced biomarkers", "improved", "decreased exposure", "less toxic",
        "reduced toxicant", "reduced risk"
    ],
    "negative_effects": [
        "increased risk", "harmful", "toxic", "adverse", "inflammation", "oxidative stress",
        "respiratory", "cardiovascular", "lung injury", "dependence", "addiction", "carcinogen"
    ],
    "neutral_mixed": [
        "mixed", "inconclusive", "uncertain", "no significant", "not significant", "further research",
        "limited evidence", "complex", "heterogeneous"
    ],
}

SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")

def classify_abstract_sentiment_clusters(abstract: Optional[str]) -> Tuple[str, str]:
    """
    Returns (cluster_label, supporting_sentences)
    If not relevant to ecig/tobacco keywords => ('Not relevant', '')
    Otherwise assigns based on first matching cluster priority:
      negative_effects > positive_cessation/positive_biomarkers > neutral_mixed
    """
    a = clean_text(abstract)
    if not a:
        return ("Not relevant", "")

    low = a.lower()
    if not (any(k in low for k in ECIG_KWS) or any(k in low for k in TOBACCO_KWS) or "nicotine" in low):
        return ("Not relevant", "")

    sentences = SENTENCE_SPLIT.split(a)
    hits = defaultdict(list)

    for s in sentences:
        s_low = s.lower()
        for cluster, kws in KEYWORD_CLUSTERS.items():
            if any(k in s_low for k in kws):
                hits[cluster].append(s.strip())

    # Priority rule
    if hits["negative_effects"]:
        return ("negative_effects", " | ".join(hits["negative_effects"][:3]))
    if hits["positive_cessation"] or hits["positive_biomarkers"]:
        ss = hits["positive_cessation"][:2] + hits["positive_biomarkers"][:2]
        return ("positive", " | ".join(ss[:3]))
    if hits["neutral_mixed"]:
        return ("neutral_mixed", " | ".join(hits["neutral_mixed"][:3]))

    return ("neutral_mixed", "")


# -------------------------
# Entity extraction (manual + optional Transformers NER)
# -------------------------

MANUAL_ORGS = [
    "Philip Morris", "Philip Morris International", "PMI",
    "Altria", "British American Tobacco", "BAT", "RJ Reynolds", "RJR",
    "Japan Tobacco", "JT", "Imperial Brands", "Imperial Tobacco",
    "JUUL", "Juul Labs",
    "Foundation for a Smoke-Free World",
    "American Cancer Society", "Truth Initiative",
]

def find_entities_manual(text: str) -> List[str]:
    t = clean_text(text)
    if not t:
        return []
    found = []
    for org in MANUAL_ORGS:
        if re.search(r"\b" + re.escape(org) + r"\b", t, flags=re.IGNORECASE):
            found.append(org)
    return found


def build_transformers_ner():
    """
    Optional: requires `pip install transformers torch`
    Uses a general-purpose NER model; results are heuristic for ORGs.
    """
    from transformers import pipeline
    return pipeline("ner", grouped_entities=True)


def find_entities_transformers_with_manual_list(text: str, ner_pipe=None) -> List[str]:
    ents = set(find_entities_manual(text))
    if ner_pipe is None:
        return sorted(ents)

    t = clean_text(text)
    if not t:
        return sorted(ents)

    try:
        ner = ner_pipe(t)
        for item in ner:
            label = (item.get("entity_group") or item.get("entity") or "").upper()
            if "ORG" in label:
                w = clean_text(item.get("word"))
                if w and len(w) >= 3:
                    ents.add(w)
    except Exception:
        pass

    return sorted(ents)


# -------------------------
# Authors parsing
# -------------------------

def extract_author_names(authors_field: Any) -> List[str]:
    """
    Works with fetch_data.py UnifiedRecord authors only for PubMed if you later add them,
    but also tolerates missing. Current fetch_data.py stores PubMed authors in XML parse
    only if you extend it; so we support a conservative fallback.

    If your JSONL contains `authors` as list-of-dicts with last/fore, we convert.
    """
    if not authors_field:
        return []
    if isinstance(authors_field, list):
        out = []
        for a in authors_field:
            if isinstance(a, dict):
                last = clean_text(a.get("last"))
                fore = clean_text(a.get("fore"))
                full = clean_text((fore + " " + last).strip())
                if full:
                    out.append(full)
            elif isinstance(a, str):
                out.append(clean_text(a))
        return [x for x in out if x]
    if isinstance(authors_field, str):
        # if someone stored as a single string, split roughly
        parts = [p.strip() for p in authors_field.split(",") if p.strip()]
        return parts
    return []


# -------------------------
# Main analysis
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="The same out_dir used by fetch_data.py")
    ap.add_argument("--include_wos", action="store_true", help="Include WoS records if present.")
    ap.add_argument("--use_transformers_ner", action="store_true", help="Use Transformers NER in addition to manual org list.")
    args = ap.parse_args()

    data_dir = os.path.join(args.out_dir, "data")
    analysis_dir = os.path.join(args.out_dir, "analysis")
    fig_dir = os.path.join(analysis_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    pubmed_path = os.path.join(data_dir, "pubmed_records.jsonl")
    wos_path = os.path.join(data_dir, "wos_records.jsonl")

    pubmed_rows = read_jsonl(pubmed_path)
    wos_rows = read_jsonl(wos_path) if args.include_wos else []

    rows = pubmed_rows + wos_rows
    if not rows:
        raise SystemExit("No records found. Run fetch_data.py first.")

    df = pd.DataFrame(rows)

    # Ensure expected columns exist
    for col in ["title", "abstract", "coi_statement", "year", "journal", "doi", "record_id", "source_db"]:
        if col not in df.columns:
            df[col] = None

    # Classifications
    df["COI_Proxy"] = df["coi_statement"].apply(classify_conflict)
    df["Subject"] = df.apply(lambda r: classify_subject(r.get("abstract"), r.get("title")), axis=1)
    sent = df["abstract"].apply(classify_abstract_sentiment_clusters)
    df["Sentiment_Cluster"] = sent.apply(lambda x: x[0])
    df["Sentiment_Sentence"] = sent.apply(lambda x: x[1])

    # Entity extraction from COI statement text (proxy)
    ner_pipe = build_transformers_ner() if args.use_transformers_ner else None
    df["COI_Entities"] = df["coi_statement"].apply(lambda t: find_entities_transformers_with_manual_list(t or "", ner_pipe))

    # Save merged table
    merged_csv = os.path.join(analysis_dir, "merged_records.csv")
    df.to_csv(merged_csv, index=False)

    # ----- Figure 3: sentiment by COI proxy -----
    pivot = pd.crosstab(df["COI_Proxy"], df["Sentiment_Cluster"])
    pivot.to_csv(os.path.join(analysis_dir, "sentiment_by_coi.csv"))
    ax = pivot.plot(kind="bar")
    ax.set_title("Sentiment Cluster by COI Proxy")
    ax.set_xlabel("COI Proxy")
    ax.set_ylabel("Paper count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure3_sentiment_by_coi.png"), dpi=200)
    plt.close()

    # ----- Entities vs COI proxy (Figure 2) -----
    ent_counter = Counter()
    for ents in df["COI_Entities"]:
        for e in ents or []:
            ent_counter[e] += 1
    ent_df = pd.DataFrame(ent_counter.most_common(50), columns=["Entity", "Count"])
    ent_df.to_csv(os.path.join(analysis_dir, "entity_counts.csv"), index=False)

    ax = ent_df.head(20).set_index("Entity")["Count"].plot(kind="bar")
    ax.set_title("Top Entities Mentioned in COI Statements (proxy)")
    ax.set_xlabel("Entity")
    ax.set_ylabel("Mentions")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure2_coi_entities.png"), dpi=200)
    plt.close()

    # ----- Authors + coauthorship network -----
    # If your JSONL lacks authors, this section will produce sparse results.
    if "authors" not in df.columns:
        df["authors"] = None

    df["Author_List"] = df["authors"].apply(extract_author_names)

    # Top 30 authors (Figure 1)
    author_counts = Counter()
    author_coi = defaultdict(lambda: Counter())  # author -> Yes/No counts
    for _, row in df.iterrows():
        coi = row.get("COI_Proxy", "No")
        for a in row["Author_List"] or []:
            author_counts[a] += 1
            author_coi[a][coi] += 1

    top30 = author_counts.most_common(30)
    top_df = pd.DataFrame(
        [{
            "Author": a,
            "PaperCount": c,
            "COI_Yes": int(author_coi[a]["Yes"]),
            "COI_No": int(author_coi[a]["No"]),
        } for a, c in top30]
    )
    top_df.to_csv(os.path.join(analysis_dir, "author_counts_top30.csv"), index=False)

    ax = top_df.set_index("Author")["PaperCount"].plot(kind="bar")
    ax.set_title("Top 30 Authors by Publication Count")
    ax.set_xlabel("Author")
    ax.set_ylabel("Paper count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure1_top30_authors.png"), dpi=200)
    plt.close()

    # Co-authorship network centrality
    G = nx.Graph()
    for _, row in df.iterrows():
        authors = row["Author_List"] or []
        authors = [a for a in authors if a]
        for a in authors:
            G.add_node(a)
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                G.add_edge(authors[i], authors[j])

    centrality = nx.degree_centrality(G) if G.number_of_nodes() > 0 else {}
    cent_df = pd.DataFrame(
        [{"Author": a, "DegreeCentrality": float(v), "PaperCount": int(author_counts.get(a, 0))}
         for a, v in centrality.items()]
    ).sort_values(["DegreeCentrality", "PaperCount"], ascending=False)

    cent_df.to_csv(os.path.join(analysis_dir, "author_centrality.csv"), index=False)

    print("Analysis complete.")
    print(f"- Merged table: {merged_csv}")
    print(f"- Figures: {fig_dir}")


if __name__ == "__main__":
    main()

