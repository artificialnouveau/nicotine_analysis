#!/usr/bin/env python3
"""
load_and_identify.py

Load records from part1's JSONL output, identify COI/industry ties per author,
and produce a unified author-level and paper-level dataset for network analysis.

Outputs (to --output_dir):
  - authors.csv          : one row per unique author with COI flags
  - papers.csv           : one row per paper with outcome + industry flags
  - author_papers.csv    : junction table (author_id, paper_id, is_industry_author)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Industry affiliation patterns (mirrors part1, extended)
# ---------------------------------------------------------------------------

INDUSTRY_ORG_PATTERNS = [
    (r"\bphilip\s+morris\b|\bpmi\b|\bphilip\s+morris\s+international\b", "Philip Morris International"),
    (r"\baltria\b", "Altria"),
    (r"\bbritish\s+american\s+tobacco\b|\bbat\b", "British American Tobacco"),
    (r"\bjapan\s+tobacco\b|\bjti\b", "Japan Tobacco International"),
    (r"\bimperial\s+brands\b|\bimperial\s+tobacco\b", "Imperial Brands"),
    (r"\brj\s*reynolds\b|\breynolds\s+american\b|\brjr\b", "R.J. Reynolds"),
    (r"\bswedish\s+match\b", "Swedish Match"),
    (r"\bjuul\b|\bjuul\s+labs\b", "JUUL Labs"),
    (r"\blorillard\b", "Lorillard"),
    (r"\bvector\s+group\b|\bliggett\b", "Vector Group / Liggett"),
    (r"\bstar\s+scientific\b", "Star Scientific"),
    (r"\bcouncil\s+for\s+tobacco\s+research\b|\bctr\b", "Council for Tobacco Research"),
    (r"\btobacco\s+institute\b", "Tobacco Institute"),
    (r"\bfoundation\s+for\s+a\s+smoke[-\s]?free\s+world\b", "Foundation for a Smoke-Free World"),
    (r"\brav?i\b.*\btobacco\b", "RAI / Reynolds American"),
]

# COI declaration patterns
NO_COI_PHRASES = [
    "no conflict of interest", "no conflicts of interest",
    "no competing interests", "no competing interest",
    "declare no conflict", "declares no conflict",
    "nothing to disclose", "no disclosures",
    "the authors declare no", "authors declare no",
]

YES_COI_PHRASES = [
    "received funding from", "funded by", "supported by", "grant from",
    "consultant", "honoraria", "speaker", "advisory board",
    "employee of", "employment", "stock", "equity", "patent",
    "royalties", "expert witness", "litigation",
]


def clean(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def author_id(last: str, fore: str, initials: str) -> str:
    """Create a stable author identifier from name parts."""
    key = f"{clean(last).lower()}|{clean(fore).lower()}|{clean(initials).lower()}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def author_display_name(last: str, fore: str) -> str:
    parts = [clean(last), clean(fore)]
    return ", ".join(p for p in parts if p)


def check_industry_affiliation(affiliations: List[str]) -> Tuple[bool, List[str]]:
    """Check if any affiliation matches a known tobacco/nicotine industry org."""
    joined = " ".join(clean(a) for a in affiliations).lower()
    matched = []
    for pattern, org_name in INDUSTRY_ORG_PATTERNS:
        if re.search(pattern, joined, re.IGNORECASE):
            matched.append(org_name)
    return (len(matched) > 0, matched)


def classify_coi_statement(coi_text: Optional[str]) -> str:
    """Returns 'Yes', 'No', or 'Missing'."""
    t = clean(coi_text).lower()
    if not t:
        return "Missing"
    for phrase in NO_COI_PHRASES:
        if phrase in t:
            return "No"
    for phrase in YES_COI_PHRASES:
        if phrase in t:
            return "Yes"
    # Has text but no clear signal — conservative: treat as declared
    return "Yes"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


# ---------------------------------------------------------------------------
# Outcome coding (same logic as part1 for consistency)
# ---------------------------------------------------------------------------

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

POS_PATTERNS = [
    r"\bcessation\b", r"\bquit\b", r"\bquitting\b", r"\babstinence\b",
    r"\bharm\s+reduction\b", r"\breduced\s+(exposure|harm|risk|toxic)\b",
    r"\blower\s+(exposure|levels|biomarkers)\b",
    r"\bimproved\b", r"\bbenefit\b", r"\bsafer\b", r"\bless\s+harmful\b",
    r"\bdecreased\s+(odds|risk)\b", r"\blower\s+(odds|risk)\b",
]
NEG_PATTERNS = [
    r"\bharmful\b", r"\btoxic\b", r"\badverse\b",
    r"\bincreased\s+(risk|harm|exposure|odds)\b",
    r"\bhigher\s+(odds|risk)\b", r"\bcarcinogen\b", r"\bcancer\b",
    r"\baddiction\b", r"\bdependence\b",
]
NEUTRAL_PATTERNS = [
    r"\bno\s+significant\b", r"\bnot\s+significant\b",
    r"\binconclusive\b", r"\bmixed\b", r"\bunclear\b",
    r"\bno\s+difference\b", r"\bno\s+association\b",
    r"\bfurther\s+research\b",
]

CONCLUSION_CUES = [
    "conclusion", "we conclude", "in conclusion",
    "our findings", "these findings", "results suggest",
    "results indicate", "we found", "overall", "therefore",
]


def extract_conclusion_sentences(abstract: str, max_sents: int = 6) -> List[str]:
    a = clean(abstract)
    if not a:
        return []
    sents = [s.strip() for s in SENT_SPLIT.split(a) if s.strip()]
    cue_hits = [s for s in sents if any(c in s.lower() for c in CONCLUSION_CUES)]
    if cue_hits:
        return cue_hits[:max_sents]
    return sents[-min(3, len(sents)):]


def code_outcome(sentences: List[str]) -> str:
    pos = any(re.search(p, s.lower()) for s in sentences for p in POS_PATTERNS)
    neg = any(re.search(p, s.lower()) for s in sentences for p in NEG_PATTERNS)
    neu = any(re.search(p, s.lower()) for s in sentences for p in NEUTRAL_PATTERNS)

    if pos and neg:
        return "Mixed"
    if neg:
        return "Negative"
    if pos:
        return "Positive"
    if neu:
        return "Neutral"
    return "Not coded"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_records(records: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process raw records into three DataFrames:
      - authors_df: unique authors with industry flags
      - papers_df: papers with outcomes and industry flags
      - edges_df: author-paper junction table
    """
    authors_map: Dict[str, Dict] = {}   # author_id -> author info
    papers_list: List[Dict] = []
    edges_list: List[Dict] = []

    for rec in records:
        paper_id = rec.get("record_id") or rec.get("pmid") or ""
        if not paper_id:
            continue

        title = clean(rec.get("title"))
        abstract = clean(rec.get("abstract"))
        year = rec.get("year")
        journal = clean(rec.get("journal"))
        doi = clean(rec.get("doi"))
        coi_statement = clean(rec.get("coi_statement"))

        # Outcome coding
        conclusion_sents = extract_conclusion_sentences(abstract)
        outcome = code_outcome(conclusion_sents)

        # COI classification
        declared_coi = classify_coi_statement(coi_statement)

        # Process authors
        authors_raw = rec.get("authors") or []
        paper_has_industry_author = False
        paper_industry_orgs: List[str] = []
        paper_author_ids: List[str] = []

        for auth in authors_raw:
            if not isinstance(auth, dict):
                continue

            last = clean(auth.get("last", ""))
            fore = clean(auth.get("fore", ""))
            initials = clean(auth.get("initials", ""))
            affiliations = auth.get("affiliations") or []

            if not last and not fore:
                continue

            aid = author_id(last, fore, initials)
            is_industry, matched_orgs = check_industry_affiliation(affiliations)

            if is_industry:
                paper_has_industry_author = True
                paper_industry_orgs.extend(matched_orgs)

            # Update author record (accumulate across papers)
            if aid not in authors_map:
                authors_map[aid] = {
                    "author_id": aid,
                    "name": author_display_name(last, fore),
                    "last": last,
                    "fore": fore,
                    "initials": initials,
                    "affiliations": [],
                    "is_industry_affiliated": False,
                    "industry_orgs": set(),
                    "paper_count": 0,
                }

            entry = authors_map[aid]
            entry["affiliations"].extend(affiliations)
            entry["paper_count"] += 1
            if is_industry:
                entry["is_industry_affiliated"] = True
                entry["industry_orgs"].update(matched_orgs)

            paper_author_ids.append(aid)

            edges_list.append({
                "author_id": aid,
                "paper_id": paper_id,
                "is_industry_author": is_industry,
            })

        # Determine industry involvement (affiliation OR declared COI)
        industry_involved = paper_has_industry_author or declared_coi == "Yes"

        papers_list.append({
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract[:500],  # truncate for CSV size
            "year": year,
            "journal": journal,
            "doi": doi,
            "coi_statement": coi_statement[:300] if coi_statement else "",
            "declared_coi": declared_coi,
            "has_industry_author": paper_has_industry_author,
            "industry_orgs": "; ".join(sorted(set(paper_industry_orgs))),
            "industry_involved": "Yes" if industry_involved else "No",
            "outcome": outcome,
            "author_count": len(paper_author_ids),
            "author_ids": ";".join(paper_author_ids),
        })

    # Build DataFrames
    papers_df = pd.DataFrame(papers_list)

    # Finalize authors
    for aid, info in authors_map.items():
        info["industry_orgs"] = "; ".join(sorted(info["industry_orgs"]))
        # Deduplicate affiliations
        seen = set()
        unique_affs = []
        for a in info["affiliations"]:
            a_clean = clean(a)
            if a_clean and a_clean.lower() not in seen:
                seen.add(a_clean.lower())
                unique_affs.append(a_clean)
        info["affiliations"] = "; ".join(unique_affs[:5])  # keep top 5

    authors_df = pd.DataFrame(list(authors_map.values()))
    edges_df = pd.DataFrame(edges_list)

    return authors_df, papers_df, edges_df


def main():
    ap = argparse.ArgumentParser(description="Load part1 data and build author/paper tables for network analysis")
    ap.add_argument("--input_dir", required=True, help="part1 output dir containing data/pubmed_records.jsonl")
    ap.add_argument("--output_dir", required=True, help="Where to write authors.csv, papers.csv, author_papers.csv")
    ap.add_argument("--include_wos", action="store_true", help="Also load wos_records.jsonl if present")
    args = ap.parse_args()

    data_dir = os.path.join(args.input_dir, "data")
    pubmed_path = os.path.join(data_dir, "pubmed_records.jsonl")
    wos_path = os.path.join(data_dir, "wos_records.jsonl")

    records = read_jsonl(pubmed_path)
    if args.include_wos and os.path.exists(wos_path):
        records.extend(read_jsonl(wos_path))

    if not records:
        raise SystemExit(f"No records found in {data_dir}. Run part1/fetch_data.py first.")

    print(f"Loaded {len(records)} records")

    authors_df, papers_df, edges_df = process_records(records)

    os.makedirs(args.output_dir, exist_ok=True)

    authors_path = os.path.join(args.output_dir, "authors.csv")
    papers_path = os.path.join(args.output_dir, "papers.csv")
    edges_path = os.path.join(args.output_dir, "author_papers.csv")

    authors_df.to_csv(authors_path, index=False)
    papers_df.to_csv(papers_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    n_industry = authors_df["is_industry_affiliated"].sum() if "is_industry_affiliated" in authors_df.columns else 0
    n_papers_ind = (papers_df["industry_involved"] == "Yes").sum() if not papers_df.empty else 0

    print(f"Authors: {len(authors_df)} ({n_industry} industry-affiliated)")
    print(f"Papers:  {len(papers_df)} ({n_papers_ind} industry-involved)")
    print(f"Edges:   {len(edges_df)}")
    print(f"Output:  {args.output_dir}")


if __name__ == "__main__":
    main()
