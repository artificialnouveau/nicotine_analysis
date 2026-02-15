#!/usr/bin/env python3
"""
enhanced_classifier.py

Enhanced outcome classifier that uses full-length abstracts from the original
JSONL source (not the truncated 500-char version in papers.csv) and expanded
keyword patterns covering health outcomes, policy, epidemiology, marketing,
cessation programs, regulatory findings, and more.

Classifies abstracts into: Positive, Negative, Neutral, Mixed, or Not applicable.

"Positive" = findings favorable to nicotine/tobacco products, cessation
             interventions, harm reduction, or tobacco control progress
"Negative" = findings unfavorable: health harms, industry deception,
             addiction, increased risk, youth targeting, etc.
"Neutral"  = no significant finding, inconclusive, descriptive
"Mixed"    = both positive and negative findings reported
"Not applicable" = no abstract, or abstract is purely methodological/chemical
                   with no directional health/policy finding

Outputs:
  - enhanced_outcome_classifications.csv
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Sentence splitting and conclusion extraction
# ---------------------------------------------------------------------------

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

CONCLUSION_CUES = [
    "conclusion", "we conclude", "in conclusion",
    "our findings", "these findings", "results suggest",
    "results indicate", "we found", "overall", "therefore",
    "in summary", "to summarize", "taken together",
    "these data suggest", "these results suggest",
    "our results", "our data", "our study",
    "this study", "the present study",
    "we recommend", "it is recommended",
    "implications", "suggest that", "indicate that",
    "demonstrate that", "show that", "reveal that",
    "collectively", "importantly", "notably",
    "evidence suggests", "data suggest", "data indicate",
    "should be", "must be", "need to be", "needs to be",
    "is needed", "are needed", "is required",
    "is warranted", "is necessary", "is essential",
    "policy", "regulation", "legislation",
    "public health", "intervention",
]


def clean(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def extract_conclusion_sentences(abstract: str, max_sents: int = 10) -> List[str]:
    """Extract conclusion-like sentences from abstract."""
    a = clean(abstract)
    if not a:
        return []
    sents = [s.strip() for s in SENT_SPLIT.split(a) if s.strip()]
    cue_hits = [s for s in sents if any(c in s.lower() for c in CONCLUSION_CUES)]
    if cue_hits:
        return cue_hits[:max_sents]
    # Fall back to last sentences
    return sents[-min(4, len(sents)):]


# ---------------------------------------------------------------------------
# Expanded keyword patterns
# ---------------------------------------------------------------------------

POS_PATTERNS = [
    # Cessation / quitting success
    r"\bcessation\b",
    r"\bquit\s*(smoking|tobacco|cigarette)?\b",
    r"\bquitting\b",
    r"\babstinence\b",
    r"\bstop(ped|ping)?\s+smoking\b",
    # Harm reduction / reduced risk
    r"\bharm\s+reduction\b",
    r"\breduced\s+(exposure|harm|risk|toxic|toxicant|level|emission|concentration)\b",
    r"\blower\s+(exposure|levels?|biomarkers?|concentrations?|emissions?|toxicants?)\b",
    r"\bless\s+(harmful|toxic|hazardous|dangerous)\b",
    r"\bdecreased\s+(odds|risk|prevalence|incidence|mortality|morbidity)\b",
    r"\blower\s+(odds|risk|prevalence|incidence|mortality)\b",
    r"\bmodified\s+risk\b",
    r"\brisk\s+reduction\b",
    r"\bless\s+exposure\b",
    # Improvement / benefit
    r"\bimproved?\b",
    r"\bbenefit(s|ial|ted)?\b",
    r"\bsafer\b",
    r"\bprotective\s+(effect|factor|role)\b",
    r"\bnon[-\s]?toxic\b",
    r"\bwell[-\s]tolerated\b",
    r"\bminimal\s+(risk|harm|toxicity)\b",
    r"\bno\s+(adverse|harmful|toxic)\s+(effect|event|outcome)\b",
    # Program / intervention effectiveness
    r"\beffective\s*(ness)?\b",
    r"\befficac(y|ious)\b",
    r"\bsuccessful(ly)?\b",
    r"\bpromising\b",
    r"\bencouraging\b",
    r"\bfavor(able|ably)\b",
    r"\bfavour(able|ably)\b",
    r"\bhelp(s|ed|ful)\b",
    # Prevalence decline
    r"\bsmoking\s+prevalence\s+(declined|decreased|dropped|fell|reduced)\b",
    r"\breduction\s+in\s+(smoking|tobacco|cigarette|nicotine)\b",
    r"\bfewer\s+(cigarettes?|smokers?)\b",
    r"\bdecline\s+in\s+(smoking|tobacco|cigarette|use|prevalence|consumption)\b",
    r"\bquit\s+rates?\s+(increased|improved|higher)\b",
    # Tobacco control success
    r"\btobacco\s+control\b.*\b(effective|successful|progress|gains?)\b",
    r"\bsmoking\s+ban(s)?\b.*\b(effective|successful|reduced|decrease)\b",
    r"\banti[-\s]?smoking\b.*\b(effective|successful|reduced)\b",
    r"\bprevention\s+(program|intervention|effort|campaign|strategy)\b",
    r"\bsignificantly\s+reduced\b",
    r"\bsubstantially\s+reduced\b",
    r"\balternative\s+to\s+(smoking|cigarettes?|combustible)\b",
    r"\bswitching\b.*\breduced\b",
]

NEG_PATTERNS = [
    # Health harms
    r"\bharmful\b",
    r"\btoxic(ity|ant|ants)?\b",
    r"\badverse\s+(effect|event|outcome|health|impact|consequence)\b",
    r"\bincreased\s+(risk|harm|exposure|odds|prevalence|incidence|mortality|morbidity)\b",
    r"\bhigher\s+(odds|risk|prevalence|incidence|mortality|rates?)\b",
    r"\bcarcinogen(ic|s|icity)?\b",
    r"\bcancer\b",
    r"\baddiction\b",
    r"\baddictive\b",
    r"\bdependence\b",
    r"\bneurotoxic\b",
    r"\bgenotoxic\b",
    r"\bmutagenic(ity)?\b",
    r"\btumor(s|igenic)?\b",
    r"\btumour(s)?\b",
    r"\bmalignant\b",
    r"\bneoplasm\b",
    r"\bleukoplakia\b",
    r"\bgingival\s+recession\b",
    r"\boral\s+(cancer|lesion|disease)\b",
    r"\blung\s+(cancer|disease|damage)\b",
    r"\bcardiovascular\s+(disease|risk|event|mortality)\b",
    r"\bheart\s+(disease|attack|failure)\b",
    r"\bstroke\b",
    r"\bmortality\b",
    r"\bpremature\s+death\b",
    r"\bdeath(s)?\b",
    r"\bhealth\s+(risk|hazard|danger|threat|consequence|effect|concern)\b",
    r"\bdangerous\b",
    r"\bhazardous\b",
    r"\bdetrimental\b",
    r"\bdeleterious\b",
    r"\bnegative\s+(effect|impact|outcome|consequence|health)\b",
    r"\bworsened?\b",
    r"\bexacerbate[ds]?\b",
    # Industry criticism
    r"\bdeceptive\b",
    r"\bmisleading\b",
    r"\bmanipulat(e|ed|ing|ion)\b",
    r"\btargeting\s+(youth|children|adolescent|teen|minor|young)\b",
    r"\bexploit(ation|ing|ed)?\b",
    r"\bfraud(ulent)?\b",
    r"\bconceal(ed|ing|ment)?\b",
    r"\bsuppress(ed|ing|ion)?\b",
    r"\bdenial\b",
    r"\bindustry\s+(tactics?|strategies?|propaganda|interference|lobbying)\b",
    # Addiction / dependence
    r"\bnicotine\s+dependence\b",
    r"\bnicotine\s+addiction\b",
    r"\bwithdrawal\b",
    r"\bcraving\b",
    r"\brelapse\b",
    r"\breinforcing\b",
    # Exposure
    r"\bsecondhand\s+smoke\b",
    r"\benvironmental\s+tobacco\s+smoke\b",
    r"\bpassive\s+smoking\b",
    r"\bthirdhand\b",
    r"\bcontaminat(ed|ion)\b",
    # Youth / initiation
    r"\byouth\s+(smoking|tobacco|nicotine|use|initiation|uptake|addiction)\b",
    r"\badolescent\s+(smoking|tobacco|nicotine|use|initiation)\b",
    r"\bgateway\b",
    r"\binitiation\b",
    r"\brising\s+(prevalence|use|consumption)\b",
    r"\bincreas(e|ed|ing)\s+in\s+(prevalence|use|consumption|smoking|uptake)\b",
    r"\bepidemic\b",
    r"\balarming\b",
    r"\bburden\b",
]

NEU_PATTERNS = [
    r"\bno\s+significant\b",
    r"\bnot\s+significant(ly)?\b",
    r"\binconclusive\b",
    r"\bmixed\s+(results?|findings?|evidence|outcomes?)\b",
    r"\bunclear\b",
    r"\bno\s+difference\b",
    r"\bno\s+association\b",
    r"\bfurther\s+research\b",
    r"\bmore\s+(research|studies|investigation|data)\s+(is\s+)?needed\b",
    r"\bremains?\s+(unclear|unknown|uncertain)\b",
    r"\blimited\s+evidence\b",
    r"\binsufficient\s+evidence\b",
    r"\bequivocal\b",
    r"\bnon[-\s]?significant\b",
    r"\bno\s+(clear|definitive|conclusive)\b",
]


def find_matches(text: str, patterns: list) -> List[str]:
    """Return list of matched pattern strings in text."""
    matches = []
    t = text.lower()
    for p in patterns:
        m = re.search(p, t, re.IGNORECASE)
        if m:
            matches.append(m.group())
    return matches


def classify_abstract(abstract: str) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Classify abstract outcome direction.
    Focuses on conclusion sentences to avoid background text pollution.

    Returns (outcome, pos_matches, neg_matches, neu_matches)
    """
    a = clean(abstract)
    if not a or len(a) < 50:
        return ("Not applicable", [], [], [])

    # Extract conclusion sentences — this is where the findings are
    conclusion_sents = extract_conclusion_sentences(a)
    conclusion_text = " ".join(conclusion_sents)

    # Classify based on conclusion sentences ONLY (avoids background pollution)
    pos_matches = find_matches(conclusion_text, POS_PATTERNS)
    neg_matches = find_matches(conclusion_text, NEG_PATTERNS)
    neu_matches = find_matches(conclusion_text, NEU_PATTERNS)

    has_pos = len(pos_matches) > 0
    has_neg = len(neg_matches) > 0
    has_neu = len(neu_matches) > 0

    # Priority: Mixed > Neutral > Negative > Positive
    if has_pos and has_neg:
        return ("Mixed", pos_matches, neg_matches, neu_matches)
    if has_neu and not has_pos and not has_neg:
        return ("Neutral", pos_matches, neg_matches, neu_matches)
    if has_neu:
        return ("Neutral", pos_matches, neg_matches, neu_matches)
    if has_neg:
        return ("Negative", pos_matches, neg_matches, neu_matches)
    if has_pos:
        return ("Positive", pos_matches, neg_matches, neu_matches)

    return ("Not applicable", pos_matches, neg_matches, neu_matches)


def load_full_abstracts(jsonl_path: str) -> Dict[str, str]:
    """Load full (non-truncated) abstracts from the original JSONL source."""
    abstracts = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pid = str(rec.get("record_id") or rec.get("pmid") or "")
                ab = rec.get("abstract") or ""
                if pid:
                    abstracts[pid] = ab
            except json.JSONDecodeError:
                continue
    return abstracts


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Enhanced outcome classification using full abstracts")
    ap.add_argument("--papers_csv", required=True, help="Path to papers.csv")
    ap.add_argument("--jsonl_path", required=True, help="Path to original JSONL with full abstracts")
    ap.add_argument("--output_csv", required=True, help="Path to output CSV")
    args = ap.parse_args()

    papers = pd.read_csv(args.papers_csv)
    print(f"Loaded {len(papers)} papers from CSV")

    full_abstracts = load_full_abstracts(args.jsonl_path)
    print(f"Loaded {len(full_abstracts)} full abstracts from JSONL")

    results = []
    for _, row in papers.iterrows():
        pid = str(row["paper_id"])
        # Use full abstract from JSONL, fall back to truncated CSV version
        abstract = full_abstracts.get(pid, "")
        if not abstract:
            abstract = str(row.get("abstract", "")) if pd.notna(row.get("abstract")) else ""

        outcome_enhanced, pos_m, neg_m, neu_m = classify_abstract(abstract)

        results.append({
            "paper_id": pid,
            "title": row.get("title", ""),
            "year": row.get("year", ""),
            "industry_category": row.get("industry_category", ""),
            "outcome_keyword": row.get("outcome", ""),
            "outcome_enhanced": outcome_enhanced,
            "pos_matches": "; ".join(pos_m[:5]),
            "neg_matches": "; ".join(neg_m[:5]),
            "neu_matches": "; ".join(neu_m[:5]),
        })

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    # Summary
    print(f"\nEnhanced classification results:")
    dist = out_df["outcome_enhanced"].value_counts()
    for outcome in ["Positive", "Negative", "Neutral", "Mixed", "Not applicable"]:
        count = dist.get(outcome, 0)
        pct = count / len(out_df) * 100
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    keyword_coded = (out_df["outcome_keyword"] != "Not coded").sum()
    enhanced_coded = (out_df["outcome_enhanced"] != "Not applicable").sum()
    print(f"\nKeyword-coded: {keyword_coded} ({keyword_coded/len(out_df)*100:.1f}%)")
    print(f"Enhanced-coded: {enhanced_coded} ({enhanced_coded/len(out_df)*100:.1f}%)")
    print(f"Coverage improvement: {keyword_coded} -> {enhanced_coded} (+{enhanced_coded - keyword_coded})")

    print(f"\nBy industry category:")
    for cat in ["Tobacco Company", "COI Declared", "Independent"]:
        subset = out_df[out_df["industry_category"] == cat]
        coded = subset[subset["outcome_enhanced"] != "Not applicable"]
        n = len(coded)
        total = len(subset)
        print(f"\n  {cat}: {n}/{total} coded ({n/total*100:.1f}%)")
        if n > 0:
            for outcome in ["Positive", "Negative", "Neutral", "Mixed"]:
                c = (coded["outcome_enhanced"] == outcome).sum()
                pct = c / n * 100
                print(f"    {outcome}: {c} ({pct:.1f}%)")

    print(f"\nOutput: {args.output_csv}")


if __name__ == "__main__":
    main()
