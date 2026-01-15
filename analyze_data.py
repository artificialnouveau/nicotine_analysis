#!/usr/bin/env python3
"""
analyze_data.py

Purpose
-------
Analyze PubMed + Web of Science (WoS) records for:
  1) What the papers are about (product/topic tags)
  2) Outcome direction (positive/negative/neutral/mixed) based on *conclusion-like* sentences
  3) Conflicts / industry involvement via:
       A) declared COI language (in metadata or extracted text)
       B) industry affiliation in author affiliations (tobacco/vape companies)
       C) funding source classification (industry vs government vs nonprofit vs university vs mixed/unknown)
  4) Association tests: industry involvement vs outcome

Inputs
------
Expects JSONL produced by fetch_data.py in:
  <out_dir>/data/pubmed_records.jsonl
  <out_dir>/data/wos_records.jsonl (optional)
Optionally uses:
  pdf_path (if present in records) to extract text snippets (funding/COI) for better recall.

Outputs
-------
Writes to: <out_dir>/analysis_v2/
  - canonical_records.csv                     (deduplicated master table)
  - audit_evidence.csv                        (sentences/patterns used for each classification)
  - contingency_outcome_by_industry.csv       (cross-tab)
  - stats_summary.json                        (chi-square/Fisher, odds ratios, counts)
  - figures/
      - outcome_by_industry.png
      - outcome_by_declared_coi.png
      - funding_class_by_industry.png

Usage Examples
--------------
1) PubMed-only analysis (no WoS):
  python analyze_data.py --out_dir ./study_dump

2) Include WoS records:
  python analyze_data.py --out_dir ./study_dump --include_wos

3) If PDFs are present and you want to use them for funding/COI extraction:
  python analyze_data.py --out_dir ./study_dump --include_wos --use_pdfs

Notes on PDFs
-------------
- PDFs are NOT required; the script works on metadata (title/abstract/affiliations/COI statements) alone.
- PDFs improve extraction of funding/disclosures, but only use PDFs you obtained in a compliant manner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# Stats
from scipy.stats import chi2_contingency, fisher_exact

# Optional PDF text extraction backend(s)
PDF_BACKEND = None
try:
    import fitz  # PyMuPDF
    PDF_BACKEND = "pymupdf"
except Exception:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        PDF_BACKEND = "pdfminer"
    except Exception:
        PDF_BACKEND = None


# -----------------------------
# Utilities: IO and cleaning
# -----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    d = clean_text(doi).lower()
    if not d:
        return None
    d = d.replace("https://doi.org/", "").replace("http://doi.org/", "")
    d = d.replace("doi:", "").strip()
    return d or None


def normalize_title(title: Optional[str]) -> str:
    t = clean_text(title).lower()
    # strip punctuation, collapse spaces
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return int(str(x)[:4])
    except Exception:
        return None


# -----------------------------
# Optional: PDF text extraction
# -----------------------------

def extract_pdf_text(pdf_path: str, max_chars: int = 60000) -> str:
    """
    Best-effort extraction. This is used to help find Funding/COI sections
    when metadata is missing.
    """
    if not pdf_path or not os.path.exists(pdf_path) or not PDF_BACKEND:
        return ""

    try:
        if PDF_BACKEND == "pymupdf":
            doc = fitz.open(pdf_path)
            parts = []
            total = 0
            for page in doc:
                t = page.get_text("text") or ""
                if t:
                    parts.append(t)
                    total += len(t)
                    if total >= max_chars:
                        break
            doc.close()
            return clean_text(" ".join(parts))[:max_chars]

        if PDF_BACKEND == "pdfminer":
            t = extract_text(pdf_path) or ""
            return clean_text(t)[:max_chars]
    except Exception:
        return ""

    return ""


def extract_section_snippet(text: str, section_names: List[str], window: int = 1600) -> str:
    """
    Extract a snippet near common headings like 'Funding', 'Acknowledgments',
    'Conflicts of interest', etc.
    """
    t = clean_text(text)
    if not t:
        return ""
    low = t.lower()

    for name in section_names:
        # heading-like match: "Funding", "FUNDING:", "Funding and support"
        m = re.search(rf"\b{name}\b", low, flags=re.IGNORECASE)
        if m:
            start = max(0, m.start())
            end = min(len(t), start + window)
            return t[start:end]
    return ""


# -----------------------------------------
# De-duplication: canonical record building
# -----------------------------------------

def canonical_key(row: Dict[str, Any]) -> str:
    """
    Canonical key for deduplicating PubMed/WoS records.
    Priority:
      1) DOI (normalized)
      2) title_norm + year + first_author_last (fallback)
    """
    doi = normalize_doi(row.get("doi"))
    if doi:
        return f"doi::{doi}"

    title = normalize_title(row.get("title"))
    year = safe_int(row.get("year"))
    fa_last = ""
    # Try to infer first author last name if available
    authors = row.get("authors")
    if isinstance(authors, list) and authors:
        first = authors[0]
        if isinstance(first, dict):
            fa_last = clean_text(first.get("last", "")).lower()
        elif isinstance(first, str):
            fa_last = clean_text(first.split()[-1]).lower()
    return f"ty::{title}::y::{year or 'na'}::fa::{fa_last or 'na'}"


def merge_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Deduplicate into canonical records. When duplicates exist, prefer PubMed
    for COI/PMCID fields, and prefer non-empty abstract/title across sources.
    """
    # Add canonical_id
    for r in records:
        r["doi"] = normalize_doi(r.get("doi"))
        r["canonical_id"] = canonical_key(r)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Define source priority: prefer PUBMED over WOS for some fields
    src_priority = {"PUBMED": 2, "WOS": 1}
    df["source_priority"] = df["source_db"].map(src_priority).fillna(0).astype(int)

    # Sort so "best" record appears first per canonical_id
    # Criteria: source_priority desc, has_abstract desc, has_title desc, has_coi_statement desc
    df["has_abstract"] = df["abstract"].apply(lambda x: 1 if clean_text(x) else 0)
    df["has_title"] = df["title"].apply(lambda x: 1 if clean_text(x) else 0)
    df["has_coi"] = df.get("coi_statement", pd.Series([None]*len(df))).apply(lambda x: 1 if clean_text(x) else 0)

    df = df.sort_values(
        by=["canonical_id", "source_priority", "has_abstract", "has_title", "has_coi"],
        ascending=[True, False, False, False, False],
    )

    # Aggregate by canonical_id, taking the first non-null in each important field
    def first_nonempty(series: pd.Series) -> Any:
        for v in series:
            if isinstance(v, list) and v:
                return v
            if clean_text(v):
                return v
        # if all empty:
        return None

    agg = {
        "canonical_id": "first",
        "doi": "first",
        "title": first_nonempty,
        "abstract": first_nonempty,
        "year": "first",
        "journal": first_nonempty,
        "source_db": lambda s: list(pd.unique(s.dropna())),
        "record_id": lambda s: list(pd.unique(s.dropna())),
        "pmid": first_nonempty,
        "pmcid": first_nonempty,
        "coi_statement": first_nonempty,
        "pdf_path": first_nonempty,
        "pmc_pdf_url": first_nonempty,
        "wos_uid": first_nonempty,
        "wos_times_cited": "max",
        "authors": first_nonempty,  # note: fetch_data.py WoS may not have authors unless you add them
    }

    g = df.groupby("canonical_id", as_index=False).agg(agg)

    # Clean year
    g["year"] = g["year"].apply(safe_int)

    return g


# -----------------------------------------
# Industry affiliation detection + resolution
# -----------------------------------------

# Canonical tobacco/vape industry orgs and patterns.
# Expand this list as needed; the resolution maps patterns -> canonical org.
INDUSTRY_ORG_PATTERNS = [
    (r"\bphilip\s+morris\b|\bpmi\b|\bphilip\s+morris\s+international\b", "Philip Morris International"),
    (r"\baltria\b", "Altria"),
    (r"\bbritish\s+american\s+tobacco\b|\bbat\b", "British American Tobacco"),
    (r"\bjapan\s+tobacco\b|\bjt\b", "Japan Tobacco"),
    (r"\bimperial\s+brands\b|\bimperial\s+tobacco\b", "Imperial Brands"),
    (r"\brj\s*reynolds\b|\breynolds\s+american\b|\brjr\b", "R.J. Reynolds"),
    (r"\bswedish\s+match\b", "Swedish Match"),
    (r"\bjuul\b|\bjuul\s+labs\b", "JUUL Labs"),
    (r"\bvuse\b", "Vuse / BAT"),
    (r"\bnjoy\b", "NJOY"),
    (r"\biqos\b", "IQOS / PMI"),
    # Generic "tobacco company" / "cigarette manufacturer" signals (kept separate)
    (r"\btobacco\s+company\b|\bcigarette\s+manufacturer\b|\bvaping\s+industry\b", "Generic tobacco/vape industry"),
]

# If you want to treat certain foundations as "industry-adjacent", keep separate:
INDUSTRY_ADJACENT_PATTERNS = [
    (r"\bfoundation\s+for\s+a\s+smoke[-\s]?free\s+world\b", "Foundation for a Smoke-Free World"),
]


def extract_affiliations(authors_field: Any) -> List[str]:
    """
    PubMed records from fetch_data.py (original version) include author affiliations.
    WoS records may not unless you extend your WoS fetch.
    """
    affs: List[str] = []
    if not authors_field:
        return affs
    if isinstance(authors_field, list):
        for a in authors_field:
            if isinstance(a, dict):
                for aff in a.get("affiliations") or []:
                    if clean_text(aff):
                        affs.append(clean_text(aff))
    return affs


def resolve_industry_orgs(texts: List[str]) -> Tuple[bool, List[str], List[str]]:
    """
    Returns:
      - industry_affiliation_yes (bool)
      - matched_canonical_orgs (list)
      - matched_raw_snippets (list of snippets evidencing the match)
    """
    matched_orgs = set()
    evidence = []
    joined = " | ".join([clean_text(t) for t in texts if clean_text(t)])
    low = joined.lower()

    for pat, canon in INDUSTRY_ORG_PATTERNS:
        if re.search(pat, low, flags=re.IGNORECASE):
            matched_orgs.add(canon)
            evidence.append(f"match:{canon}")

    # adjacent patterns can be treated separately; still evidence
    for pat, canon in INDUSTRY_ADJACENT_PATTERNS:
        if re.search(pat, low, flags=re.IGNORECASE):
            matched_orgs.add(canon)
            evidence.append(f"match:{canon}")

    return (len(matched_orgs) > 0, sorted(matched_orgs), evidence)


# -----------------------------------------
# Funding source extraction (separate from COI)
# -----------------------------------------

FUNDING_SECTION_HEADINGS = ["funding", "funding information", "support", "acknowledg", "financial support", "grant"]

GOV_FUNDERS = [
    r"\bnih\b", r"\bnational institutes of health\b",
    r"\bcdc\b", r"\bcenters for disease control\b",
    r"\bfda\b", r"\bfood and drug administration\b",
    r"\bnsf\b", r"\bnational science foundation\b",
    r"\bwellcome\b", r"\bmedical research council\b|\bmrc\b",
    r"\beuropean commission\b|\bhorizon\b",
]

NONPROFIT_FUNDERS = [
    r"\bfoundation\b", r"\bcharit", r"\bnonprofit\b",
    r"\bamerican cancer society\b", r"\btruth initiative\b",
]

UNIVERSITY_FUNDERS = [
    r"\buniversity\b", r"\bcollege\b", r"\bdepartment\b", r"\binstitute\b",
]

INDUSTRY_FUNDING_SIGNALS = [
    r"\bfunded by\b", r"\bsupported by\b", r"\bgrant from\b", r"\bsponsor",
    # plus any industry org patterns
]


def classify_funding_source(funding_text: str) -> Tuple[str, List[str]]:
    """
    Classify funding text into one of:
      - Industry
      - Government
      - Nonprofit
      - University/Institutional
      - Mixed
      - None/Not stated

    Returns (funding_class, evidence_tags)
    """
    t = clean_text(funding_text)
    if not t:
        return ("None/Not stated", [])

    low = t.lower()
    evidence = []

    def any_match(patterns: List[str]) -> bool:
        return any(re.search(p, low, flags=re.IGNORECASE) for p in patterns)

    has_gov = any_match(GOV_FUNDERS)
    has_np = any_match(NONPROFIT_FUNDERS)
    has_uni = any_match(UNIVERSITY_FUNDERS)

    # Industry: either explicit funding phrasing + industry org mention, OR industry org mention alone in funding context
    industry_org_hit, orgs, _ = resolve_industry_orgs([t])
    has_industry = industry_org_hit or any_match(INDUSTRY_FUNDING_SIGNALS)

    if industry_org_hit:
        evidence.append("industry_org:" + ",".join(orgs))
    if has_gov:
        evidence.append("gov")
    if has_np:
        evidence.append("nonprofit")
    if has_uni:
        evidence.append("university")
    if has_industry:
        evidence.append("industry")

    classes = [c for c, flag in [
        ("Industry", has_industry),
        ("Government", has_gov),
        ("Nonprofit", has_np),
        ("University/Institutional", has_uni),
    ] if flag]

    if not classes:
        return ("None/Not stated", evidence)
    if len(classes) == 1:
        return (classes[0], evidence)
    return ("Mixed", evidence)


# -----------------------------------------
# Declared COI extraction/classification
# -----------------------------------------

NO_COI_PHRASES = [
    "no conflict of interest", "no conflicts of interest",
    "no competing interests", "no competing interest",
    "declare no conflict", "declares no conflict",
    "nothing to disclose", "no disclosures",
    "the authors declare no", "authors declare no",
]

YES_COI_PHRASES = [
    "conflict of interest", "competing interests", "disclosure",
    "received funding", "funded by", "supported by", "grant from",
    "consultant", "honoraria", "speaker", "advisory board",
    "employee", "employment", "stock", "equity", "patent", "royalties",
    "expert witness", "litigation",
]


def classify_declared_coi(coi_text: Optional[str]) -> Tuple[str, List[str]]:
    """
    Returns:
      - 'Yes' / 'No' / 'Missing'
      - evidence tags
    """
    t = clean_text(coi_text)
    if not t:
        return ("Missing", [])
    low = t.lower()

    for p in NO_COI_PHRASES:
        if p in low:
            return ("No", [f"phrase:{p}"])
    for p in YES_COI_PHRASES:
        if p in low:
            return ("Yes", [f"phrase:{p}"])
    # ambiguous: treat as Yes (conservative) OR Unknown; here we use 'Yes*' style.
    return ("Yes", ["default_yes"])


# -----------------------------------------
# Topic tagging (ecig/tobacco/nicotine)
# -----------------------------------------

ECIG_KWS = [
    "e-cig", "ecig", "electronic cigarette", "electronic nicotine delivery", "ends",
    "vape", "vaping", "e-liquid", "eliquid", "juul",
]
TOBACCO_KWS = [
    "tobacco", "cigarette", "smoking", "smoker", "combustible", "cigar", "cigarillo",
    "heated tobacco", "htp", "snus",
]
NICOTINE_KWS = [
    "nicotine", "nicotinic",
    "nicotine replacement", "nrt", "patch", "gum", "lozenge",
]


def tag_topics(title: str, abstract: str) -> List[str]:
    text = (clean_text(title) + " " + clean_text(abstract)).lower()
    tags = []
    if any(k in text for k in ECIG_KWS):
        tags.append("E-cigarettes/ENDS")
    if any(k in text for k in TOBACCO_KWS):
        tags.append("Tobacco/Combustibles")
    if any(k in text for k in NICOTINE_KWS):
        tags.append("Nicotine/NRT")
    if not tags:
        tags.append("Other/Unclear")
    # Make multi-label unique, stable order:
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# -----------------------------------------
# Conclusion-sentence outcome coding
# -----------------------------------------

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Outcome direction patterns
POS_PATTERNS = [
    r"\bcessation\b", r"\bquit\b", r"\bquitting\b", r"\babstinence\b",
    r"\bharm\s+reduction\b", r"\breduced\s+(exposure|harm|risk|toxic|toxicant)\b",
    r"\blower\s+(exposure|levels|biomarkers)\b",
    r"\bimproved\b", r"\bbenefit\b", r"\bsafer\b", r"\bless\s+harmful\b",
]
NEG_PATTERNS = [
    r"\bharmful\b", r"\btoxic\b", r"\badverse\b", r"\bincreased\s+(risk|harm|exposure)\b",
    r"\binflammation\b", r"\boxidative\s+stress\b", r"\binjury\b",
    r"\bcardiovascular\b", r"\brespiratory\b", r"\blung\b.*\binjury\b",
    r"\bcancer\b|\bcarcinogen\b",
    r"\baddiction\b|\bdependence\b",
]
NEUTRAL_PATTERNS = [
    r"\bno\s+significant\b", r"\bnot\s+significant\b", r"\binconclusive\b",
    r"\bmixed\b", r"\bunclear\b", r"\blimited\s+evidence\b", r"\bfurther\s+research\b",
    r"\bno\s+difference\b",
]

# “Conclusion-like” cues to prioritize sentences
CONCLUSION_CUES = [
    "conclusion", "conclusions", "we conclude", "in conclusion",
    "our findings", "these findings", "this study suggests", "this study indicates",
    "results suggest", "results indicate", "we found", "we demonstrate",
    "implications", "overall", "therefore",
]


def pick_conclusion_sentences(title: str, abstract: str, max_sentences: int = 6) -> List[str]:
    """
    Extract a small set of sentences most likely to represent the paper's conclusions.
    Heuristic:
      - Prefer sentences containing conclusion cues
      - Else take last 2-3 sentences of abstract (common structure)
    """
    a = clean_text(abstract)
    if not a:
        return []

    sents = [s.strip() for s in SENT_SPLIT.split(a) if s.strip()]
    if not sents:
        return []

    # Cue-based selection
    cue_hits = []
    for s in sents:
        low = s.lower()
        if any(c in low for c in CONCLUSION_CUES):
            cue_hits.append(s)

    if cue_hits:
        return cue_hits[:max_sentences]

    # Otherwise fallback to tail of abstract
    tail = sents[-min(3, len(sents)):]
    return tail[:max_sentences]


def outcome_from_sentences(sentences: List[str]) -> Tuple[str, List[str], List[str]]:
    """
    Determine outcome direction:
      - Positive
      - Negative
      - Neutral
      - Mixed (both pos and neg)
      - Not coded (no clear signals)

    Returns: (outcome_label, evidence_sentences, evidence_tags)
    """
    if not sentences:
        return ("Not coded", [], [])

    pos_hits = []
    neg_hits = []
    neu_hits = []
    tags = []

    for s in sentences:
        low = s.lower()
        if any(re.search(p, low) for p in POS_PATTERNS):
            pos_hits.append(s)
        if any(re.search(p, low) for p in NEG_PATTERNS):
            neg_hits.append(s)
        if any(re.search(p, low) for p in NEUTRAL_PATTERNS):
            neu_hits.append(s)

    if pos_hits:
        tags.append("pos_patterns")
    if neg_hits:
        tags.append("neg_patterns")
    if neu_hits:
        tags.append("neutral_patterns")

    # Decide label
    if pos_hits and neg_hits:
        return ("Mixed", (pos_hits + neg_hits)[:5], tags)
    if neg_hits:
        return ("Negative", neg_hits[:5], tags)
    if pos_hits:
        return ("Positive", pos_hits[:5], tags)
    if neu_hits:
        return ("Neutral", neu_hits[:5], tags)

    return ("Not coded", [], tags)


# -----------------------------------------
# Association tests + effect sizes
# -----------------------------------------

def odds_ratio_2x2(a: int, b: int, c: int, d: int, add_half_if_zero: bool = True) -> Tuple[float, float]:
    """
    Odds ratio for table:
        outcome=1   outcome=0
    exp=1    a         b
    exp=0    c         d

    Returns (OR, logOR_se). Adds 0.5 continuity correction if any cell is 0.
    """
    if add_half_if_zero and any(x == 0 for x in [a, b, c, d]):
        a += 1/2
        b += 1/2
        c += 1/2
        d += 1/2

    or_val = (a * d) / (b * c)
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    return float(or_val), float(se)


def ci95_from_log(or_val: float, se: float) -> Tuple[float, float]:
    lo = math.exp(math.log(or_val) - 1.96 * se)
    hi = math.exp(math.log(or_val) + 1.96 * se)
    return float(lo), float(hi)


# -----------------------------------------
# Main
# -----------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory used by fetch_data.py (contains data/).")
    ap.add_argument("--include_wos", action="store_true", help="Include Web of Science JSONL if present.")
    ap.add_argument("--use_pdfs", action="store_true", help="If pdf_path exists, extract snippets for funding/COI.")
    ap.add_argument("--max_pdf_chars", type=int, default=60000, help="Max chars to read from PDFs when --use_pdfs is set.")
    args = ap.parse_args()

    data_dir = os.path.join(args.out_dir, "data")
    pubmed_path = os.path.join(data_dir, "pubmed_records.jsonl")
    wos_path = os.path.join(data_dir, "wos_records.jsonl")

    pubmed_rows = read_jsonl(pubmed_path)
    wos_rows = read_jsonl(wos_path) if args.include_wos else []

    records = pubmed_rows + wos_rows
    if not records:
        raise SystemExit("No records found. Run fetch_data.py first.")

    # Deduplicate into canonical records
    df = merge_records(records)
    if df.empty:
        raise SystemExit("No records after de-duplication (unexpected).")

    # Ensure expected columns exist
    for col in ["title", "abstract", "coi_statement", "authors", "pdf_path", "year", "doi"]:
        if col not in df.columns:
            df[col] = None

    # Optional: extract PDF snippets for funding/COI (best effort)
    df["pdf_text"] = ""
    df["funding_snippet"] = ""
    df["coi_snippet"] = ""
    if args.use_pdfs:
        for i, row in df.iterrows():
            pdf_path = row.get("pdf_path")
            if isinstance(pdf_path, str) and pdf_path and os.path.exists(pdf_path) and PDF_BACKEND:
                txt = extract_pdf_text(pdf_path, max_chars=args.max_pdf_chars)
                df.at[i, "pdf_text"] = txt
                df.at[i, "funding_snippet"] = extract_section_snippet(txt, FUNDING_SECTION_HEADINGS, window=1600)
                df.at[i, "coi_snippet"] = extract_section_snippet(txt, ["conflict", "competing", "disclosure"], window=1600)

    # -----------------------------
    # 1) Topic tags
    # -----------------------------
    df["Topic_Tags"] = df.apply(lambda r: tag_topics(r.get("title") or "", r.get("abstract") or ""), axis=1)

    # -----------------------------
    # 2) Declared COI (from metadata, optionally augmented by PDF snippet)
    # -----------------------------
    def coi_text_used(r) -> str:
        base = clean_text(r.get("coi_statement"))
        if base:
            return base
        # fallback to PDF snippet if enabled
        snip = clean_text(r.get("coi_snippet"))
        return snip

    df["COI_Text_Used"] = df.apply(coi_text_used, axis=1)
    coi_class = df["COI_Text_Used"].apply(classify_declared_coi)
    df["Declared_COI"] = coi_class.apply(lambda x: x[0])
    df["Declared_COI_Evidence"] = coi_class.apply(lambda x: x[1])

    # -----------------------------
    # 3) Industry affiliation from author affiliations
    # -----------------------------
    df["Affiliations"] = df["authors"].apply(extract_affiliations)
    ind = df["Affiliations"].apply(resolve_industry_orgs)
    df["Industry_Affiliation"] = ind.apply(lambda x: "Yes" if x[0] else "No")
    df["Industry_Orgs"] = ind.apply(lambda x: x[1])
    df["Industry_Affil_Evidence"] = ind.apply(lambda x: x[2])

    # "Industry involvement" definition: either declared COI yes OR industry affiliation yes
    # (You can change this if you want them separate in tests.)
    df["Industry_Involved"] = df.apply(
        lambda r: "Yes" if (r["Industry_Affiliation"] == "Yes" or r["Declared_COI"] == "Yes") else "No",
        axis=1
    )

    # -----------------------------
    # 4) Funding extraction & classification (separate from COI)
    # -----------------------------
    def funding_text_used(r) -> str:
        # We only have reliable funding fields if you extract them; fallback to PDF snippet if enabled.
        # If neither exists, empty.
        snip = clean_text(r.get("funding_snippet"))
        return snip

    df["Funding_Text_Used"] = df.apply(funding_text_used, axis=1)
    fund_class = df["Funding_Text_Used"].apply(classify_funding_source)
    df["Funding_Class"] = fund_class.apply(lambda x: x[0])
    df["Funding_Evidence"] = fund_class.apply(lambda x: x[1])

    # -----------------------------
    # 5) Outcome coding from conclusion-like sentences
    # -----------------------------
    df["Conclusion_Sentences"] = df.apply(
        lambda r: pick_conclusion_sentences(r.get("title") or "", r.get("abstract") or "", max_sentences=6),
        axis=1
    )
    outc = df["Conclusion_Sentences"].apply(outcome_from_sentences)
    df["Outcome"] = outc.apply(lambda x: x[0])
    df["Outcome_Evidence_Sentences"] = outc.apply(lambda x: x[1])
    df["Outcome_Evidence_Tags"] = outc.apply(lambda x: x[2])

    # -----------------------------
    # Build audit table (evidence)
    # -----------------------------
    audit_cols = [
        "canonical_id", "doi", "year", "title",
        "Declared_COI", "COI_Text_Used", "Declared_COI_Evidence",
        "Industry_Affiliation", "Industry_Orgs", "Industry_Affil_Evidence",
        "Funding_Class", "Funding_Text_Used", "Funding_Evidence",
        "Outcome", "Conclusion_Sentences", "Outcome_Evidence_Sentences", "Outcome_Evidence_Tags",
        "Topic_Tags",
    ]
    audit_df = df[audit_cols].copy()

    # -----------------------------
    # Association tests
    # -----------------------------
    # Primary association: Industry_Involved vs Outcome (Positive/Negative/Neutral/Mixed/Not coded)
    ctab = pd.crosstab(df["Industry_Involved"], df["Outcome"])
    # Ensure consistent column order
    outcome_order = ["Positive", "Negative", "Neutral", "Mixed", "Not coded"]
    for col in outcome_order:
        if col not in ctab.columns:
            ctab[col] = 0
    ctab = ctab[outcome_order]

    # Chi-square test (only if table appropriate)
    chi2_res = {}
    try:
        chi2, p, dof, expected = chi2_contingency(ctab.values)
        chi2_res = {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}
    except Exception:
        chi2_res = {"error": "chi2_contingency_failed"}

    # 2x2 OR: Positive vs Not Positive by Industry_Involved
    # Build 2x2:
    #               Positive   NotPositive
    # Industry Yes      a          b
    # Industry No       c          d
    a = int(ctab.loc["Yes", "Positive"]) if "Yes" in ctab.index else 0
    c = int(ctab.loc["No", "Positive"]) if "No" in ctab.index else 0
    b = int(ctab.loc["Yes", outcome_order].sum() - a) if "Yes" in ctab.index else 0
    d = int(ctab.loc["No", outcome_order].sum() - c) if "No" in ctab.index else 0

    # Fisher exact for 2x2 if small counts; always computable
    fisher_p = None
    try:
        _, fisher_p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        fisher_p = float(fisher_p)
    except Exception:
        fisher_p = None

    or_val, se = odds_ratio_2x2(a, b, c, d, add_half_if_zero=True)
    lo, hi = ci95_from_log(or_val, se)

    stats_summary = {
        "n_records_canonical": int(len(df)),
        "n_industry_involved_yes": int((df["Industry_Involved"] == "Yes").sum()),
        "n_industry_involved_no": int((df["Industry_Involved"] == "No").sum()),
        "contingency_outcome_by_industry": ctab.to_dict(),
        "chi_square_test": chi2_res,
        "positive_vs_notpositive": {
            "table": {"a_ind_yes_pos": a, "b_ind_yes_notpos": b, "c_ind_no_pos": c, "d_ind_no_notpos": d},
            "odds_ratio": float(or_val),
            "ci95": [float(lo), float(hi)],
            "fisher_p_value": fisher_p,
        },
    }

    # -----------------------------
    # Write outputs
    # -----------------------------
    analysis_dir = os.path.join(args.out_dir, "analysis_v2")
    fig_dir = os.path.join(analysis_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    canonical_csv = os.path.join(analysis_dir, "canonical_records.csv")
    audit_csv = os.path.join(analysis_dir, "audit_evidence.csv")
    ctab_csv = os.path.join(analysis_dir, "contingency_outcome_by_industry.csv")
    stats_json = os.path.join(analysis_dir, "stats_summary.json")

    # Save canonical table (with key fields)
    keep_cols = [
        "canonical_id", "source_db", "record_id", "pmid", "pmcid", "wos_uid",
        "doi", "year", "journal", "title", "abstract",
        "Topic_Tags",
        "Declared_COI", "Industry_Affiliation", "Industry_Orgs",
        "Funding_Class",
        "Outcome",
        "Industry_Involved",
        "pdf_path",
    ]
    # Add missing columns if any:
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None

    df[keep_cols].to_csv(canonical_csv, index=False)
    audit_df.to_csv(audit_csv, index=False)
    ctab.to_csv(ctab_csv)

    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)

    # -----------------------------
    # Figures
    # -----------------------------
    # 1) Outcome by industry involvement
    ax = ctab.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Outcome (conclusion-coded) by Industry Involvement")
    ax.set_xlabel("Industry Involved (Yes/No)")
    ax.set_ylabel("Paper count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "outcome_by_industry.png"), dpi=200)
    plt.close()

    # 2) Outcome by declared COI
    ctab2 = pd.crosstab(df["Declared_COI"], df["Outcome"])
    for col in outcome_order:
        if col not in ctab2.columns:
            ctab2[col] = 0
    ctab2 = ctab2[outcome_order]
    ax = ctab2.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Outcome (conclusion-coded) by Declared COI")
    ax.set_xlabel("Declared COI (Yes/No/Missing)")
    ax.set_ylabel("Paper count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "outcome_by_declared_coi.png"), dpi=200)
    plt.close()

    # 3) Funding class by industry affiliation
    ctab3 = pd.crosstab(df["Industry_Affiliation"], df["Funding_Class"])
    ax = ctab3.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Funding Class by Industry Affiliation")
    ax.set_xlabel("Industry Affiliation (Yes/No)")
    ax.set_ylabel("Paper count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "funding_class_by_industry.png"), dpi=200)
    plt.close()

    print("Analysis complete.")
    print(f"- Canonical records: {canonical_csv}")
    print(f"- Audit evidence:   {audit_csv}")
    print(f"- Contingency:      {ctab_csv}")
    print(f"- Stats summary:    {stats_json}")
    print(f"- Figures:          {fig_dir}")
    if args.use_pdfs:
        print(f"PDF backend: {PDF_BACKEND or 'None (not available)'}")


if __name__ == "__main__":
    main()
