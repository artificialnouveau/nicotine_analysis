
# Nicotine / Tobacco / E-Cigarette Literature Collection & COI/Outcome Analysis

This repository contains two primary scripts:

1. **`fetch_data.py`** — Collects bibliographic metadata from **PubMed** (and optionally **Web of Science**) and downloads **open-access PDFs** when available.
2. **`analyze_data.py`** — Deduplicates records across sources and performs structured analysis to identify:
   - Product/topic focus (e-cigarettes / tobacco / nicotine / other)
   - Outcome direction (positive/negative/neutral/mixed) using **conclusion-like abstract sentences**
   - Conflicts of interest (declared COI, industry affiliations, and funding classification)
   - Basic association tests (industry involvement vs outcome)

---

## Contents

- [Nicotine / Tobacco / E-Cigarette Literature Collection \& COI/Outcome Analysis](#nicotine--tobacco--e-cigarette-literature-collection--coioutcome-analysis)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Repository Structure](#repository-structure)
  - [Prerequisites](#prerequisites)
    - [API keys and accounts (required to fetch)](#api-keys-and-accounts-required-to-fetch)
    - [Python environment](#python-environment)
  - [Queries](#queries)
  - [Usage](#usage)
    - [1) Fetch PubMed data](#1-fetch-pubmed-data)
    - [2) Fetch Web of Science data (optional)](#2-fetch-web-of-science-data-optional)
    - [3) Fetch both PubMed + WoS](#3-fetch-both-pubmed--wos)
    - [4) Analyze](#4-analyze)
  - [Outputs](#outputs)
  - [How the analysis classifies papers](#how-the-analysis-classifies-papers)
    - [Topic tags](#topic-tags)
    - [Outcome coding](#outcome-coding)
    - [Conflict of interest and industry involvement](#conflict-of-interest-and-industry-involvement)
    - [Funding source classification](#funding-source-classification)
    - [Deduplication](#deduplication)
    - [Association tests](#association-tests)
  - [Notes and limitations](#notes-and-limitations)
  - [Troubleshooting](#troubleshooting)
    - [PubMed error: `retstart cannot be larger than 9998`](#pubmed-error-retstart-cannot-be-larger-than-9998)
    - [No PDFs downloaded](#no-pdfs-downloaded)
    - [PDF-based extraction not working](#pdf-based-extraction-not-working)

---

## Overview

The goal is to assemble a dataset of papers relevant to **e-cigarettes / vaping / tobacco / nicotine**, then evaluate:

1. **What the papers are about** (topic tags)
2. Whether they report **positive vs negative findings** about these products/substances
3. Whether there is **conflict of interest**, defined as either:
   - a declared COI statement (or COI text detectable in PDFs), or
   - industry involvement via author affiliations and/or industry-linked funding signals
4. Whether there is a statistical association between **industry involvement** and **outcome direction**

---

## Repository Structure

Typical layout:

```

.
├── fetch_data.py
├── analyze_data.py
├── pubmed_query.txt
├── wos_query.txt
└── (output folder created at runtime, e.g. ./study_dump/)
├── data/
│   ├── pubmed_records.jsonl
│   └── wos_records.jsonl
├── pdfs/
│   ├── pmc/
│   └── oa/
└── analysis_v2/
├── canonical_records.csv
├── audit_evidence.csv
├── contingency_outcome_by_industry.csv
├── stats_summary.json
└── figures/
├── outcome_by_industry.png
├── outcome_by_declared_coi.png
└── funding_class_by_industry.png

````

---

## Prerequisites

### API keys and accounts (required to fetch)

To run `fetch_data.py`, you must have:

- **NCBI API key** (recommended/necessary for stable PubMed usage at scale)
- **Web of Science API key** (required only if using WoS)

You also must provide an email address for NCBI (and Unpaywall if enabled), as recommended by NCBI policies.

### Python environment

Recommended: Python 3.9+

Install dependencies:

```bash
pip install requests pandas matplotlib scipy
````

Optional (for PDF text extraction in `analyze_data.py`):

* PyMuPDF (recommended)

  ```bash
  pip install pymupdf
  ```
* OR pdfminer.six

  ```bash
  pip install pdfminer.six
  ```

If neither PDF library is installed, `analyze_data.py` will still run, but it will rely on metadata and will not extract funding/COI snippets from PDFs.

---

## Queries

The search queries are stored in:

* **`pubmed_query.txt`** — Used by `fetch_data.py` for PubMed
* **`wos_query.txt`** — Used by `fetch_data.py` for Web of Science

Edit these files to refine or broaden the search.

---

## Usage

### 1) Fetch PubMed data

This collects PubMed metadata and (optionally) downloads OA PDFs.

**Recommended (chunk by year to bypass PubMed’s 9,999 history paging limit):**

```bash
python fetch_data.py \
  --sources pubmed \
  --pubmed_query_file pubmed_query.txt \
  --out_dir ./study_dump \
  --email "you@domain.com" \
  --ncbi_api_key "<NCBI_API_KEY>" \
  --download_pdfs \
  --pubmed_chunk_by_year \
  --pubmed_start_year 1960 \
  --pubmed_end_year 2025 \
  --sleep 0.34
```

Notes:

* `--pubmed_chunk_by_year` splits the query into **one query per year**, each independently retrievable within PubMed’s limit.
* If a single year still exceeds 9,999 results, the script warns that you may need smaller chunks (e.g., month-based chunking).

### 2) Fetch Web of Science data (optional)

```bash
export WOS_API_KEY="<CLARIVATE_API_KEY>"

python fetch_data.py \
  --sources wos \
  --wos_query_file wos_query.txt \
  --out_dir ./study_dump \
  --sleep 0.34
```

### 3) Fetch both PubMed + WoS

```bash
export WOS_API_KEY="<CLARIVATE_API_KEY>"

python fetch_data.py \
  --sources both \
  --pubmed_query_file pubmed_query.txt \
  --wos_query_file wos_query.txt \
  --out_dir ./study_dump \
  --email "you@domain.com" \
  --ncbi_api_key "<NCBI_API_KEY>" \
  --download_pdfs \
  --pubmed_chunk_by_year \
  --pubmed_start_year 1960 \
  --pubmed_end_year 2025 \
  --sleep 0.34
```

### 4) Analyze

Run analysis on the fetched JSONL outputs.

**PubMed-only analysis:**

```bash
python analyze_data.py --out_dir ./study_dump
```

**Include WoS records (if fetched):**

```bash
python analyze_data.py --out_dir ./study_dump --include_wos
```

**Use PDFs for better COI/funding extraction (if PDFs exist):**

```bash
python analyze_data.py --out_dir ./study_dump --include_wos --use_pdfs
```

---

## Outputs

`analyze_data.py` writes to:

**`<out_dir>/analysis_v2/`**

* `canonical_records.csv`
  Deduplicated “master table” of papers (canonical records) with key classification fields.

* `audit_evidence.csv`
  Evidence table containing the exact sentences/snippets and rule hits that triggered each classification.

* `contingency_outcome_by_industry.csv`
  Cross-tab of outcome categories by industry involvement.

* `stats_summary.json`
  Summary of counts, chi-square results (when feasible), Fisher exact test (2x2), and odds ratio + confidence interval.

* `figures/`

  * `outcome_by_industry.png`
  * `outcome_by_declared_coi.png`
  * `funding_class_by_industry.png`

---

## How the analysis classifies papers

### Topic tags

Each paper receives one or more topic tags derived from title + abstract keyword matching:

* `E-cigarettes/ENDS`
* `Tobacco/Combustibles`
* `Nicotine/NRT`
* `Other/Unclear`

### Outcome coding

The outcome label is based on **conclusion-like sentences** extracted from the abstract:

1. Prefer sentences containing cues like “conclusion”, “overall”, “these findings suggest…”
2. If no cue sentences exist, use the final 2–3 sentences of the abstract.

Then it applies pattern rules to classify:

* `Positive` — signals of benefit/harm reduction/cessation etc.
* `Negative` — signals of harm/toxicity/risk increases etc.
* `Neutral` — signals like “no significant difference”, “inconclusive”
* `Mixed` — both positive and negative signals present
* `Not coded` — no strong match

Evidence sentences used for the decision are stored in `audit_evidence.csv`.

### Conflict of interest and industry involvement

The pipeline tracks:

1. **Declared COI**:

   * From `coi_statement` (PubMed metadata)
   * Optionally augmented by PDF snippet extraction (if `--use_pdfs` is enabled)

2. **Industry affiliation (author affiliations)**:

   * Pattern matching against a list of known tobacco/vape organizations (e.g., PMI, BAT, Altria, JUUL, etc.)

3. **Industry involvement (combined)**:

   * Defined as **Yes** if either:

     * Declared COI is Yes, OR
     * Industry affiliation is Yes

This combined flag is used in the main association tests.

### Funding source classification

Funding is classified separately from COI. If PDFs are available and `--use_pdfs` is enabled, the script searches for “Funding/Acknowledgments/Support” sections and classifies into:

* Industry
* Government
* Nonprofit
* University/Institutional
* Mixed
* None/Not stated

Evidence tags are recorded in `audit_evidence.csv`.

### Deduplication

PubMed and WoS results can overlap. Records are deduplicated into canonical papers using:

1. DOI (preferred)
2. Otherwise: normalized title + year + first author last name (fallback)

Across duplicates, the pipeline prefers the “best” filled metadata (e.g., PubMed COI, non-empty abstract).

### Association tests

The analysis produces:

* A contingency table of `Industry_Involved` (Yes/No) by outcome class
* Chi-square test (when applicable)
* A simplified 2x2 comparison of **Positive vs Not Positive** with:

  * Fisher exact test p-value
  * Odds ratio and 95% CI (with continuity correction when needed)

---

## Notes and limitations

* **PDFs are optional.** The analysis works without them, but funding/COI extraction is improved when PDFs are available.
* **WoS metadata may not include authors/affiliations by default** depending on your entitlement and API response options. Industry affiliation detection is strongest when affiliation fields are present.
* Keyword-based rules are **heuristic**. Use `audit_evidence.csv` to review the basis for each label and refine rules as needed.
* PubMed chunking by year resolves the **retstart <= 9998** history limitation, but exceptionally large years may still exceed 9,999 hits.

---

## Troubleshooting

### PubMed error: `retstart cannot be larger than 9998`

Use PubMed chunking:

```bash
--pubmed_chunk_by_year
```

If a single year still exceeds 9,999 results, narrow the query or implement smaller chunks (e.g., month ranges).

### No PDFs downloaded

This script downloads PDFs only when:

* PubMed Central OA provides a PDF (PMCID exists and OA link is available), or
* Unpaywall returns an OA PDF URL (only if `--use_unpaywall` is enabled)

### PDF-based extraction not working

Install one of:

* `pymupdf` (recommended), or
* `pdfminer.six`

Then re-run with:

```bash
python analyze_data.py --out_dir ./study_dump --use_pdfs
```