#!/usr/bin/env python3
"""
fetch_data.py

Fetch PubMed records for a query, save metadata to JSONL, and download PDFs ONLY when available via:
  - PubMed Central Open Access (PMC OA)
  - (Optional) Unpaywall OA PDF URLs by DOI

Also fetch Web of Science (Clarivate) metadata via Web of Science API Expanded:
  https://api.clarivate.com/api/wos
using X-ApiKey authentication.

Key feature:
  - PubMed chunking by publication year to bypass the ESearch history paging limit (<= 9,999 records per chunk).

Outputs (under --out_dir):
  data/
    pubmed_records.jsonl
    wos_records.jsonl
  pdfs/
    pmc/<PMCID>.pdf
    oa/<hash>.pdf  (Unpaywall OA PDFs)

Usage examples (see bottom of file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import xml.etree.ElementTree as ET

from scihub import SciHub

sh = SciHub()

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_FCGI = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Web of Science API Expanded (Clarivate)
WOS_EXPANDED_SEARCH = "https://api.clarivate.com/api/wos"

# PubMed ESearch/History paging limit
PUBMED_ESRCH_MAX = 9999


# -------------------------
# Utilities
# -------------------------

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sleep(base: float) -> None:
    time.sleep(base + random.uniform(0, base * 0.25))


def _requests_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s


def _http_request(
    session: requests.Session,
    url: str,
    *,
    method: str = "GET",
    params: Optional[dict] = None,
    data: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 60,
    max_retries: int = 5,
) -> requests.Response:
    """
    HTTP helper with retries and backoff.

    - Retries on 429 and 5xx.
    - Raises a descriptive error on 4xx (often query/limit issues).
    """
    last_err: Optional[Exception] = None
    method = method.upper()

    for attempt in range(1, max_retries + 1):
        try:
            if method == "POST":
                r = session.post(url, params=params, data=data, headers=headers, timeout=timeout)
            else:
                r = session.get(url, params=params, headers=headers, timeout=timeout)

            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")

            if 400 <= r.status_code < 500:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:400]}")

            r.raise_for_status()
            return r

        except Exception as e:
            last_err = e
            backoff = min(60, 2**attempt) + random.uniform(0, 1.0)
            time.sleep(backoff)

    raise RuntimeError(
        f"{method} failed after {max_retries} retries: {url} params={params} err={last_err}"
    )


def _clean_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def read_query_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        q = f.read().strip()
    if not q:
        raise ValueError(f"Query file is empty: {path}")
    return q


def _load_existing_ids(jsonl_path: str, id_field: str) -> set:
    """
    Used for --append mode: keep a set of IDs already written to JSONL.
    """
    ids = set()
    if not os.path.exists(jsonl_path):
        return ids
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                val = obj.get(id_field)
                if val:
                    ids.add(str(val))
            except Exception:
                continue
    return ids


def _hash_to_filename(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


def download_pdf(session: requests.Session, url: str, out_path: str, overwrite: bool) -> None:
    """
    Download PDF to out_path. Uses a .part temp file for atomic write.
    """
    _safe_mkdir(os.path.dirname(out_path))
    if not overwrite and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return

    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = out_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, out_path)


# -------------------------
# Data model
# -------------------------

@dataclass
class Author:
    last: str
    fore: str
    initials: str
    affiliations: List[str]


@dataclass
class UnifiedRecord:
    # Universal fields
    source_db: str  # "PUBMED" or "WOS"
    record_id: str  # PMID for PubMed; WoS UID for WoS

    # Bibliographic
    title: Optional[str]
    abstract: Optional[str]
    year: Optional[int]
    journal: Optional[str]
    doi: Optional[str]
    authors: Optional[List[Dict[str, Any]]]  # each author has last/fore/initials/affiliations


    # PubMed-specific
    pmid: Optional[str]
    pmcid: Optional[str]
    coi_statement: Optional[str]
    pmc_pdf_url: Optional[str]
    pdf_path: Optional[str]
    pmc_xml_path: Optional[str]
    abstract_sections: Optional[List[Dict[str, str]]]


    # WoS-specific
    wos_uid: Optional[str]
    wos_times_cited: Optional[int]

    # Optional bookkeeping
    pubmed_chunk_label: Optional[str] = None


# -------------------------
# PubMed ingestion
# -------------------------

def _pubmed_year_query(base_query: str, start_year: int, end_year: int) -> str:
    """
    Add a PubMed date-of-publication filter for a year range.

    PubMed supports [dp] (Date of Publication) range queries:
      (YYYY:YYYY[dp]) works as an inclusive range in practice for year filters.
    """
    return f"({base_query}) AND ({start_year}:{end_year}[dp])"


def pubmed_esearch(
    session: requests.Session,
    query: str,
    email: str,
    api_key: Optional[str],
    retmax: int,
) -> Tuple[int, str, str]:
    """
    ESearch using the NCBI history server.
    Uses POST to avoid URL-length limits for large queries.
    """
    params = {
        "db": "pubmed",
        "term": query,
        "usehistory": "y",
        "retmode": "xml",
        "retmax": min(100000, retmax),
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    r = _http_request(session, f"{NCBI_EUTILS}/esearch.fcgi", method="POST", data=params)
    root = ET.fromstring(r.text)

    count = int(root.findtext("./Count", default="0"))
    webenv = root.findtext("./WebEnv")
    query_key = root.findtext("./QueryKey")
    if not webenv or not query_key:
        raise RuntimeError("Failed to get WebEnv/QueryKey from esearch response.")
    return count, webenv, query_key


def pubmed_efetch_batch(
    session: requests.Session,
    webenv: str,
    query_key: str,
    email: str,
    api_key: Optional[str],
    retstart: int,
    retmax: int,
) -> str:
    """
    EFetch a batch from the history server.
    Uses POST for consistency and robustness.
    """
    params = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retmode": "xml",
        "retstart": retstart,
        "retmax": retmax,
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    r = _http_request(session, f"{NCBI_EUTILS}/efetch.fcgi", method="POST", data=params)
    return r.text


def parse_pubmed_article(article: ET.Element) -> UnifiedRecord:
    """
    Parse essential bibliographic fields from a PubmedArticle XML element.

    Key upgrades:
      - Preserves structured abstract labels as `abstract_sections`
      - Extracts authors + affiliations (needed for industry affiliation detection)
    """
    pmid = article.findtext(".//MedlineCitation/PMID")
    if not pmid:
        raise ValueError("Missing PMID.")

    title = _clean_text(article.findtext(".//Article/ArticleTitle"))
    journal = _clean_text(article.findtext(".//Article/Journal/Title"))

    # Abstract sections (preserve labels)
    abstract_sections: List[Dict[str, str]] = []
    abstract_nodes = article.findall(".//Article/Abstract/AbstractText")
    if abstract_nodes:
        for n in abstract_nodes:
            label = (n.attrib.get("Label") or n.attrib.get("NlmCategory") or "").strip()
            txt = _clean_text("".join(n.itertext()))
            if txt:
                abstract_sections.append({"label": label, "text": txt})

    # Collapsed abstract text (for backward compatibility / general NLP)
    abstract = None
    if abstract_sections:
        parts = []
        for sec in abstract_sections:
            if sec["label"]:
                parts.append(f'{sec["label"]}: {sec["text"]}')
            else:
                parts.append(sec["text"])
        abstract = _clean_text(" ".join(parts))

    # Year
    year = None
    year_txt = (
        article.findtext(".//Article/Journal/JournalIssue/PubDate/Year")
        or article.findtext(".//Article/ArticleDate/Year")
        or article.findtext(".//MedlineDate")
    )
    if year_txt:
        m = re.search(r"(19|20)\d{2}", year_txt)
        if m:
            year = int(m.group(0))

    # IDs (PMCID/DOI)
    pmcid = None
    doi = None
    for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = (aid.attrib.get("IdType", "") or "").lower()
        val = (aid.text or "").strip()
        if id_type == "pmc" and val:
            pmcid = val if val.startswith("PMC") else f"PMC{val}"
        elif id_type == "doi" and val:
            doi = val

    coi_statement = _clean_text(article.findtext(".//CoiStatement"))

    # Authors + affiliations (critical for industry detection)
    authors: List[Dict[str, Any]] = []
    for a in article.findall(".//Article/AuthorList/Author"):
        last = (a.findtext("./LastName") or "").strip()
        fore = (a.findtext("./ForeName") or "").strip()
        initials = (a.findtext("./Initials") or "").strip()

        affs: List[str] = []
        for aff in a.findall("./AffiliationInfo/Affiliation"):
            txt = _clean_text("".join(aff.itertext()))
            if txt:
                affs.append(txt)

        # Keep only if we have at least a name
        if last or fore:
            authors.append(
                {"last": last, "fore": fore, "initials": initials, "affiliations": affs}
            )

    return UnifiedRecord(
        source_db="PUBMED",
        record_id=str(pmid),
        title=title,
        abstract=abstract,
        year=year,
        journal=journal,
        doi=doi,
        pmid=str(pmid),
        pmcid=pmcid,
        coi_statement=coi_statement,
        pmc_pdf_url=None,
        pdf_path=None,
        wos_uid=None,
        wos_times_cited=None,
        pubmed_chunk_label=None,
        pmc_xml_path=None,
        abstract_sections=abstract_sections if abstract_sections else None,
        authors=authors
    )


def pmc_oa_lookup_pdf_url(
    session: requests.Session,
    pmcid: str,
    email: str,
    api_key: Optional[str],
) -> Optional[str]:
    links = pmc_oa_lookup_links(session, pmcid, email, api_key)
    return links.get("pdf")

    """
    Query PMC OA service and return a dict mapping link format -> URL.
    Common formats include: 'pdf', 'tgz' (OA package), sometimes 'xml'/'nxml' (varies).
    """
    params = {"id": pmcid, "format": "xml", "email": email}
    if api_key:
        params["api_key"] = api_key

    r = _http_request(session, PMC_OA_FCGI, method="GET", params=params)
    root = ET.fromstring(r.text)

    links: Dict[str, str] = {}
    for link in root.findall(".//link"):
        fmt = (link.attrib.get("format") or "").lower().strip()
        href = (link.attrib.get("href") or "").strip()
        if fmt and href:
            links[fmt] = href
    return links

# -------------------------
# Unpaywall (optional OA PDF)
# -------------------------

def unpaywall_best_pdf_url(session: requests.Session, doi: str, email: str) -> Optional[str]:
    """
    Unpaywall REST API: https://api.unpaywall.org/v2/<doi>?email=<email>

    Returns an OA PDF URL if best_oa_location.url_for_pdf exists; otherwise falls back to any OA location.
    """
    doi = doi.strip()
    if not doi:
        return None

    url = f"https://api.unpaywall.org/v2/{doi}"
    r = _http_request(session, url, method="GET", params={"email": email}, timeout=60)
    data = r.json()

    best = data.get("best_oa_location") or {}
    pdf = best.get("url_for_pdf")
    if pdf:
        return pdf

    for loc in data.get("oa_locations") or []:
        pdf = loc.get("url_for_pdf")
        if pdf:
            return pdf
    return None


# -------------------------
# Web of Science ingestion (Expanded API)
# -------------------------

def _deep_get(obj: Any, path: List[Any], default: Any = None) -> Any:
    cur = obj
    for key in path:
        try:
            if isinstance(key, int):
                cur = cur[key]
            else:
                cur = cur.get(key)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def download_pmc_jats_xml(
    session: requests.Session,
    pmcid: str,
    email: str,
    api_key: Optional[str],
    out_xml_dir: str,
    overwrite: bool,
    oa_links: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Download PMC JATS XML (preferred) for an OA PMCID.

    Priority:
      1) OA package (format=tgz) -> extract .nxml/.xml from tarball
      2) direct xml/nxml link if present
      3) fallback: E-utilities efetch db=pmc&id=<pmcid>&retmode=xml

    Returns path to saved .nxml (or .xml) or None.
    """
    import tarfile
    import tempfile

    _safe_mkdir(out_xml_dir)
    final_path = os.path.join(out_xml_dir, f"{pmcid}.nxml")

    if (not overwrite) and os.path.exists(final_path) and os.path.getsize(final_path) > 0:
        return final_path

    try:
        if oa_links is None:
            oa_links = pmc_oa_lookup_links(session, pmcid, email, api_key)

        # 1) Try OA tarball (best: includes JATS .nxml)
        tgz_url = oa_links.get("tgz")
        if tgz_url:
            with tempfile.TemporaryDirectory() as td:
                tgz_path = os.path.join(td, f"{pmcid}.tgz")
                download_pdf(session, tgz_url, tgz_path, overwrite=True)  # reuse downloader (binary-safe)

                # Extract first .nxml or .xml found
                with tarfile.open(tgz_path, "r:gz") as tf:
                    members = tf.getmembers()
                    # Prefer *.nxml, else *.xml
                    nxml_members = [m for m in members if m.name.lower().endswith(".nxml")]
                    xml_members = [m for m in members if m.name.lower().endswith(".xml")]

                    target = None
                    if nxml_members:
                        target = nxml_members[0]
                    elif xml_members:
                        target = xml_members[0]

                    if target is None:
                        return None

                    extracted = tf.extractfile(target)
                    if not extracted:
                        return None
                    data = extracted.read()
                    if not data:
                        return None

                tmp = final_path + ".part"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, final_path)
                return final_path

        # 2) Direct xml/nxml link (not always present)
        direct_xml = oa_links.get("xml") or oa_links.get("nxml")
        if direct_xml:
            tmp = final_path + ".part"
            with session.get(direct_xml, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp, final_path)
            return final_path

        # 3) Fallback: efetch from PMC
        params = {"db": "pmc", "id": pmcid, "retmode": "xml", "email": email}
        if api_key:
            params["api_key"] = api_key
        r = _http_request(session, f"{NCBI_EUTILS}/efetch.fcgi", method="POST", data=params, timeout=90)

        # Basic sanity check: should look like XML; still save if it does
        txt = r.text.strip()
        if not txt.startswith("<"):
            return None

        tmp = final_path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(txt)
        os.replace(tmp, final_path)
        return final_path

    except Exception:
        return None

def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def wos_search_page(
    session: requests.Session,
    api_key: str,
    usr_query: str,
    database_id: str,
    count: int,
    first_record: int,
    option_view: str,
    wos_base_url: str,
) -> Dict[str, Any]:
    """
    Query the WoS Expanded API for a page of records.
    """
    params = {
        "databaseId": database_id,
        "usrQuery": usr_query,
        "count": count,
        "firstRecord": first_record,  # 1-indexed
        "optionView": option_view,    # FR (full record) or SR (short record)
    }
    headers = {"X-ApiKey": api_key, "Accept": "application/json"}
    r = _http_request(session, wos_base_url, method="GET", params=params, headers=headers, timeout=90)
    return r.json()


def parse_wos_rec(rec: Dict[str, Any]) -> UnifiedRecord:
    """
    Parse essential fields from a WoS record. WoS JSON structure varies across entitlements/optionView.
    """
    uid = rec.get("UID") or rec.get("uid") or rec.get("UT")
    uid = str(uid) if uid else None

    title = None
    titles = _deep_get(rec, ["static_data", "summary", "titles", "title"], default=None)
    for t in _as_list(titles):
        if isinstance(t, dict):
            title = title or t.get("content")
    title = _clean_text(title)

    journal = None
    for t in _as_list(titles):
        if isinstance(t, dict) and (t.get("type") or "").lower() in ("source", "source_title", "journal"):
            journal = journal or t.get("content")
    journal = _clean_text(journal)

    year = None
    pub_info = _deep_get(rec, ["static_data", "summary", "pub_info"], default={})
    if isinstance(pub_info, dict):
        y = pub_info.get("pubyear") or pub_info.get("pubyear_display")
        if y:
            try:
                year = int(str(y)[:4])
            except Exception:
                year = None

    doi = None
    idents = _deep_get(rec, ["dynamic_data", "cluster_related", "identifiers", "identifier"], default=None)
    for ident in _as_list(idents):
        if isinstance(ident, dict) and (ident.get("type") or "").lower() == "doi":
            doi = ident.get("value")
    doi = _clean_text(doi)

    abstract = None
    abs_obj = _deep_get(rec, ["static_data", "fullrecord_metadata", "abstracts", "abstract"], default=None)
    for a in _as_list(abs_obj):
        txt = _deep_get(a, ["abstract_text", "p"], default=None)
        if txt:
            if isinstance(txt, list):
                abstract = " ".join([str(p) for p in txt if p])
            else:
                abstract = str(txt)
            break
    abstract = _clean_text(abstract)

    times_cited = None
    tc = _deep_get(rec, ["dynamic_data", "citation_related", "tc_list", "silo_tc"], default=None)
    if isinstance(tc, list) and tc:
        try:
            times_cited = int(tc[0].get("local_count"))
        except Exception:
            times_cited = None
    elif isinstance(tc, dict):
        try:
            times_cited = int(tc.get("local_count"))
        except Exception:
            times_cited = None

    return UnifiedRecord(
        source_db="WOS",
        record_id=str(uid) if uid else "UNKNOWN_WOS_UID",
        title=title,
        abstract=abstract,
        year=year,
        journal=journal,
        doi=doi,
        pmid=None,
        pmcid=None,
        coi_statement=None,
        pmc_pdf_url=None,
        pdf_path=None,
        wos_uid=str(uid) if uid else None,
        wos_times_cited=times_cited,
        pubmed_chunk_label=None,
    )


def pmc_oa_lookup_links(
    session: requests.Session,
    pmcid: str,
    email: str,
    api_key: Optional[str],
) -> Dict[str, Optional[str]]:
    """
    Lookup PubMed Central (PMC) Open Access links for a PMCID via the PMC OA service.

    Why this exists
    ---------------
    The PMC OA endpoint can return multiple links (PDF, XML, TGZ package, etc.) for the OA version
    of a paper. We use it to:
      - download the PDF (format="pdf")
      - download the JATS XML (format="xml") or, preferably, the .tgz package when available

    Endpoint
    --------
    https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=<PMCID>&format=xml

    Parameters
    ----------
    session : requests.Session
        Configured requests session with a User-Agent.
    pmcid : str
        A PMC ID like "PMC1234567" (either already prefixed with PMC or not).
    email : str
        Contact email per NCBI recommendations.
    api_key : Optional[str]
        NCBI API key (optional).

    Returns
    -------
    Dict[str, Optional[str]]
        Keys include:
          - "pdf": direct OA PDF URL (if present)
          - "xml": direct OA JATS XML URL (if present)
          - "tgz": OA tarball package URL (if present; often includes XML + assets)
          - "other": list-like string of other formats (rarely used), or None
          - "all": dict-style string of all discovered links (debug)

        Missing formats will be returned as None.

    Notes
    -----
    - Not all PMC records are OA; for non-OA records the response may have no usable links.
    - Some OA records provide 'tgz' but not 'xml'; in that case you can fetch tgz and extract XML.
    """
    if not pmcid:
        return {"pdf": None, "xml": None, "tgz": None, "other": None, "all": None}

    pmcid = pmcid.strip()
    if pmcid and not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    params = {"id": pmcid, "format": "xml", "email": email}
    if api_key:
        params["api_key"] = api_key

    r = _http_request(session, PMC_OA_FCGI, method="GET", params=params, timeout=60)
    root = ET.fromstring(r.text)

    # PMC OA response structure:
    # <oa>
    #   <records>
    #     <record id="PMCxxxx" ...>
    #       <link format="pdf" href="..."/>
    #       <link format="tgz" href="..."/>
    #       <link format="xml" href="..."/>
    #     </record>
    #   </records>
    # </oa>
    links: Dict[str, Optional[str]] = {"pdf": None, "xml": None, "tgz": None}
    all_links: Dict[str, str] = {}

    for link in root.findall(".//link"):
        fmt = (link.attrib.get("format") or "").strip().lower()
        href = (link.attrib.get("href") or "").strip()
        if not fmt or not href:
            continue

        # Keep first occurrence per format (usually only one)
        all_links[fmt] = href
        if fmt in links and not links[fmt]:
            links[fmt] = href

    # Capture any other link formats (e.g., "epub", etc.) for debugging
    other_fmts = sorted([k for k in all_links.keys() if k not in ("pdf", "xml", "tgz")])
    other_str = ",".join(other_fmts) if other_fmts else None

    out: Dict[str, Optional[str]] = {
        "pdf": links.get("pdf"),
        "xml": links.get("xml"),
        "tgz": links.get("tgz"),
        "other": other_str,
        "all": json.dumps(all_links, ensure_ascii=False) if all_links else None,
    }
    return out


# -------------------------
# Main
# -------------------------
def main() -> None:
    import argparse
    import json
    import os
    import re
    import shutil
    import unicodedata
    import xml.etree.ElementTree as ET
    from dataclasses import asdict
    from datetime import datetime
    from typing import Optional

    # -------------------------
    # Local helpers (kept inside main to make this drop-in)
    # -------------------------
    def safe_pdf_filename_from_title(title: str, max_len: int = 180) -> str:
        """Sanitize title into a filesystem-safe basename (ASCII, no special chars)."""
        if not title:
            return "paper"
        s = unicodedata.normalize("NFKD", title)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^A-Za-z0-9 _-]+", "", s)
        s = re.sub(r"[ _-]+", "_", s).strip("_")
        return (s or "paper")[:max_len]

    def _download_ok(path: str) -> bool:
        try:
            return os.path.exists(path) and os.path.getsize(path) > 0
        except OSError:
            return False

    def _looks_like_pdf(path: str) -> bool:
        """Basic signature check to avoid treating HTML as a PDF."""
        try:
            with open(path, "rb") as f:
                return f.read(4) == b"%PDF"
        except OSError:
            return False

    def _title_pdf_path(out_dir: str, title: str, suffix: Optional[str]) -> str:
        """
        Build a title-based pdf path. If collision and overwrite=False, append suffix.
        """
        base = safe_pdf_filename_from_title(title)
        p = os.path.join(out_dir, f"{base}.pdf")
        if (not args.overwrite) and os.path.exists(p):
            if suffix:
                suf = safe_pdf_filename_from_title(str(suffix), max_len=40)
                p = os.path.join(out_dir, f"{base}_{suf}.pdf")
            else:
                p = os.path.join(out_dir, f"{base}_dup.pdf")
        return p

    # -------------------------
    # Args
    # -------------------------
    ap = argparse.ArgumentParser()

    # Sources
    ap.add_argument(
        "--sources",
        default="pubmed",
        help="Comma-separated: pubmed,wos,both (e.g., 'pubmed', 'wos', 'both').",
    )

    # PubMed
    ap.add_argument("--pubmed_query_file", default=None, help="Text file containing the PubMed query.")
    ap.add_argument("--email", default=None, help="Email for NCBI/Unpaywall (required for PubMed).")
    ap.add_argument("--ncbi_api_key", default=None, help="NCBI API key (optional but recommended).")
    ap.add_argument("--pubmed_batch_size", type=int, default=200, help="PubMed EFetch batch size.")
    ap.add_argument("--pubmed_retmax", type=int, default=1000000, help="Max PubMed records to fetch (cap applies per chunk).")

    # PubMed chunking (Option B)
    ap.add_argument(
        "--pubmed_chunk_by_year",
        action="store_true",
        help="Split PubMed query by publication year to bypass the 9,999 history paging limit.",
    )
    ap.add_argument("--pubmed_start_year", type=int, default=1960, help="Start year for PubMed chunking.")
    ap.add_argument(
        "--pubmed_end_year",
        type=int,
        default=None,
        help="End year for PubMed chunking (defaults to current year).",
    )

    # PDFs / full text artifacts
    ap.add_argument("--download_pdfs", action="store_true", help="Download PDFs when available (PMC OA; optionally Unpaywall OA).")
    ap.add_argument("--use_unpaywall", action="store_true", help="Also try Unpaywall OA PDFs by DOI (requires --email).")

    # NEW: PMC JATS XML
    ap.add_argument(
        "--download_pmc_xml",
        action="store_true",
        help="Download PMC JATS XML for PMCID records (best effort). "
             "If --download_pdfs is set, XML is also downloaded by default.",
    )

    # WoS
    ap.add_argument("--wos_query_file", default=None, help="Text file containing the WoS Advanced Search query (usrQuery).")
    ap.add_argument("--wos_api_key", default=os.environ.get("WOS_API_KEY"), help="WoS API key (or set env WOS_API_KEY).")
    ap.add_argument("--wos_database_id", default="WOS", help="WoS databaseId (commonly WOS or WOK depending on entitlement).")
    ap.add_argument("--wos_count", type=int, default=100, help="WoS page size.")
    ap.add_argument("--wos_option_view", default="FR", help="WoS optionView: FR (full) or SR (short).")
    ap.add_argument("--wos_base_url", default=WOS_EXPANDED_SEARCH, help="Override WoS base URL if needed.")

    # Output / rate limiting
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--sleep", type=float, default=0.34, help="Sleep seconds between requests.")
    ap.add_argument("--overwrite", action="store_true", help="Delete previous output directory contents before writing new output.")
    ap.add_argument("--append", action="store_true", help="Append to existing JSONL files (skip IDs already present).")

    args = ap.parse_args()

    sources = args.sources.strip().lower()
    if sources == "both":
        want_pubmed = True
        want_wos = True
    else:
        parts = [p.strip() for p in sources.split(",") if p.strip()]
        want_pubmed = "pubmed" in parts
        want_wos = "wos" in parts

    out_dir = args.out_dir
    data_dir = os.path.join(out_dir, "data")
    pdf_dir = os.path.join(out_dir, "pdfs")
    pmc_pdf_dir = os.path.join(pdf_dir, "pmc")
    oa_pdf_dir = os.path.join(pdf_dir, "oa")

    # NEW: PMC XML dir
    pmc_xml_dir = os.path.join(data_dir, "pmc_xml")

    if args.overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    _safe_mkdir(data_dir)
    _safe_mkdir(pmc_pdf_dir)
    _safe_mkdir(oa_pdf_dir)
    _safe_mkdir(pmc_xml_dir)

    pubmed_path = os.path.join(data_dir, "pubmed_records.jsonl")
    wos_path = os.path.join(data_dir, "wos_records.jsonl")

    pubmed_mode = "a" if args.append else "w"
    wos_mode = "a" if args.append else "w"

    session = _requests_session(user_agent="nicotine-coi-analysis/1.0 (academic research)")

    # ---------------- PubMed ----------------
    if want_pubmed:
        if not args.pubmed_query_file:
            raise SystemExit("--pubmed_query_file is required when sources include pubmed.")
        if not args.email:
            raise SystemExit("--email is required for PubMed (NCBI recommends a contact email).")

        base_query = read_query_text(args.pubmed_query_file)

        if args.pubmed_end_year is None:
            args.pubmed_end_year = datetime.utcnow().year

        if args.pubmed_chunk_by_year:
            year_ranges = [(y, y) for y in range(args.pubmed_start_year, args.pubmed_end_year + 1)]
        else:
            year_ranges = [(None, None)]

        existing_pmids = _load_existing_ids(pubmed_path, "record_id") if args.append else set()

        wrote = 0
        with open(pubmed_path, pubmed_mode, encoding="utf-8") as out_f:
            for (y0, y1) in year_ranges:
                if y0 is None:
                    query = base_query
                    label = "ALL_YEARS"
                else:
                    query = _pubmed_year_query(base_query, y0, y1)
                    label = f"{y0}"

                count, webenv, query_key = pubmed_esearch(
                    session=session,
                    query=query,
                    email=args.email,
                    api_key=args.ncbi_api_key,
                    retmax=args.pubmed_retmax,
                )

                # PubMed history paging limit enforcement
                total = min(count, args.pubmed_retmax, PUBMED_ESRCH_MAX)

                print(f"[PubMed:{label}] hits: {count} (will fetch {total})")
                if count > PUBMED_ESRCH_MAX:
                    print(
                        f"[WARN][PubMed:{label}] This chunk exceeds {PUBMED_ESRCH_MAX} hits. "
                        f"Only the first {PUBMED_ESRCH_MAX} will be fetched. "
                        f"If you need full coverage for {label}, split further (e.g., by months)."
                    )

                for retstart in range(0, total, args.pubmed_batch_size):
                    xml_text = pubmed_efetch_batch(
                        session=session,
                        webenv=webenv,
                        query_key=query_key,
                        email=args.email,
                        api_key=args.ncbi_api_key,
                        retstart=retstart,
                        retmax=min(args.pubmed_batch_size, total - retstart),
                    )

                    root = ET.fromstring(xml_text)
                    articles = root.findall(".//PubmedArticle")

                    for art in articles:
                        rec = parse_pubmed_article(art)
                        rec.pubmed_chunk_label = label

                        if rec.record_id in existing_pmids:
                            continue

                        # ---------- Full-text artifacts (best effort; OA only) ----------
                        # Decide whether to download PMC XML:
                        want_xml = args.download_pmc_xml or args.download_pdfs

                        # PMC OA links (reuse for PDF + XML)
                        oa_links = None
                        if rec.pmcid and (args.download_pdfs or want_xml):
                            try:
                                oa_links = pmc_oa_lookup_links(session, rec.pmcid, args.email, args.ncbi_api_key)
                            except Exception as e:
                                oa_links = None
                                print(f"[WARN][PubMed:{label}] PMC OA lookup failed for {rec.pmcid}: {e}")

                        # 1) PMC OA PDF (TITLE-BASED filename)
                        if args.download_pdfs and rec.pmcid and oa_links:
                            try:
                                pdf_url = oa_links.get("pdf")
                                if pdf_url:
                                    rec.pmc_pdf_url = pdf_url

                                    title_for_name = getattr(rec, "title", "") or rec.pmcid
                                    pdf_path = _title_pdf_path(
                                        out_dir=pmc_pdf_dir,
                                        title=title_for_name,
                                        suffix=rec.pmcid,  # used only on collision when overwrite=False
                                    )

                                    download_pdf(session, pdf_url, pdf_path, overwrite=args.overwrite)

                                    if _download_ok(pdf_path) and _looks_like_pdf(pdf_path):
                                        print(f"[OK][PubMed:{label}] Downloaded PMC OA PDF: {pdf_path}")
                                        rec.pdf_path = pdf_path
                                    else:
                                        print(f"[WARN][PubMed:{label}] PMC download wrote non-PDF/empty file: {pdf_path}")

                            except Exception as e:
                                print(f"[WARN][PubMed:{label}] PMC PDF failed for {rec.pmcid}: {e}")

                        # 2) PMC JATS XML (preferred via tgz, fallback efetch)
                        if want_xml and rec.pmcid:
                            try:
                                rec.pmc_xml_path = download_pmc_jats_xml(
                                    session=session,
                                    pmcid=rec.pmcid,
                                    email=args.email,
                                    api_key=args.ncbi_api_key,
                                    out_xml_dir=pmc_xml_dir,
                                    overwrite=args.overwrite,
                                    oa_links=oa_links,
                                )
                            except Exception as e:
                                rec.pmc_xml_path = None
                                print(f"[WARN][PubMed:{label}] PMC XML failed for {rec.pmcid}: {e}")

                        # 3) Unpaywall OA PDF by DOI (only if we don't already have a PDF)
                        if args.download_pdfs and args.use_unpaywall:
                            if not args.email:
                                raise SystemExit("--email is required for Unpaywall.")
                            if rec.doi and (not rec.pdf_path):
                                title_for_name = getattr(rec, "title", "") or rec.doi
                                doi_suffix = (_hash_to_filename(rec.doi)[:8] if rec.doi else None)
                                pdf_path = _title_pdf_path(
                                    out_dir=oa_pdf_dir,
                                    title=title_for_name,
                                    suffix=doi_suffix,  # used only on collision when overwrite=False
                                )

                                try:
                                    up_pdf = unpaywall_best_pdf_url(session, rec.doi, email=args.email)
                                    if up_pdf:
                                        download_pdf(session, up_pdf, pdf_path, overwrite=args.overwrite)

                                        if _download_ok(pdf_path) and _looks_like_pdf(pdf_path):
                                            print(f"[OK][PubMed:{label}] Downloaded Unpaywall OA PDF: {pdf_path}")
                                            rec.pdf_path = pdf_path
                                        else:
                                            print(f"[WARN][PubMed:{label}] Unpaywall wrote non-PDF/empty file: {pdf_path}")
                                    else:
                                        print(f"[WARN][PubMed:{label}] Unpaywall returned no OA PDF for DOI={rec.doi}")

                                except Exception as e:
                                    print(f"[WARN][PubMed:{label}] Unpaywall PDF failed for DOI={rec.doi}: {e}")

                                    # Fallback URL: record link/url if present, else DOI resolver
                                    fallback_url = (
                                        getattr(rec, "link", None)
                                        or getattr(rec, "url", None)
                                        or f"https://doi.org/{rec.doi}"
                                    )

                                    try:
                                        # Per your request: keep the exact form `result = sh.download('...', path='...')`
                                        result = sh.fetch(fallback_url)
                                        with open(pdf_path, 'wb+') as fd:
                                            fd.write(result['pdf'])

                                        if _download_ok(pdf_path) and _looks_like_pdf(pdf_path):
                                            print(f"[OK][PubMed:{label}] Downloaded fallback PDF via sh.download: {pdf_path}")
                                            rec.pdf_path = pdf_path
                                        else:
                                            print(f"[WARN][PubMed:{label}] sh.download wrote non-PDF/empty file: {pdf_path}")

                                    except Exception as e2:
                                        print(f"[WARN][PubMed:{label}] sh.download fallback failed for {fallback_url}: {e2}")

                        out_f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                        existing_pmids.add(rec.record_id)
                        wrote += 1

                    print(
                        f"[PubMed:{label}] fetched {min(retstart + args.pubmed_batch_size, total)}/{total} "
                        f"(wrote={wrote})"
                    )
                    _sleep(args.sleep)

        print(f"[PubMed] wrote: {pubmed_path}")

    # ---------------- Web of Science ----------------
    if want_wos:
        if not args.wos_query_file:
            raise SystemExit("--wos_query_file is required when sources include wos.")
        if not args.wos_api_key:
            raise SystemExit("--wos_api_key is required for WoS (or set env WOS_API_KEY).")

        usr_query = read_query_text(args.wos_query_file)

        existing_uids = _load_existing_ids(wos_path, "record_id") if args.append else set()

        first = 1
        page_size = max(1, min(args.wos_count, 100))
        total_found: Optional[int] = None
        wrote = 0

        with open(wos_path, wos_mode, encoding="utf-8") as out_f:
            while True:
                payload = wos_search_page(
                    session=session,
                    api_key=args.wos_api_key,
                    usr_query=usr_query,
                    database_id=args.wos_database_id,
                    count=page_size,
                    first_record=first,
                    option_view=args.wos_option_view,
                    wos_base_url=args.wos_base_url,
                )

                if total_found is None:
                    total_found = int(_deep_get(payload, ["QueryResult", "RecordsFound"], default=0))
                    print(f"[WoS] records found: {total_found}")

                recs = _deep_get(payload, ["Data", "Records", "records", "REC"], default=[])
                recs = _as_list(recs)

                if not recs:
                    break

                for rec_obj in recs:
                    try:
                        rec = parse_wos_rec(rec_obj)
                        if rec.record_id in existing_uids:
                            continue
                        out_f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                        existing_uids.add(rec.record_id)
                        wrote += 1
                    except Exception as e:
                        print(f"[WARN][WoS] failed to parse record at firstRecord={first}: {e}")

                first += page_size
                print(f"[WoS] progress: firstRecord={first} wrote={wrote}/{total_found}")
                _sleep(args.sleep)

                if total_found is not None and first > total_found:
                    break

        print(f"[WoS] wrote: {wos_path}")

    print("Done.")


if __name__ == "__main__":
    main()


"""
-------------------------
Run examples
-------------------------

1) PubMed only, chunked by year (recommended to bypass 9,999 limit per query):
  python fetch_data.py \
    --sources pubmed \
    --pubmed_query_file pubmed_query.txt \
    --out_dir ./study_dump \
    --email "ahnjili@gmail.com" \
    --ncbi_api_key "29bc87a06c11260f28077927d1a9d280d008" \
    --download_pdfs \
    --use_unpaywall \
    --pubmed_chunk_by_year \
    --pubmed_start_year 1964 \
    --pubmed_end_year 2025 \
    --sleep 0.34

2) Web of Science only:
  export WOS_API_KEY="<CLARIVATE_API_KEY>"
  python fetch_data.py \
    --sources wos \
    --wos_query_file wos_query.txt \
    --out_dir ./study_dump \
    --sleep 0.34

3) Both PubMed + WoS:
  export WOS_API_KEY="<CLARIVATE_API_KEY>"
  python fetch_data.py \
    --sources both \
    --pubmed_query_file pubmed_query.txt \
    --wos_query_file wos_query.txt \
    --out_dir ./study_dump \
    --email "ahnjili@gmail.com" \
    --ncbi_api_key "29bc87a06c11260f28077927d1a9d280d008" \
    --download_pdfs \
    --pubmed_chunk_by_year \
    --pubmed_start_year 1964 \
    --pubmed_end_year 2024 \
    --sleep 0.34

4) Append (do not overwrite; skip IDs already in JSONL):
  python fetch_data.py ... --append

5) Overwrite everything (delete old outputs first):
  python fetch_data.py ... --overwrite

Optional: Unpaywall OA PDFs by DOI (best-effort):
  python fetch_data.py ... --download_pdfs --use_unpaywall --email "you@domain.com"
"""

