#!/usr/bin/env python3
"""
fetch_data.py

Fetch PubMed records for a query, save metadata to JSONL, and download PDFs
ONLY when available via:
  - PubMed Central Open Access (PMC OA)
  - (Optional) Unpaywall OA PDF URLs by DOI

Also fetch Web of Science (Clarivate) metadata via Web of Science API Expanded
(https://api.clarivate.com/api/wos) using X-ApiKey authentication.

Usage examples are at the bottom of this file.

Outputs (under --out_dir):
  data/
    pubmed_records.jsonl
    wos_records.jsonl
  pdfs/
    pmc/<PMCID>.pdf
    oa/<hash_or_doi>.pdf  (Unpaywall OA PDFs)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests
import xml.etree.ElementTree as ET

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_FCGI = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Web of Science API Expanded (Clarivate)
WOS_EXPANDED_SEARCH = "https://api.clarivate.com/api/wos"


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sleep(base: float) -> None:
    time.sleep(base + random.uniform(0, base * 0.25))


def _requests_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s


def _http_get(
    session: requests.Session,
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 60,
    max_retries: int = 5,
) -> requests.Response:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            # Retry on throttling or transient server errors
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            backoff = min(60, 2**attempt) + random.uniform(0, 1.0)
            time.sleep(backoff)
    raise RuntimeError(f"GET failed after {max_retries} retries: {url} params={params} err={last_err}")


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
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]
    return h


def download_pdf(session: requests.Session, url: str, out_path: str, overwrite: bool) -> None:
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
# PubMed ingestion
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

    # PubMed-specific
    pmid: Optional[str]
    pmcid: Optional[str]
    coi_statement: Optional[str]
    pmc_pdf_url: Optional[str]
    pdf_path: Optional[str]

    # WoS-specific
    wos_uid: Optional[str]
    wos_times_cited: Optional[int]


def pubmed_esearch(
    session: requests.Session,
    query: str,
    email: str,
    api_key: Optional[str],
    retmax: int,
) -> Tuple[int, str, str]:
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

    r = _http_get(session, f"{NCBI_EUTILS}/esearch.fcgi", params=params)
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

    r = _http_get(session, f"{NCBI_EUTILS}/efetch.fcgi", params=params)
    return r.text


def parse_pubmed_article(article: ET.Element) -> UnifiedRecord:
    pmid = article.findtext(".//MedlineCitation/PMID")
    if not pmid:
        raise ValueError("Missing PMID.")

    title = _clean_text(article.findtext(".//Article/ArticleTitle"))
    journal = _clean_text(article.findtext(".//Article/Journal/Title"))

    abstract_nodes = article.findall(".//Article/Abstract/AbstractText")
    abstract = None
    if abstract_nodes:
        parts = []
        for n in abstract_nodes:
            label = n.attrib.get("Label")
            txt = "".join(n.itertext())
            txt = _clean_text(txt)
            if txt:
                parts.append(f"{label}: {txt}" if label else txt)
        abstract = _clean_text(" ".join(parts))

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
    )


def pmc_oa_lookup_pdf_url(
    session: requests.Session,
    pmcid: str,
    email: str,
    api_key: Optional[str],
) -> Optional[str]:
    params = {"id": pmcid, "format": "xml", "email": email}
    if api_key:
        params["api_key"] = api_key

    r = _http_get(session, PMC_OA_FCGI, params=params)
    root = ET.fromstring(r.text)
    for link in root.findall(".//link"):
        fmt = (link.attrib.get("format") or "").lower()
        href = link.attrib.get("href")
        if fmt == "pdf" and href:
            return href
    return None


# -------------------------
# Unpaywall (optional OA PDF)
# -------------------------

def unpaywall_best_pdf_url(session: requests.Session, doi: str, email: str) -> Optional[str]:
    """
    Unpaywall REST API: https://api.unpaywall.org/v2/<doi>?email=<email>

    Returns a PDF URL if an OA location with url_for_pdf exists; otherwise None.
    """
    doi = doi.strip()
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}"
    r = _http_get(session, url, params={"email": email}, timeout=60)
    data = r.json()

    best = data.get("best_oa_location") or {}
    pdf = best.get("url_for_pdf")
    if pdf:
        return pdf

    # Fall back to any OA location
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
    params = {
        "databaseId": database_id,
        "usrQuery": usr_query,
        "count": count,
        "firstRecord": first_record,  # 1-indexed
        "optionView": option_view,    # FR (full record) or SR (short record)
    }
    headers = {"X-ApiKey": api_key, "Accept": "application/json"}
    r = _http_get(session, wos_base_url, params=params, headers=headers, timeout=90)
    return r.json()


def parse_wos_rec(rec: Dict[str, Any]) -> UnifiedRecord:
    uid = rec.get("UID") or rec.get("uid") or rec.get("UT")
    uid = str(uid) if uid else None

    title = None
    # Typical location: static_data.summary.titles.title[*]
    titles = _deep_get(rec, ["static_data", "summary", "titles", "title"], default=None)
    for t in _as_list(titles):
        if isinstance(t, dict):
            # Common keys: "type" and "content"
            if (t.get("type") or "").lower() in ("item", "source", "title"):
                title = title or t.get("content")
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

    # Abstract often lives under fullrecord_metadata.abstracts.*; structure varies.
    abstract = None
    abs_obj = _deep_get(rec, ["static_data", "fullrecord_metadata", "abstracts", "abstract"], default=None)
    # Some responses have abstract as list; each has abstract_text with paragraphs
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
    # Sometimes tc_list.silo_tc is a list of dicts; sometimes tc_list has "local_count"
    if isinstance(tc, list) and tc:
        # "WOS" or "WOK" etc.
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
    )


def main() -> None:
    ap = argparse.ArgumentParser()

    # Sources
    ap.add_argument("--sources", default="pubmed", help="Comma-separated: pubmed,wos,both (e.g., 'pubmed', 'wos', 'both').")

    # PubMed
    ap.add_argument("--pubmed_query_file", default=None, help="Text file containing the PubMed query.")
    ap.add_argument("--email", default=None, help="Email for NCBI/Unpaywall (required for PubMed; recommended for Unpaywall).")
    ap.add_argument("--ncbi_api_key", default=None, help="NCBI API key (optional but recommended).")
    ap.add_argument("--pubmed_batch_size", type=int, default=200, help="PubMed EFetch batch size.")
    ap.add_argument("--pubmed_retmax", type=int, default=1000000, help="Maximum PubMed records to fetch.")

    # PDFs
    ap.add_argument("--download_pdfs", action="store_true", help="Download PDFs when available (PMC OA; optionally Unpaywall OA).")
    ap.add_argument("--use_unpaywall", action="store_true", help="Also try Unpaywall OA PDFs by DOI (requires --email).")

    # WoS
    ap.add_argument("--wos_query_file", default=None, help="Text file containing the WoS Advanced Search query (usrQuery).")
    ap.add_argument("--wos_api_key", default=os.environ.get("WOS_API_KEY"), help="Web of Science API key (or set env WOS_API_KEY).")
    ap.add_argument("--wos_database_id", default="WOS", help="WoS databaseId (commonly WOS or WOK depending on entitlement).")
    ap.add_argument("--wos_count", type=int, default=100, help="WoS page size (count).")
    ap.add_argument("--wos_option_view", default="FR", help="WoS optionView: FR (full) or SR (short).")
    ap.add_argument("--wos_base_url", default=WOS_EXPANDED_SEARCH, help="Override WoS base URL if needed.")

    # Output / rate limiting
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--sleep", type=float, default=0.34, help="Sleep seconds between requests (rate limiting).")
    ap.add_argument("--overwrite", action="store_true", help="If set, deletes previous output directory contents before writing new output.")
    ap.add_argument("--append", action="store_true", help="If set, appends to existing JSONL files (and skips existing IDs). Default overwrites JSONL unless --append is used.")

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

    if args.overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    _safe_mkdir(data_dir)
    _safe_mkdir(pmc_pdf_dir)
    _safe_mkdir(oa_pdf_dir)

    pubmed_path = os.path.join(data_dir, "pubmed_records.jsonl")
    wos_path = os.path.join(data_dir, "wos_records.jsonl")

    # Decide write modes
    pubmed_mode = "a" if args.append else "w"
    wos_mode = "a" if args.append else "w"

    session = _requests_session(user_agent="nicotine-coi-analysis/1.0 (academic research)")

    # ---------------- PubMed ----------------
    if want_pubmed:
        if not args.pubmed_query_file:
            raise SystemExit("--pubmed_query_file is required when sources include pubmed.")
        if not args.email:
            raise SystemExit("--email is required for PubMed (NCBI recommends a contact email).")

        query = read_query_text(args.pubmed_query_file)

        existing_pmids = _load_existing_ids(pubmed_path, "record_id") if args.append else set()

        count, webenv, query_key = pubmed_esearch(
            session=session,
            query=query,
            email=args.email,
            api_key=args.ncbi_api_key,
            retmax=args.pubmed_retmax,
        )
        total = min(count, args.pubmed_retmax)
        print(f"[PubMed] hits: {count} (will fetch {total})")

        fetched = 0
        with open(pubmed_path, pubmed_mode, encoding="utf-8") as out_f:
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
                    if rec.record_id in existing_pmids:
                        continue

                    # PDFs (best effort; OA only)
                    if args.download_pdfs:
                        # 1) PMC OA PDF
                        if rec.pmcid:
                            try:
                                pdf_url = pmc_oa_lookup_pdf_url(session, rec.pmcid, args.email, args.ncbi_api_key)
                                if pdf_url:
                                    rec.pmc_pdf_url = pdf_url
                                    pdf_path = os.path.join(pmc_pdf_dir, f"{rec.pmcid}.pdf")
                                    download_pdf(session, pdf_url, pdf_path, overwrite=args.overwrite)
                                    rec.pdf_path = pdf_path
                            except Exception as e:
                                print(f"[WARN][PubMed] PMC PDF failed for {rec.pmcid}: {e}")

                        # 2) Unpaywall OA PDF by DOI
                        if args.use_unpaywall:
                            if not args.email:
                                raise SystemExit("--email is required for Unpaywall.")
                            if rec.doi and (not rec.pdf_path):
                                try:
                                    up_pdf = unpaywall_best_pdf_url(session, rec.doi, email=args.email)
                                    if up_pdf:
                                        fname = _hash_to_filename(rec.doi) + ".pdf"
                                        pdf_path = os.path.join(oa_pdf_dir, fname)
                                        download_pdf(session, up_pdf, pdf_path, overwrite=args.overwrite)
                                        rec.pdf_path = pdf_path
                                except Exception as e:
                                    print(f"[WARN][PubMed] Unpaywall PDF failed for DOI={rec.doi}: {e}")

                    out_f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    existing_pmids.add(rec.record_id)
                    fetched += 1

                print(f"[PubMed] fetched {min(retstart + args.pubmed_batch_size, total)}/{total}")
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
        page_size = max(1, min(args.wos_count, 100))  # common max is 100
        total_found = None
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

1) PubMed only (metadata + PMC OA PDFs only):
  python fetch_data.py \
    --sources pubmed \
    --pubmed_query_file pubmed_query.txt \
    --out_dir ./study_dump \
    --email "you@domain.com" \
    --ncbi_api_key "<NCBI_API_KEY>" \
    --download_pdfs \
    --sleep 0.34

2) Web of Science only (metadata):
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
    --email "you@domain.com" \
    --ncbi_api_key "<NCBI_API_KEY>" \
    --download_pdfs \
    --sleep 0.34

4) Append (do not overwrite; skip IDs already in JSONL):
  python fetch_data.py ... --append

5) Overwrite everything (delete old outputs first):
  python fetch_data.py ... --overwrite

Optional: Unpaywall OA PDFs by DOI (best-effort):
  python fetch_data.py ... --download_pdfs --use_unpaywall --email "you@domain.com"
"""

