#!/usr/bin/env python3
"""Bounded, citation-only Crossref literature radar for Chemistry Discovery V4.

Never fetch or redistribute copyrighted full text. Search matches and DOI
metadata do NOT imply exact-pair experimental evidence, novelty or safety.
Only separately reviewed curated references can support scientific Gate D.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

CURATED = Path(__file__).resolve().parent / "literature" / "curated_evidence.json"
DEFAULT_OUT = Path("chemistry/v4/output/literature.json")
AS_OF = dt.datetime.now(ZoneInfo('Europe/Paris')).date()
QUERIES = [
    "moisture swing direct air capture anion exchange phosphate carbonate",
    "pyrophosphate moisture swing IRA900 CO2",
    "strong base resin pore connectivity phosphate carbonate moisture swing",
    "active learning moisture swing adsorption direct air capture",
    "moisture swing resin borate sulfite oxalate citrate",
    "moisture-swing direct air capture adsorption kinetic cycling stability",
]
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.I)


def load_curated(path=CURATED):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported curated-evidence schema")
    seen = set()
    for ref in data["references"]:
        if not ref.get("id") or not ref.get("title") or not ref.get("url"):
            raise ValueError("Curated reference missing identity/title/URL")
        if ref["id"] in seen:
            raise ValueError("Duplicate curated reference ID")
        seen.add(ref["id"])
        if ref.get("doi") and not DOI_PATTERN.match(ref["doi"]):
            raise ValueError("Malformed DOI")
        if ref.get("review_state") not in {"CURATED_PRIMARY", "PREVIOUS_AUDIT_PENDING_REFRESH"}:
            raise ValueError("Unrecognized human review state")
    return data


def publication_date(work):
    # Prefer first actual publication, NOT last DOI metadata update.
    for field in ("published-online", "published-print", "published", "issued"):
        parts = work.get(field, {}).get("date-parts", [])
        if parts and parts[0] and isinstance(parts[0][0], int):
            t = (parts[0] + [1, 1])[:3]
            try:
                return dt.date(*t).isoformat()
            except ValueError:
                continue
    return None


def normalize_item(work, as_of):
    doi = str(work.get("DOI", "")).strip().lower()
    title = work.get("title", [])
    title = str(title[0]).strip() if title else ""
    if not DOI_PATTERN.match(doi) or not title:
        return None
    published = publication_date(work)
    if published and published > as_of.isoformat():
        return None
    if work.get("type") not in {"journal-article", "posted-content", "proceedings-article"}:
        return None
    return {
        "doi": doi, "title": title, "published": published,
        "url": "https://doi.org/" + doi,
        "publisher": str(work.get("publisher") or ""),
        "type": str(work.get("type") or ""),
        "metadata_source": "Crossref REST works public API",
        "review_state": "NEW_METADATA_UNREVIEWED",
        "specificity": "UNASSESSED_BY_HUMAN",
        "copyright_policy": "Citation metadata only; no full text retained",
        "metadata_missing_date": published is None
    }


def fetch_json(url, *, timeout=12):
    req = urllib.request.Request(
        url, headers={"User-Agent": "ATHAR-V4-Literature/1.0 (public-metadata-only)",
                      "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as stream:
        return json.load(stream)


def scan(offline=False, *, per_query=6, as_of=AS_OF, client=fetch_json):
    curated = load_curated()
    known = {r["doi"].lower() for r in curated["references"] if r.get("doi")}
    found = {}
    errors = []
    counts = {}
    if not offline:
        for query in QUERIES:
            params = urllib.parse.urlencode({
                "query.bibliographic": query, "rows": per_query,
                "filter": "from-pub-date:2018-01-01,until-pub-date:" + as_of.isoformat(),
                "select": "DOI,title,type,publisher,published,published-online,published-print,issued",
            })
            url = "https://api.crossref.org/works?" + params
            try:
                page = client(url)
                items = page.get("message", {}).get("items", [])
                if not isinstance(items, list):
                    raise ValueError("Crossref metadata structure changed")
                counts[query] = len(items)
                for raw in items:
                    item = normalize_item(raw, as_of)
                    if not item:
                        continue
                    if item["doi"] in known:
                        continue
                    item["matched_queries"] = [query]
                    if item["doi"] not in found:
                        found[item["doi"]] = item
                    elif query not in found[item["doi"]]["matched_queries"]:
                        found[item["doi"]]["matched_queries"].append(query)
            except (urllib.error.URLError, ValueError, TimeoutError, OSError,
                    json.JSONDecodeError) as exc:
                errors.append({"query": query, "kind": type(exc).__name__})
    return {
        "schema_version": 1, "as_of": as_of.isoformat(),
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "OFFLINE_CURATED_ONLY" if offline else (
            "LIVE_OK" if not errors else "DEGRADED_PARTIAL" if counts else "DEGRADED_CURATED_ONLY"),
        "source": "Crossref public bibliographic metadata; existing publisher-reviewed registry",
        "queries": QUERIES, "query_result_counts": counts,
        "curated": curated["references"],
        "new_metadata_to_review": sorted(found.values(),
                                         key=lambda x: (x["published"] or "", x["doi"]),
                                         reverse=True),
        "errors": errors, "policy": {
            "no_auto_promotion_from_search": True,
            "no_novelty_claim_from_search_absence": True,
            "no_full_text_caching": True,
            "exact_pair_evidence_requires_primary_source_review": True,
            "publisher_datasheet_not_adsorption_proof": True,
            "license_of_underlying_training_dataset_requires_separate_review": True,
        },
    }


def evidence_for_candidate(resin, ion, registry):
    key = resin + "__" + ion
    exact, context = [], []
    for record in registry.get("curated", []):
        if record.get("review_state") != "CURATED_PRIMARY":
            continue
        if key in record.get("candidate_ids", []):
            exact.append({"id": record["id"], "url": record["url"],
                          "scope": record["scope"], "doi": record.get("doi")})
        elif (resin in record.get("resins", []) or
              ion in record.get("ions", []) or
              record.get("scope") in {"family_mechanism", "dataset_and_method",
                                       "phosphate_family_prior_art"}):
            context.append({"id": record["id"], "url": record["url"],
                            "scope": record["scope"], "doi": record.get("doi")})
    return {
        "candidate": key, "exact_curated": exact, "related_context": context,
        "new_unreviewed_metadata_count": len(registry.get("new_metadata_to_review", [])),
        "scan_status": registry.get("status", "UNKNOWN"),
        "novelty_claim": False,
        "reason": "Metadata hits cannot establish experimental result for a specific pair",
    }


def write_report(data, filename):
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    md = out.with_name("literature.md")
    lines = ["# V4 literature audit", "",
             "**Only references in the manually curated registry are treated as reviewed.**",
             "DOI searches never automatically create scientific PASS or novelty claims.", "",
             "Search status: " + data["status"],
             "As of: " + data["as_of"], "",
             "## Curated bibliography"]
    for item in data["curated"]:
        lines.append(f"- [{item['id']}]({item['url']}) — {item['scope']} / {item['review_state']}")
    lines += ["", "## Newly found DOI metadata (manual review queue)"]
    for item in data["new_metadata_to_review"]:
        title = item["title"].replace("\n", " ").replace("|", "/")
        lines.append(f"- [{title}]({item['url']}) ({item['published'] or 'date unknown'}) — UNREVIEWED")
    lines += ["", "## Retrieval failures (do not infer absence of evidence)"]
    lines.extend(f"- {e['query']}: {e['kind']}" for e in data["errors"])
    md.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="Deterministic seeded run, no external network")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--rows", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.rows <= 12:
        parser.error("--rows must be 1..12 for a bounded public-API query")
    data = scan(args.offline, per_query=args.rows)
    write_report(data, args.output)
    print("LITERATURE", data["status"], "CURATED", len(data["curated"]),
          "REVIEW_QUEUE", len(data["new_metadata_to_review"]),
          "FAILED_QUERIES", len(data["errors"]), flush=True)


if __name__ == "__main__":
    main()
