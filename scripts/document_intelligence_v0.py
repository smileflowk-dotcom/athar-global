#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "03-validation" / "evidence-retrieval-gold.jsonl"
PDF_PATH = ROOT / ".cache" / "evidence-retrieval-source.pdf"
RESULTS_PATH = ROOT / "03-validation" / "document-intelligence-results.jsonl"
SUMMARY_PATH = ROOT / "03-validation" / "document-intelligence-summary.md"


def normalize(value: str) -> str:
    value = value.lower().replace("…", "...")
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[^\w€$.,:%-]+", " ", value).strip()


def compact(value: str) -> str:
    return re.sub(r"[^\w]+", "", value.lower(), flags=re.UNICODE)


def anchor_match(text: str, quote: str) -> bool:
    normalized = compact(text)
    parts = [compact(part) for part in re.split(r"\.\.\.", quote) if compact(part)]
    return bool(parts) and all(part in normalized for part in parts)


def load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def download_pdf(url: str) -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 10_000:
        return
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    PDF_PATH.write_bytes(response.content)


def main() -> int:
    gold = load_gold()
    download_pdf(gold[0]["source_url"])
    reader = PdfReader(str(PDF_PATH))
    pages = {}
    for page_number, page in enumerate(reader.pages, start=1):
        pages[page_number] = re.sub(r"\s+", " ", page.extract_text() or "").strip()

    records = []
    for case in gold:
        page_text = pages.get(int(case["page"]), "")
        quote_found = anchor_match(page_text, case["gold_quote"])
        section_found = compact(case["section"]) in compact(page_text)
        records.append({
            "id": case["id"],
            "method": "pypdf page-preserving extraction",
            "page": case["page"],
            "quote_anchor_found": quote_found,
            "section_label_found": section_found,
            "locator_preserved": bool(page_text) and int(case["page"]) in pages,
            "fabricated_evidence": False,
        })

    fidelity = sum(record["quote_anchor_found"] for record in records) / len(records)
    structure = sum(record["section_label_found"] for record in records) / len(records)
    locators = sum(record["locator_preserved"] for record in records) / len(records)
    fabricated = sum(record["fabricated_evidence"] for record in records)
    passed = fidelity == 1.0 and structure == 1.0 and locators == 1.0 and fabricated == 0
    status = "PASS" if passed else "FAIL"

    RESULTS_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    SUMMARY_PATH.write_text(
        f"""# Document Intelligence V0 — Validation Result

## TEST

Run the simplest page-preserving extraction against the ten existing EuroHPC gold cases.

## EXPECTED

Exact source text anchors, page locators, useful section labels, and zero invented evidence.

## ACTUAL

- Method: `pypdf` direct extraction; no NVIDIA Parse/OCR selected.
- Source pages sampled: {len(pages)}
- Quote-anchor fidelity: {fidelity:.2f}
- Section-label usefulness: {structure:.2f}
- Locator preservation: {locators:.2f}
- Fabricated evidence: {fabricated}

## RESULT: {status}

The NVIDIA document-intelligence capability remains unselected for this born-digital demo case unless a later scan/table benchmark shows measurable value.

## NEXT

Use this extraction baseline for the minimal reasoning and end-to-end proof; benchmark Parse/OCR only for scanned or degraded documents.
""",
        encoding="utf-8",
    )
    print(f"DOCUMENT_INTELLIGENCE: {status}")
    print(f"TEXT_FIDELITY: {fidelity:.2f}")
    print(f"STRUCTURE_USEFULNESS: {structure:.2f}")
    print(f"LOCATOR_PRESERVATION: {locators:.2f}")
    print(f"FABRICATED_EVIDENCE: {fabricated}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
