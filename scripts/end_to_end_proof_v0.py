#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evidence_retrieval_v0 import EMBED_MODEL, RERANK_MODEL, chunk_pages, download_pdf, embed_texts, extract_pages, load_gold, rerank  # noqa: E402
from evidence_reasoning_v0 import chat, parse_json, resolve_chat_model  # noqa: E402

RESULTS_PATH = ROOT / "03-validation" / "end-to-end-proof-results.jsonl"
SUMMARY_PATH = ROOT / "03-validation" / "end-to-end-proof-summary.md"


def anchor_in(text: str, quote: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower())
    parts = [re.sub(r"\s+", " ", p.lower()).strip() for p in re.split(r"\.\.\.|…", quote) if p.strip()]
    return bool(parts) and all(part in normalized for part in parts)


def main() -> int:
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    nebius_key = os.getenv("NEBIUS_API_KEY")
    if not nvidia_key:
        print("BLOCKED: NVIDIA_API_KEY missing")
        return 2
    if not nebius_key:
        print("BLOCKED: NEBIUS_API_KEY missing")
        return 2
    gold = {row["id"]: row for row in load_gold()}
    selected = [gold["EUROHPC-001"], gold["EUROHPC-003"], gold["EUROHPC-007"]]
    model_ids = []
    from evidence_reasoning_v0 import get_models  # noqa: E402
    model_ids = get_models(nebius_key)
    chat_model = resolve_chat_model(model_ids)
    if not chat_model:
        print("BLOCKED: no NVIDIA Nemotron chat/reasoning model available")
        return 2
    embed_model = EMBED_MODEL
    rerank_model = RERANK_MODEL

    download_pdf(selected[0]["source_url"])
    pages = extract_pages()
    chunks = chunk_pages(pages, selected[0]["source_url"], selected[0]["document"])
    corpus_vectors, _ = embed_texts([chunk["text"] for chunk in chunks], "passage", nvidia_key)
    records = []
    for case in selected:
        query_vector, _ = embed_texts([case["query"]], "query", nvidia_key)
        scores = corpus_vectors @ query_vector[0]
        top5 = []
        for index in np.argsort(-scores)[:5]:
            item = dict(chunks[int(index)])
            item["embedding_score"] = float(scores[int(index)])
            top5.append(item)
        reranked, _ = rerank(case["query"], top5, nvidia_key)
        evidence = f"CANDIDATE 1 | page {reranked[0]['page']}\n{reranked[0]['text']}"
        requirement = {
            "EUROHPC-001": "The maximum total amount available under this call for tenders is EUR 80,000,000.00.",
            "EUROHPC-003": "Variants to the proposed solution are allowed.",
            "EUROHPC-007": "The tenderer must indicate its country of establishment and provide acceptable supporting evidence.",
        }[case["id"]]
        raw, request_id = chat(nebius_key, chat_model, requirement, evidence, case["document"], case["page"])
        parsed = {}
        error = ""
        try:
            parsed = parse_json(raw)
        except Exception as exc:
            error = str(exc)
        evidence_text = str(parsed.get("evidence", ""))
        selected_source = any(anchor_in(item["text"], evidence_text) for item in reranked[:3]) if evidence_text else False
        locator_ok = any(str(item["page"]) in str(parsed.get("source", "")) for item in reranked[:3])
        record = {
            "id": case["id"],
            "document_extraction": {"method": "pypdf", "pages": sorted({item["page"] for item in chunks})},
            "retrieval": {"embed_model": embed_model, "rerank_model": rerank_model, "top5": top5, "reranked_top3": reranked[:3]},
            "reasoning": {"model": chat_model, "input_evidence": evidence, "raw_response": raw, "parsed": parsed, "request_id": request_id},
            "validation": {"evidence_grounded_in_retrieved_text": selected_source, "locator_preserved": locator_ok, "parse_error": error},
        }
        records.append(record)

    passed = all(record["validation"]["evidence_grounded_in_retrieved_text"] and record["validation"]["locator_preserved"] and not record["validation"]["parse_error"] for record in records)
    status = "PASS" if passed else "FAIL"
    RESULTS_PATH.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    SUMMARY_PATH.write_text(
        f"""# Minimal End-to-End Proof V0

## TRACE

`procurement PDF → pypdf page extraction → NVIDIA embedding → NVIDIA rerank → NVIDIA Nemotron reasoning → finding + evidence + page`

## MODELS

- Embed: `{embed_model}`
- Rerank: `{rerank_model}`
- Reasoning: `{chat_model}`

## RESULT: {status}

- Findings tested: {len(records)}
- Evidence grounded in retrieved text: {sum(r['validation']['evidence_grounded_in_retrieved_text'] for r in records)}/{len(records)}
- Locator preserved: {sum(r['validation']['locator_preserved'] for r in records)}/{len(records)}
- Parse errors: {sum(bool(r['validation']['parse_error']) for r in records)}

## NEXT

Connect the validated bricks behind a human-reviewed product flow.
""",
        encoding="utf-8",
    )
    print(f"END_TO_END: {status}")
    print(f"FINDINGS_TESTED: {len(records)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
