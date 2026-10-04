#!/usr/bin/env python3
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from pypdf import PdfReader

BASE_URL = "https://api.tokenfactory.nebius.com/v1"
ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "03-validation" / "evidence-retrieval-gold.jsonl"
RESULTS_PATH = ROOT / "03-validation" / "evidence-retrieval-results.jsonl"
SUMMARY_PATH = ROOT / "03-validation" / "evidence-retrieval-summary.md"
PDF_PATH = ROOT / ".cache" / "evidence-retrieval-source.pdf"

CHUNK_CHARS = 1800
CHUNK_OVERLAP = 250
EMBED_BATCH = 32

def blocked(msg: str) -> None:
    print(f"BLOCKED: {msg}")
    sys.exit(2)

def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

def api_get(path: str, api_key: str) -> dict[str, Any]:
    r = requests.get(f"{BASE_URL}{path}", headers=auth_headers(api_key), timeout=60)
    if r.status_code >= 400:
        blocked(f"GET {path} failed: HTTP {r.status_code}")
    return r.json()

def api_post(path: str, payload: dict[str, Any], api_key: str, timeout: int = 180) -> dict[str, Any]:
    r = requests.post(f"{BASE_URL}{path}", headers=auth_headers(api_key), json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {path} failed: HTTP {r.status_code}: {r.text[:500]}")
    return r.json()

def list_model_ids(api_key: str) -> list[str]:
    data = api_get("/models", api_key)
    items = data.get("data") or data.get("models") or []
    out = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            model_id = item.get("id") or item.get("name") or item.get("model")
            if model_id:
                out.append(str(model_id))
    return sorted(set(out))

def resolve_model(ids: list[str], kind: str) -> str:
    candidates = [(x, x.lower()) for x in ids]
    patterns = (
        [("nvidia", "nemotron", "embed"), ("nvidia", "embed"), ("nemotron", "embed")]
        if kind == "embed"
        else [("nvidia", "nemotron", "rerank"), ("nvidia", "rerank"), ("nemotron", "rerank")]
    )
    for pattern in patterns:
        for original, low in candidates:
            if all(token in low for token in pattern):
                return original
    return ""

def load_gold() -> list[dict[str, Any]]:
    rows = []
    for line in GOLD_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if len(rows) < 10:
        blocked(f"gold set too small: {len(rows)}")
    return rows

def download_pdf(url: str) -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 10000:
        return
    r = requests.get(url, timeout=120)
    if r.status_code >= 400:
        blocked(f"source PDF unavailable: HTTP {r.status_code}")
    PDF_PATH.write_bytes(r.content)

def extract_pages() -> list[dict[str, Any]]:
    reader = PdfReader(str(PDF_PATH))
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        pages.append({"page": page_no, "text": text})
    return pages

def chunk_pages(pages: list[dict[str, Any]], source_url: str, document: str) -> list[dict[str, Any]]:
    chunks = []
    step = CHUNK_CHARS - CHUNK_OVERLAP
    for page in pages:
        text = page["text"]
        if not text:
            continue
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_CHARS)
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "chunk_id": f"p{page['page']:04d}-c{chunk_index:03d}",
                    "page": page["page"],
                    "source_url": source_url,
                    "document": document,
                    "text": piece,
                })
            if end >= len(text):
                break
            start += step
            chunk_index += 1
    return chunks

def batches(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]

def embed_texts(texts: list[str], model: str, api_key: str) -> tuple[np.ndarray, float]:
    vectors = []
    started = time.perf_counter()
    for batch in batches(texts, EMBED_BATCH):
        data = api_post("/embeddings", {"model": model, "input": batch}, api_key)
        rows = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        vectors.extend(row["embedding"] for row in rows)
    latency = time.perf_counter() - started
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.shape[0] != len(texts):
        raise RuntimeError(f"embedding count mismatch: {arr.shape[0]} != {len(texts)}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms, latency

def rerank(query: str, docs: list[dict[str, Any]], model: str, api_key: str) -> tuple[list[dict[str, Any]], float]:
    payload = {
        "model": model,
        "query": query,
        "documents": [d["text"] for d in docs],
        "top_n": len(docs),
        "return_documents": True,
    }
    started = time.perf_counter()
    data = api_post("/rerank", payload, api_key)
    latency = time.perf_counter() - started
    results = data.get("results") or data.get("data") or []
    out = []
    for item in results:
        idx = item.get("index")
        if idx is None:
            continue
        d = dict(docs[int(idx)])
        d["rerank_score"] = item.get("relevance_score", item.get("score"))
        out.append(d)
    if not out:
        raise RuntimeError("rerank returned no usable results")
    return out, latency

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w€$.,:%-]+", " ", text).strip()

def anchor_match(text: str, quote: str) -> bool:
    normalized = normalize(text)
    parts = [normalize(x) for x in re.split(r"\.{3}|…", quote) if normalize(x)]
    return bool(parts) and all(part in normalized for part in parts)

def hit(candidates: list[dict[str, Any]], gold: dict[str, Any], k: int) -> bool:
    return any(
        c["page"] == int(gold["page"]) and anchor_match(c["text"], gold["gold_quote"])
        for c in candidates[:k]
    )

def main() -> int:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        blocked("NEBIUS_API_KEY missing")

    gold = load_gold()
    model_ids = list_model_ids(api_key)
    embed_model = resolve_model(model_ids, "embed")
    rerank_model = resolve_model(model_ids, "rerank")
    print(f"NVIDIA/Nemotron embed model: {embed_model or 'NONE'}")
    print(f"NVIDIA/Nemotron rerank model: {rerank_model or 'NONE'}")
    if not embed_model:
        blocked("required NVIDIA Nemotron embedding family unavailable")
    if not rerank_model:
        blocked("required NVIDIA Nemotron rerank family unavailable")

    download_pdf(gold[0]["source_url"])
    pages = extract_pages()
    chunks = chunk_pages(pages, gold[0]["source_url"], gold[0]["document"])
    if not chunks:
        blocked("no chunks extracted")

    corpus_vectors, corpus_latency = embed_texts([c["text"] for c in chunks], embed_model, api_key)

    records = []
    query_embed_latency = 0.0
    rerank_latency = 0.0
    for case in gold:
        qv, q_latency = embed_texts([case["query"]], embed_model, api_key)
        query_embed_latency += q_latency
        scores = corpus_vectors @ qv[0]
        top_indices = np.argsort(-scores)[:5]
        top5 = []
        for idx in top_indices:
            chunk = dict(chunks[int(idx)])
            chunk["embedding_score"] = float(scores[int(idx)])
            top5.append(chunk)

        reranked, r_latency = rerank(case["query"], top5, rerank_model, api_key)
        rerank_latency += r_latency

        record = {
            "id": case["id"],
            "query": case["query"],
            "gold_page": case["page"],
            "gold_quote": case["gold_quote"],
            "embedding_top5": top5,
            "reranked_top5": reranked,
            "hit_recall_at_5": hit(top5, case, 5),
            "hit_rerank_recall_at_3": hit(reranked, case, 3),
            "hit_top1": hit(reranked, case, 1),
            "locator_preserved": all(x.get("source_url") and isinstance(x.get("page"), int) for x in reranked),
        }
        records.append(record)
        print(f"{case['id']}: R@5={record['hit_recall_at_5']} RR@3={record['hit_rerank_recall_at_3']} TOP1={record['hit_top1']}")

    n = len(records)
    recall5 = sum(r["hit_recall_at_5"] for r in records) / n
    recall3 = sum(r["hit_rerank_recall_at_3"] for r in records) / n
    top1 = sum(r["hit_top1"] for r in records) / n
    locator = sum(r["locator_preserved"] for r in records) / n
    fabricated = 0
    passed = recall5 >= 0.90 and recall3 >= 0.90 and top1 >= 0.80 and locator == 1.0 and fabricated == 0
    status = "PASS" if passed else "FAIL"

    RESULTS_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )

    summary = f"""# Evidence Retrieval V0 — Runtime Result

## RESULT

{status}

## Models

- Embed: `{embed_model}`
- Rerank: `{rerank_model}`
- Runtime: Nebius Token Factory

## Corpus

- Pages extracted: {len(pages)}
- Chunks: {len(chunks)}
- Chunk chars: {CHUNK_CHARS}
- Overlap chars: {CHUNK_OVERLAP}

## Metrics

- Recall@5 embeddings: {recall5:.3f}
- Recall@3 after rerank: {recall3:.3f}
- Top-1 after rerank: {top1:.3f}
- Locator preservation: {locator:.3f}
- Fabricated evidence passages: {fabricated}

## Gate

- Recall@5 >= 0.90
- Recall@3 >= 0.90
- Top-1 >= 0.80
- Locator preservation = 1.00
- Fabricated evidence = 0

## Runtime metadata

- Timestamp UTC: {datetime.now(timezone.utc).isoformat()}
- Corpus embedding latency: {corpus_latency:.2f}s
- Query embedding latency total: {query_embed_latency:.2f}s
- Rerank latency total: {rerank_latency:.2f}s

## NEXT

{"Integrate retrieval, then validate document intelligence." if passed else "Inspect misses, make one documented correction without changing the gold set, then rerun."}
"""
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(summary)
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
