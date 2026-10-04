#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path

import requests

BASE_URL = "https://api.tokenfactory.nebius.com/v1"
ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "03-validation" / "evidence-reasoning-gold.jsonl"
RESULTS_PATH = ROOT / "03-validation" / "evidence-reasoning-results.jsonl"
SUMMARY_PATH = ROOT / "03-validation" / "evidence-reasoning-summary.md"


def blocked(message: str) -> None:
    print(f"BLOCKED: {message}")
    raise SystemExit(2)


def headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def get_models(api_key: str) -> list[str]:
    response = requests.get(f"{BASE_URL}/models", headers=headers(api_key), timeout=60)
    if response.status_code >= 400:
        blocked(f"GET /models failed: HTTP {response.status_code}")
    data = response.json()
    items = data.get("data") or data.get("models") or []
    model_ids = []
    for item in items:
        if isinstance(item, str):
            model_ids.append(item)
        elif isinstance(item, dict):
            model_id = item.get("id") or item.get("name") or item.get("model")
            if model_id:
                model_ids.append(str(model_id))
    return sorted(set(model_ids))


def resolve_chat_model(model_ids: list[str]) -> str:
    candidates = []
    for model_id in model_ids:
        low = model_id.lower()
        if "nvidia" not in low or "nemotron" not in low:
            continue
        if any(token in low for token in ("embed", "rerank", "ocr", "parse")):
            continue
        size_rank = 0 if "nano" in low else 1 if "mini" in low or "small" in low else 2
        candidates.append((size_rank, len(model_id), model_id))
    return sorted(candidates)[0][2] if candidates else ""


def chat(api_key: str, model: str, requirement: str, evidence: str, document: str, page: int) -> tuple[str, str]:
    system = (
        "You are an evidence-grounded procurement review classifier. "
        "Use only the supplied requirement and exact evidence. Never infer facts outside it. "
        "Do not make claims about fraud, illegality, guilt, or final legal compliance. "
        "Return one JSON object only with exactly these keys: status, finding, evidence, source, reason. "
        "status must be SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE. "
        "Copy the evidence value exactly. Source must include the document title and page number. "
        "Reason must be at most two sentences."
    )
    user = json.dumps({
        "requirement": requirement,
        "exact_evidence": evidence,
        "document": document,
        "page": page,
    }, ensure_ascii=False)
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 300,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    response = requests.post(f"{BASE_URL}/chat/completions", headers=headers(api_key), json=payload, timeout=180)
    if response.status_code >= 400:
        blocked(f"chat completion failed: HTTP {response.status_code}")
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content), response.headers.get("x-request-id", "")


def parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response did not contain a JSON object")
    return json.loads(content[start:end + 1])


def main() -> int:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        blocked("NEBIUS_API_KEY missing")
    gold = [json.loads(line) for line in GOLD_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    model = resolve_chat_model(get_models(api_key))
    if not model:
        blocked("no NVIDIA Nemotron chat/reasoning model available")

    records = []
    for case in gold:
        raw, request_id = chat(api_key, model, case["requirement"], case["evidence"], case["document"], case["page"])
        validation = {"status_correct": False, "locator_preserved": False, "fabricated_evidence": True, "unsupported_legal_conclusion": True}
        parsed = {}
        error = ""
        try:
            parsed = parse_json(raw)
            status = parsed.get("status")
            source = str(parsed.get("source", ""))
            reason_text = str(parsed.get("reason", ""))
            forbidden = re.search(r"\b(fraud|fraudulent|illegal|illegality|guilt|legally compliant|legal compliance)\b", f"{parsed.get('finding', '')} {reason_text}", re.IGNORECASE)
            validation = {
                "status_correct": status == case["expected_status"],
                "locator_preserved": case["document"] in source and re.search(rf"\bpage\s*{case['page']}\b", source, re.IGNORECASE) is not None,
                "fabricated_evidence": parsed.get("evidence") != case["evidence"],
                "unsupported_legal_conclusion": forbidden is not None,
            }
        except Exception as exc:
            error = str(exc)
        records.append({"id": case["id"], "expected_status": case["expected_status"], "model": model, "raw_response": raw, "parsed": parsed, "validation": validation, "error": error, "request_id": request_id})

    accuracy = sum(r["validation"]["status_correct"] for r in records) / len(records)
    locators = sum(r["validation"]["locator_preserved"] for r in records)
    fabricated = sum(r["validation"]["fabricated_evidence"] for r in records)
    unsupported = sum(r["validation"]["unsupported_legal_conclusion"] for r in records)
    passed = accuracy >= 0.8 and locators == 5 and fabricated == 0 and unsupported == 0
    status = "PASS" if passed else "FAIL"

    RESULTS_PATH.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    SUMMARY_PATH.write_text(
        f"""# Evidence Reasoning V0 — Validation Result

## TEST

Classify five fixed requirement/evidence pairs with a NVIDIA Nemotron chat model through Nebius Token Factory.

## EXPECTED

At least four of five statuses correct, all five document/page locators preserved, zero fabricated evidence, and zero unsupported legal conclusions.

## ACTUAL

- Model: `{model}`
- Endpoint: Nebius Token Factory `/v1/chat/completions`
- Status accuracy: {accuracy:.2f} ({sum(r['validation']['status_correct'] for r in records)}/5)
- Locator preservation: {locators}/5
- Fabricated evidence: {fabricated}
- Unsupported legal conclusions: {unsupported}

## RESULT: {status}

## NEXT

{"Run the minimal three-finding end-to-end proof." if passed else "Do not proceed to end-to-end integration; inspect the constrained output failures."}
""",
        encoding="utf-8",
    )
    print(f"EVIDENCE_REASONING: {status}")
    print(f"MODEL: {model}")
    print(f"ACCURACY: {accuracy:.2f}")
    print(f"LOCATOR_PRESERVATION: {locators}/5")
    print(f"FABRICATED_EVIDENCE: {fabricated}")
    print(f"UNSUPPORTED_LEGAL_CONCLUSIONS: {unsupported}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
