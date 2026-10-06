#!/usr/bin/env python3
import json, os, requests
from pathlib import Path

API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    print("BLOCKED: NVIDIA_API_KEY missing")
    raise SystemExit(2)

url = "https://integrate.api.nvidia.com/v1/chat/completions"
models = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
]
evidence = """
We are screening moisture-swing direct-air-capture sorbents. Published evidence used for this V0:
- double-quaternary anion-exchange resin (DQ-AER) has reported high CO2 uptake with carbonate counter-ion;
- phosphate-family counter-ions have shown strong moisture-swing behavior in other quaternary-ammonium resins;
- candidate hypotheses to challenge, not assume: DQ-AER+HPO4^2-, DQ-AER+PO4^3-, DQ-AER+P2O7^4-.
We need an independent, conservative hypothesis audit. Do not claim experimental validation.
"""
prompt = evidence + """
Return strict JSON with keys:
model_role, ranking (array of 3 objects with candidate, rationale, key_failure_mode, next_computation),
best_candidate, falsification_test, confidence_0_to_1.
Rank by physical plausibility and value of the next computational test, not hype.
"""

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
out = {"status":"BLOCKED","attempts":[]}
for model in models:
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":"You are a conservative computational-chemistry research auditor. Distinguish hypothesis from evidence."},
            {"role":"user","content":prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 1200,
        "response_format": {"type":"json_object"},
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=180)
        attempt = {"model":model,"http_status":r.status_code}
        if r.status_code >= 400:
            attempt["error"] = r.text[:1000]
            out["attempts"].append(attempt)
            continue
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        attempt["response"] = json.loads(content)
        out["attempts"].append(attempt)
        out["status"] = "PASS"
        out["selected_model"] = model
        out["audit"] = attempt["response"]
        break
    except Exception as e:
        out["attempts"].append({"model":model,"error":repr(e)})

Path("03-validation").mkdir(exist_ok=True)
Path("03-validation/nvidia-moisture-swing-audit-v0.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
summary = ["# NVIDIA Moisture-Swing Audit V0", "", f"Status: **{out['status']}**", ""]
if out.get("selected_model"):
    summary += [f"Model: \`{out['selected_model']}\`", "", "This is an NVIDIA NIM reasoning audit, not atomistic or experimental validation.", "", "## Audit", "", "\`\`\`json", json.dumps(out["audit"], indent=2), "\`\`\`"]
else:
    summary += ["No NVIDIA reasoning endpoint succeeded.", "", "\`\`\`json", json.dumps(out["attempts"], indent=2), "\`\`\`"]
Path("03-validation/nvidia-moisture-swing-audit-v0.md").write_text("\n".join(summary), encoding="utf-8")
print("\n".join(summary))
raise SystemExit(0 if out["status"]=="PASS" else 2)
