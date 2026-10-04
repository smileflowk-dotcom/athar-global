# CONTEXT

## PROJECT

ATHAR Global

## CORE PROMISE

Every finding leads back to the evidence.

## HACKATHON

Nebius × NVIDIA Global AI Hackathon

## TARGET

Global public procurement review

## PRODUCT PRINCIPLE

Upload procurement dossier
→ analyze
→ surface findings
→ show exact evidence
→ human validates

## TECH DIRECTION

Use as many NVIDIA open-model capabilities as are genuinely useful, including candidates such as:

- NVIDIA document parsing
- OCR
- embeddings
- reranking
- Nemotron reasoning
- multimodal models
- agent observability / evaluation

Nebius should be the primary runtime/inference platform where appropriate.

IMPORTANT: These are candidates, not confirmed architecture decisions. Do not mark any NVIDIA component as selected until it is validated.

## CONSTRAINTS

- global, not country-specific
- human remains final decision-maker
- no automatic accusation of fraud, illegality, or wrongdoing
- evidence-first
- traceable outputs
- simple UX
- hackathon demo must be understandable in under 3 minutes
- no unnecessary complexity

## CURRENT PHASE

Scientific workflow pivot — execution blocked

## CURRENT NEXT

Provide a verified NVIDIA GPU execution path, then run the fixed-seed MNIST dropout comparison.

## SCIENTIFIC PIVOT STATUS

- Branch: `hackathon/scientific-workflow-pivot`
- Paper: *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*
- GPU runtime: unavailable in the current Codex environment
- PyTorch: unavailable
- `nvidia-smi`: unavailable
- Existing Actions path: `ubuntu-latest` CPU/API workflows only
- Experiment: not run
- Metrics: not available
- NVIDIA execution was not substituted or simulated

## VERIFIED PROVE STATE

Evidence Retrieval V0: PASS on the fixed 10-case EuroHPC gold set.

- NVIDIA embedding: `nvidia/llama-nemotron-embed-vl-1b-v2`
- NVIDIA reranking: `nvidia/llama-nemotron-rerank-vl-1b-v2`
- Recall@5: 1.00
- Recall@3 after rerank: 1.00
- Top-1: 0.80
- Locator preservation: 1.00
- Fabricated evidence: 0

Document Intelligence V0: PASS for the born-digital demo PDF using the simple `pypdf` baseline.

- Text fidelity: 1.00
- Structure usefulness: 0.80
- Locator preservation: 1.00
- Fabricated evidence: 0
- NVIDIA Parse/OCR: not selected; benchmark later for scanned or degraded documents.

Evidence Reasoning V0: PASS through Nebius Token Factory.

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`
- Accuracy: 4/5 (0.80)
- Locator preservation: 5/5
- Fabricated evidence: 0
- Unsupported legal conclusions: 0

Minimal end-to-end proof: PASS for 3 representative findings.

The human remains the final decision-maker. No UI, autonomous legal conclusion, fraud accusation, or country-specific rule engine was added.
