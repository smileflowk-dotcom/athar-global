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

ICM intelligence mapping

## CURRENT NEXT

Run Evidence Retrieval V0 on the fixed 10-case EuroHPC gold set: source PDF → page-preserving chunks → Nemotron embedding → rerank → metrics. No UI or reasoning layer until this earns PASS.
