# Minimal End-to-End Proof V0

## TRACE

`procurement PDF → pypdf page extraction → NVIDIA embedding → NVIDIA rerank → NVIDIA Nemotron reasoning → finding + evidence + page`

## MODELS

- Embed: `nvidia/llama-nemotron-embed-vl-1b-v2`
- Rerank: `nvidia/llama-nemotron-rerank-vl-1b-v2`
- Reasoning: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`

## RESULT: FAIL

- Findings tested: 3
- Evidence grounded in retrieved text: 2/3
- Locator preserved: 3/3
- Parse errors: 0

## NEXT

Connect the validated bricks behind a human-reviewed product flow.
