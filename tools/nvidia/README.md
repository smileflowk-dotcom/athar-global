# NVIDIA Tools — ATHAR Global

Purpose: authoritative routing map for NVIDIA capabilities used by ATHAR Global.

## Current tool families

### Document intelligence
- Nemotron Parse
- Nemotron OCR

Use for:
- born-digital document structure
- scanned pages
- tables
- layout-aware extraction

Do not use both automatically. Benchmark routing by document type.

### Retrieval
- Nemotron Embed
- Nemotron Rerank

Use for:
- candidate passage retrieval
- evidence ranking
- multilingual / multimodal retrieval experiments

### Reasoning
- Nemotron reasoning models

Use for:
- rule vs observed fact comparison
- evidence-first explanation
- insufficient-evidence classification

Never use reasoning output as the final legal or fraud decision.

### Multimodal
- Nemotron Nano Omni candidate

Use only if ATHAR must process non-text evidence such as image/audio/video attachments and the benchmark shows added value.

### Agent evaluation / observability
- NVIDIA NeMo / NeMo Agent Toolkit / NeMo Platform evaluation tooling

Use for:
- agent traces
- evaluation
- optimization
- retrieval evaluation where useful

Do not introduce this before the core evidence pipeline is working.

## Selection rule

A NVIDIA tool is integrated only if it passes:

1. measurable value;
2. simple role in architecture;
3. explainable contribution in the hackathon demo;
4. no unnecessary duplication.

## Official sources

- https://build.nvidia.com/models?q=nemotron
- https://build.nvidia.com/nvidia/nemotron-parse
- https://build.nvidia.com/nvidia/nemotron-ocr-v2
- https://build.nvidia.com/explore/retrieval
- https://docs.nvidia.com/nemo/
