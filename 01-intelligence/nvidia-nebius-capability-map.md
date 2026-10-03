# NVIDIA + Nebius Capability Map — ATHAR Global

Status: VERIFIED INTELLIGENCE / NOT YET ARCHITECTURE

## Goal

Identify official NVIDIA and Nebius capabilities that can materially improve ATHAR Global while keeping the product simple:

`procurement dossier → findings → exact evidence → human validation`

## Hackathon constraints

FACT
- The project must run on Nebius Token Factory or Nebius AI Cloud.
- The project must use at least one NVIDIA open-source model.
- Best Apps & Agents is the natural current track candidate for ATHAR Global.
- A public repository, working demo, README/setup instructions, open-source license, and a public demo video under 3 minutes are required.
- Token Factory runtime inference calls are sufficient to satisfy the Nebius runtime requirement; Serverless Endpoints/Jobs are encouraged but not required.

SOURCE
- https://nebiusglobalaihackathon.devpost.com/rules
- https://nebiusglobalaihackathon.devpost.com/
- https://nebiusglobalaihackathon.devpost.com/updates/46205-how-to-build-a-winning-project

## NVIDIA capabilities relevant to ATHAR

### 1. Nemotron Parse 2.0

FACT
- NVIDIA provides Nemotron Parse 2.0 for document parsing.
- The model is intended for extracting text and metadata from document images, including structured content.

ATHAR VALUE
- Recover document structure before evidence retrieval.
- Preserve sections, tables, and layout cues that matter in procurement dossiers.

STATUS
- HIGH-PRIORITY CANDIDATE.
- Must be benchmarked against simpler PDF extraction before adoption.

SOURCE
- https://build.nvidia.com/nvidia/nemotron-parse
- https://build.nvidia.com/models?q=nemotron

### 2. Nemotron OCR v2

FACT
- NVIDIA provides Nemotron OCR v2 for multilingual OCR on complex real-world images.
- It targets text recognition, layout, and table extraction.

ATHAR VALUE
- Scanned procurement documents.
- Poor-quality pages, photographed annexes, stamps, and image-only attachments.

STATUS
- HIGH-PRIORITY CANDIDATE for scanned documents only.
- Do not route born-digital PDFs through OCR unless needed.

SOURCE
- https://build.nvidia.com/nvidia/nemotron-ocr-v2

### 3. Nemotron Embeddings

FACT
- NVIDIA provides Nemotron embedding models for semantic retrieval and RAG.
- NVIDIA also provides multimodal embedding models that represent document images for retrieval.

ATHAR VALUE
- Retrieve candidate evidence passages from large procurement dossiers.
- Support multilingual and multimodal retrieval experiments.

STATUS
- HIGH-PRIORITY CANDIDATE.

SOURCE
- https://build.nvidia.com/explore/retrieval
- https://build.nvidia.com/models?q=nemotron

### 4. Nemotron Rerank

FACT
- NVIDIA provides reranking models designed to score whether a passage contains information relevant to a query.
- A multimodal reranker is available for image/document retrieval workflows.

ATHAR VALUE
- Improve precision after initial retrieval.
- Reduce the number of irrelevant passages shown to the human reviewer.

STATUS
- HIGH-PRIORITY CANDIDATE.
- Must prove measurable retrieval gain over embedding-only retrieval.

SOURCE
- https://build.nvidia.com/models?q=rerank
- https://build.nvidia.com/explore/retrieval

### 5. Nemotron reasoning models

FACT
- The hackathon explicitly encourages Nemotron reasoning models on Nebius Token Factory.
- Best Apps & Agents guidance suggests using stronger models for hard reasoning and smaller/faster models for routine calls.

ATHAR VALUE
- Compare rule, observed fact, and retrieved evidence.
- Produce a constrained evidence-first explanation.
- Identify insufficient evidence instead of forcing a conclusion.

STATUS
- REQUIRED CANDIDATE for the reasoning layer.
- Exact model tier is not selected yet.

SOURCE
- https://nebiusglobalaihackathon.devpost.com/rules
- https://nebiusglobalaihackathon.devpost.com/

### 6. NVIDIA RAG / NeMo Retriever blueprint

FACT
- NVIDIA publishes a RAG blueprint combining Nemotron reasoning, embeddings, reranking, OCR, and document-structure models.

ATHAR VALUE
- Reference architecture for evidence retrieval.
- Useful source for integration patterns, not a reason to copy the whole stack.

STATUS
- REFERENCE ONLY until benchmarked.

SOURCE
- https://build.nvidia.com/nvidia/build-a-rag-pipeline

## Nebius capabilities relevant to ATHAR

### 1. Token Factory

FACT
- Token Factory exposes an OpenAI-compatible inference API.
- It includes chat/completions, embeddings, rerank, responses, file APIs, and model listing.

ATHAR VALUE
- Primary hackathon inference runtime.
- Lets ATHAR call NVIDIA reasoning/retrieval models without operating model-serving infrastructure.

STATUS
- REQUIRED RUNTIME CANDIDATE.

SOURCE
- https://api.tokenfactory.nebius.com/docs

### 2. Serverless Endpoints

FACT
- Nebius provides serverless endpoints for deployed inference/services.

ATHAR VALUE
- Possible hosting path for production-like demo APIs.

STATUS
- OPTIONAL for MVP.
- Use only if it simplifies the working demo.

SOURCE
- https://nebius.com/blog/posts/introducing-serverless

### 3. Serverless Jobs

FACT
- Nebius provides serverless Jobs for background/asynchronous workloads.

ATHAR VALUE
- Possible fit for long document ingestion or batch evaluation.

STATUS
- OPTIONAL.
- Do not introduce unless document processing latency requires it.

SOURCE
- https://nebius.com/blog/posts/introducing-serverless

## Current evidence-backed stack hypothesis

This is still a HYPOTHESIS, not a decision:

```
PDF / scan
  ↓
Parse OR OCR only when needed
  ↓
Embed
  ↓
Rerank
  ↓
Nemotron reasoning
  ↓
finding + exact evidence
  ↓
human validation
```

## Simplification rule

Use the maximum NVIDIA capability that creates measurable value, not the maximum number of models.

Every component must survive this test:

1. Does it improve evidence quality, speed, or robustness?
2. Can the improvement be measured?
3. Can its role be explained in the 3-minute hackathon demo?
4. Does it keep the user experience simple?

If not, remove it.

## NEXT

Validate one brick only:

**Evidence retrieval pipeline**

Test:
`procurement pages → embed → rerank → top evidence passage`

Do not build UI or reasoning workflow until retrieval earns a PASS.
