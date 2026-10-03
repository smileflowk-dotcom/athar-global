# Nebius Tools — ATHAR Global

Purpose: runtime map for the Nebius components used by ATHAR Global.

## Primary runtime

### Token Factory

Use for:
- chat / reasoning
- embeddings
- reranking
- model discovery
- file APIs when useful

Official API:
- POST /v1/chat/completions
- POST /v1/embeddings
- POST /v1/rerank
- POST /v1/responses
- GET /v1/models

Source:
- https://api.tokenfactory.nebius.com/docs

## Optional runtime components

### Serverless Endpoints
Use only if ATHAR needs a deployed inference/service endpoint for the demo.

### Serverless Jobs
Use only if long document processing or benchmark execution benefits from background jobs.

## Rule

Token Factory is the default.
Do not add Nebius infrastructure unless it makes the working vertical slice simpler or more robust.
