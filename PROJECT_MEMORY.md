# PROJECT_MEMORY

Purpose: lightweight operational memory for ATHAR Global / Scientific Workflow Pivot.

This file stores only durable project memory:
- decisions
- verified facts
- failed paths
- current state
- next action

Keep it concise. Update only after meaningful verified changes.

## DECISIONS

- Hackathon-first execution. Business model and go-to-market are deferred until after submission.
- Pivot from public-procurement review to a scientific workflow product for the Nebius × NVIDIA hackathon.
- Core demo flow:
  Paper → Claim → Experiment → NVIDIA/Nebius reasoning → Compare → Evidence
- Demo case:
  *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* + MNIST.
- Keep scope minimal:
  no auth, billing, marketplace, collaboration, broad scientific agent, or multi-paper support before submission.
- Product rule:
  minimum input → maximum usable output → minimum time.
- Method:
  READ → PLAN → ACT → VERIFY → LEARN → COMMIT → NEXT.
- One brick at a time. Do not advance on unverified assumptions.

## VERIFIED FACTS

- Repo: smileflowk-dotcom/athar-global
- Scientific pivot branch: hackathon/scientific-workflow-pivot
- Existing repo skills:
  - skills/benchmark.md
  - skills/document-intelligence.md
  - skills/evidence-reasoning.md
  - skills/evidence-retrieval.md
- Existing repo tool docs:
  - tools/nvidia/README.md
  - tools/nebius/README.md
- GitHub Actions has working access to:
  - NVIDIA_API_KEY
  - NEBIUS_API_KEY
- Nebius Token Factory + NVIDIA Nemotron has already worked successfully in GitHub Actions.
- Verified model used in PROVE:
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
- Previous PROVE workflow succeeded end-to-end on main:
  Document Intelligence: PASS
  Evidence Reasoning: PASS
  End-to-End proof: PASS
- Evidence Retrieval V0 validated:
  Recall@5 = 1.00
  Recall@3 = 1.00
  Top-1 = 0.80
  Locator preservation = 1.00
  Fabricated evidence = 0

## FAILED / AVOIDED PATHS

- Do not treat Codex local environment missing secrets as proof GitHub secrets are unavailable.
- Do not require local GPU access in Codex.
- Do not require Docker.
- Do not force NVIDIA GPU execution if the hackathon requirements are already satisfied through real NVIDIA model usage on Nebius Token Factory.
- Do not silently substitute unavailable capabilities and still claim NVIDIA/Nebius execution.
- Do not expand back into the original procurement-specific ATHAR product during the hackathon.

## CURRENT STATE

Scientific pivot is active.

Current experiment:
- Paper: Dropout
- Dataset: MNIST
- Comparison:
  same architecture / same seed / same training settings
  only controlled difference = dropout
- Runtime path:
  GitHub Actions → PyTorch experiment → Nebius Token Factory → NVIDIA Nemotron conclusion
- Script:
  scripts/scientific_dropout_v0.py
- Workflow:
  .github/workflows/scientific-dropout-v0.yml
- Expected outputs:
  03-validation/scientific-dropout-v0.json
  03-validation/scientific-dropout-v0.md

## CURRENT ROADMAP

1. PROVE
   Validate one scientific vertical slice.
2. CONNECT
   Connect the proven steps into one reliable flow.
3. VISUAL DEMO
   Build the smallest judge-friendly UI.
4. SUBMIT
   README, architecture, demo video <3 min, Devpost, screenshots, NVIDIA/Nebius feedback.
5. POST-HACKATHON
   Market, customers, pricing, business model.

## CURRENT NEXT

Inspect the latest Scientific Dropout V0 GitHub Actions run.

If PASS:
- freeze the backend vertical slice;
- move directly to VISUAL DEMO.

If FAIL:
- make one minimal documented correction;
- rerun once;
- do not change scope.
