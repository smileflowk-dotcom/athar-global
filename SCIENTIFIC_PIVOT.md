# Scientific Workflow Pivot — Hackathon

## One-line product

Turn a research paper into a small, reproducible, evidence-linked experiment.

## Hackathon demo

Paper → Claim → Experiment → NVIDIA Run → Compare → Evidence

## Demo case

Paper: *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*

Goal: reproduce one simple claim with a small MNIST experiment.

## Product rule

Minimum input → maximum usable output → minimum time.

No business model, billing, auth, marketplace, collaboration, or broad scientific platform during the hackathon.

## Success criteria

1. The system identifies one testable claim from the paper.
2. It links the claim to exact evidence in the paper.
3. It produces a minimal faithful experiment.
4. The experiment runs on NVIDIA-backed compute or uses NVIDIA runtime capability in the end-to-end path.
5. It compares paper claim vs reproduced result.
6. It produces a compact evidence trace:
   - claim
   - source/page
   - experiment
   - run result
   - comparison
   - conclusion

## UX target

Six visible blocks maximum:

Paper → Claim → Experiment → NVIDIA Run → Compare → Evidence

## Stop rule

Do not add features unless they improve the 3-minute hackathon demo.
