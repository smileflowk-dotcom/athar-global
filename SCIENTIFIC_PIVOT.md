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

## Current validation state

RESULT: BLOCKED

The requested MNIST comparison was not executed. This environment has no `NVIDIA_API_KEY`, no `NEBIUS_API_KEY`, no PyTorch installation, and no available `nvidia-smi` GPU runtime. The existing repository Actions workflows run on `ubuntu-latest` and call hosted APIs; they do not provide an NVIDIA GPU execution path.

No paper claim, experiment metric, or NVIDIA run result is recorded until a verified NVIDIA GPU runtime is available.

NEXT: provide an approved NVIDIA GPU execution path already supported by the repository or environment, then run the smallest fixed-seed MNIST dropout comparison.
