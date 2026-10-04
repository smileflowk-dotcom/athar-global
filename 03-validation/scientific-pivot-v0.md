# Scientific Workflow Pivot V0

## TEST

Attempt to validate the smallest faithful MNIST comparison for the dropout claim using an NVIDIA GPU runtime.

## EXPECTED

Run identical no-dropout and dropout networks with only dropout changed, record raw outputs, and compare the observed generalization metrics to one paper claim.

## ACTUAL

The current environment has no NVIDIA GPU runtime, no PyTorch installation, and no NVIDIA or Nebius API key available locally. Existing repository Actions are CPU/API workflows and do not provide an NVIDIA GPU path.

## RESULT: BLOCKED

No paper claim, experiment metric, or NVIDIA run result was fabricated. The experiment was not run.

## NEXT

Provide a verified NVIDIA GPU execution path already supported by the repository or environment, then execute this single fixed-seed comparison.
