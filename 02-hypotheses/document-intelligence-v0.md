# HYPOTHESIS — Document Intelligence V0

## HYPOTHESIS

For the born-digital EuroHPC procurement PDF, simple page-preserving PDF extraction is sufficient for the retrieval demo; NVIDIA Parse/OCR should not be added unless it improves measured fidelity or structure.

## RATIONALE

The existing retrieval gold set uses exact page locators and exact source passages. The smallest useful test is to compare those passages and section labels against direct extraction before introducing another model or service.

## RISK

This conclusion is limited to the sampled born-digital PDF. Scanned, image-only, or table-heavy documents may still require Nemotron Parse or OCR.

## TEST

Extract the ten existing gold cases with `pypdf`, preserve page numbers, and measure exact quote anchors, section labels, and source-text grounding.
