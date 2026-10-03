# Skill — Document Intelligence

## WHEN TO USE

Use when a procurement file must be converted into retrieval-ready content.

## INPUT

- PDF page
- scan
- image-only document
- table-heavy page

## NVIDIA CANDIDATES

- Nemotron Parse
- Nemotron OCR

## ROUTING HYPOTHESIS

- born-digital / structured document → Parse first
- image-only / degraded scan → OCR
- difficult pages → benchmark both if necessary

This routing is not yet validated.

## OUTPUT

- extracted text
- page locator
- layout / structure metadata where available
- table content where available

## TEST

Compare against a small gold transcription / structure set.

Measure:
- text accuracy
- table preservation
- locator preservation
- latency
- cost

## FAIL CONDITIONS

Do not adopt a NVIDIA parsing/OCR component if it performs worse than the simpler baseline on the target document class.
