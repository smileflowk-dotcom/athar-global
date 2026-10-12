# Publication and intellectual-property safety — ATHAR Global

Status: public operational guardrails; not a patent opinion or a claim of confidentiality.

## Scope and current reality
- This repository, including its branches, draft PRs, and published workflow definitions, is **public**.
- The repository's existing `LICENSE` grants MIT permissions to the covered published software and associated documentation. This policy does **not** revoke or retroactively narrow those permissions.
- Published details and previous public commits, forks, logs, and artifacts must be assumed disclosed. Moving a branch, deleting a file or changing visibility is not guaranteed to undo that disclosure.
- Chemistry Discovery V4 is a research/computational **screening** workflow, not independently demonstrated industrial CO2 capture or proof of newly discovered materials.
- `lab20` or other candidate dossiers, if shared publicly, remain hypotheses and proposed experimental tests, not validated materials.

## What is acceptable in public
- Existing openly licensed code, general research problem and study design, limitations and known references.
- Reproducibility and gate definitions that have already been disclosed and explicitly approved for publication.
- Explicitly labeled summary-level scientific claims that are backed by accessible evidence, do not rely on third-party data without permission, and have been reviewed.

## What MUST NOT be added to this public repository without written release approval
1. New potentially patentable formulations, unannounced resin/ion/process combinations, privileged analyses, optimization recipes, or commercially sensitive comparisons.
2. Unpublished new atomistic predictions, candidate rankings, nonpublic laboratory outcomes or proprietary raw data.
3. Private agreements, negotiation terms, personal contact lists, correspondence, or lab/supplier confidential input.
4. API keys, tokens, passwords, credentials and secrets; do not rely only on `.gitignore`.
5. Third-party datasets, spreadsheets, publications or model checkpoints unless redistribution and intended commercial rights have been verified.

**A new branch or draft pull request inside this repository is still PUBLIC. Never treat it as confidential storage.** Public GitHub Actions may also disclose logs and run artifacts; do not run confidential candidate discovery in a public repository.

## Proposed two-track workflow (procedural, NOT access control)
**Public track:** methodology overview, already disclosed protocols, citations, uncontroversial code, research discussion.

**Confidential track:** new hypotheses, potentially protectable engineering improvements, novel datasets, unpublished simulations, lab feedback and prospective partners. Work in a *separately created and verified PRIVATE repository or other access-controlled system*; give access only to named collaborators. No secret files or private code should be committed to this public repository, even temporarily.

## Mandatory release checklist before any public post, preprint, PR or outreach attachment
- [ ] Identify authors, owners, co-inventors, institutional and collaboration obligations.
- [ ] Review novelty / prior art and determine whether patent advice or filing must precede disclosure.
- [ ] Confirm all included content is already public or formally cleared for release.
- [ ] Check source/data/model/figure licenses and permissions; specifically revisit `PYU-pub/MSA-ML` before redistributing the workbook or asserting commercial rights.
- [ ] Check candidate identity, charge/speciation, provenance, uncertainty and complete original A–E scientific gates.
- [ ] Separate measured experimental results from ML estimates, AIMNet2 proxies and independent GFN1 computational checks.
- [ ] Do not claim laboratory validation, new material discovery, novel phosphate counter-ions, industrial-scale performance or independent external ML validation without evidence.
- [ ] Obtain explicit final publication approval from the responsible project owner and scientific collaborators.

## Research status to state with external audiences
"ATHAR Global is developing an evidence-gated computational screening workflow to prioritize laboratory tests of moisture-responsive CO2 sorbents. Our models yield hypotheses; no new material is claimed experimentally validated by this pipeline."

## Next action
Create and VERIFY a separate private R&D workspace before recording additional sensitive results. Prepare drafts privately; seek patent/IP advice before any public disclosure of a potentially patentable invention. Do not auto-publish generated reports.

Updated: 2026-10-12.
