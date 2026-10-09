# Moisture-Swing CO2 Discovery Engine — Project Brief

Status: R&D proving ground. Computational screening is active; no material is experimentally validated yet.

## 1. What this project is

A scientific discovery engine for identifying and prioritizing moisture-responsive CO2 sorbents based on ion-exchange resins.

Core workflow:

public experimental evidence -> ML screening -> uncertainty/OOD filtering -> diversity/feasibility gates -> NVIDIA ALCHEMI/AIMNet2 atomistic screening -> independent computational cross-check -> lab protocol -> experimental validation -> feedback loop.

The engine is designed to reduce the number of expensive wet-lab experiments needed to identify promising direct-air-capture sorbents.

## 2. Exact problem we solve

Direct Air Capture teams and materials labs face a large search space:

- many resin/backbone chemistries;
- many possible counter-ions;
- humidity strongly changes CO2 behavior;
- adsorption capacity alone is not enough;
- lab testing every combination is slow and expensive;
- literature results are fragmented and hard to compare;
- computational signals can be misleading if not validated across methods.

The concrete problem is:

> Which resin + counter-ion + humidity configuration should a lab test next to maximize the chance of finding a useful moisture-swing CO2 sorbent while minimizing wasted experiments?

Our product is not “a CO2 capture machine.” It is a decision engine that produces experimentally testable candidates and the evidence trail behind them.

## 3. Who this is for

### Primary users

1. Direct Air Capture R&D teams
   - Need: faster sorbent discovery and lower experimental search cost.
   - Buyer/value owner: Head of R&D, Materials Lead, CTO, Process Development Lead.

2. University and national-lab materials groups
   - Need: prioritize experiments, generate publishable hypotheses, compare sorbent families.
   - User: PI, postdoc, materials chemist, adsorption scientist.

3. Ion-exchange resin manufacturers
   - Need: discover new high-value applications for existing resin platforms.
   - Buyer/value owner: Innovation Director, R&D Director, Application Development.

4. Carbon-capture technology companies
   - Need: candidate materials compatible with their DAC/contacting/regeneration architecture.
   - Buyer/value owner: CTO, Sorbent Team, Process Engineering.

### Secondary users

- industrial decarbonization R&D groups;
- public research programs funding DAC;
- scientific discovery platforms seeking a validated materials workflow.

## 4. What value they receive

Instead of testing tens or hundreds of combinations blindly, the engine returns:

- ranked candidate list;
- predicted moisture-swing response;
- uncertainty and out-of-distribution risk;
- provenance/evidence;
- feasibility flags;
- atomistic support;
- independent cross-check status;
- control/reference candidate;
- next experiment that reduces uncertainty the most.

Commercially, the value proposition is:

> “We help your lab decide what to test next.”

Not:

> “We guarantee a new sorbent.”

## 5. Current strongest hypothesis

Current leading challenger:

IRA900 + P2O7^4- (pyrophosphate)

Control/reference:

IRA900 + CO3^2- (carbonate)

ALCHEMI/AIMNet2 robustness result:
- carbonate hydration shift: +0.23828125 eV
- P2O7 hydration shift: +0.21875 eV
- both positive
- robustness run status: PASS

Interpretation:
P2O7 survived the current atomistic robustness gate and is directionally competitive with carbonate in the local hydration proxy.

This is computational evidence only, not experimental validation.

Independent GFN2-xTB cross-check is being rerun after the first attempt encountered SCF non-convergence.

## 6. New discovery round

Scale Engine V2 screened:

- 20,000 virtual configurations
- 20 resins
- 10 counter-ions

New counter-ion hypotheses explored:
- B(OH)4^-
- C2O4^2-
- H2PO4^-
- HCO3^-
- SO3^2-
- citrate^3-

Current top new ML challenger:
- QMPR-2 + SO3^2-

Status:
hypothesis-screening only; not yet physically validated.

## 7. Dedicated scientific stack

### Evidence and data
- public literature
- PYU-pub/MSA-ML public dataset
- traceable resin/counter-ion metadata
- GitHub artifacts for reproducibility

### ML / ranking
- Python
- pandas / NumPy
- scikit-learn
- LightGBM
- grouped experimental validation
- uncertainty ensemble
- out-of-distribution scoring
- diversity-aware filtering
- staged 20K/10K -> 1K -> 100 -> 20 -> 5 funnel

### NVIDIA / atomistic
- NVIDIA nvalchemi-toolkit
- AIMNet2
- ASE geometry optimization
- hydration-state screening
- near/far CO2 interaction proxy

### Independent computational check
- GFN2-xTB via tblite
- RDKit for reduced molecular structures
- ASE optimizer

### Reasoning / audit
- NVIDIA Nemotron through Nebius Token Factory
- used for evidence/reasoning audit only
- never treated as physicochemical proof

### Orchestration / reproducibility
- GitHub Actions
- versioned scripts
- JSON/CSV artifacts
- explicit pass/hold gates
- no Docker requirement

### Market validation
- AgentMail
- dedicated inbox: kamal-co2@agentmail.to
- qualified lab / DAC / R&D contact list
- outreach measures interest in experimental validation

## 8. Repository role

Current proving-ground repository:
smileflowk-dotcom/athar-global

Relevant modules:
- scale/ — candidate generation, ML ranking, diversity and gates
- scripts/ — reproducible screening and atomistic runs
- 01-intelligence/moisture-swing/ — scientific protocol/evidence
- 02-hypotheses/ — candidate hypotheses
- 03-validation/ — computational validation artifacts
- skills/ — scientific screening rules
- .github/workflows/ — reproducible execution

Important:
ATHAR Global is still the proving ground. A dedicated standalone CO2 discovery repository should be created only when the independent computational gate and/or external lab interest justify promoting this work into its own product/research asset.

## 9. Scientific gates

1. Evidence hypothesis
2. ML screening
3. uncertainty/OOD
4. feasibility
5. ALCHEMI/AIMNet2 support
6. independent computational method
7. lab-ready protocol
8. experimental validation
9. feedback into the engine

Only step 8 can validate a material experimentally.

## 10. Business model hypotheses

Most realistic early offers:

### A. Paid discovery sprint
Input: target resin family / DAC process constraints.
Output: 3-10 ranked lab candidates + evidence + proposed experiments.

### B. Lab co-development
We provide computational prioritization; partner lab provides measurements; both iterate.

### C. R&D licensing / private deployment
For DAC or resin companies that want the engine applied to proprietary candidate spaces.

### D. Success-based / sponsored validation
Company funds a defined experimental validation campaign around selected candidates.

Near-term objective is not pricing optimization. It is proving:
1. labs want the shortlist;
2. they will review or test it;
3. computational ranking saves them real experimental effort.

## 11. Current customer-discovery signal

A dedicated CO2 outreach campaign has started through AgentMail.

Initial target groups:
- national laboratories
- university DAC/materials groups
- resin/materials companies
- carbon-capture R&D teams

Question being tested:
> Would you review or experimentally validate a small shortlist of moisture-responsive resin sorbents selected by our computational pipeline?

Positive replies, requests for data, or willingness to test are the first market validation milestones.

## 12. Immediate next gates

1. Finish the independent GFN2-xTB rerun for IRA900 + P2O7.
2. Chemistry-review QMPR-2 + SO3^2- before expensive atomistic work.
3. Continue targeted outreach and track replies.
4. If one strong candidate + control survives and a lab is interested, prepare the lab packet.
5. Then create the dedicated repository with the ICM structure.
