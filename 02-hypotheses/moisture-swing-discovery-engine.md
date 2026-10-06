# Moisture-Swing Discovery Engine — Current Hypothesis

## Working hypothesis

The best moisture-swing sorbent will emerge from interactions among material architecture, functional chemistry, counter-ion, water response, and operating conditions; it cannot be discovered reliably by counter-ion screening alone.

## Current implementation hypothesis

Use public experimental data to learn the response surface and uncertainty first, then reserve atomistic calculations for shortlisted, information-rich candidates.

## Immediate validation target

Reproduce the public MSA-ML baseline using its own dataset and protocol, then verify:

1. grouped test performance versus naive baselines
2. leave-one-resin-out generalization
3. predictive uncertainty behavior
4. active-learning selection behavior
5. whether the model can recover known strong moisture-swing conditions

Only after these pass should the candidate space be expanded and ALCHEMI used again for targeted screening.

## Current scientific status

No new material candidate is validated.

V1-V6 are retained as infrastructure/method-learning evidence, not as final discovery evidence.
