# Prospective test protocol (draft for a Stage 1 Registered Report)

**Question.** Does a structure-only model, fixed before any rating, predict how pleasant a
trained panel finds the smell of new binary odor mixtures built from chocolate's
odor-active compounds?

The stimuli and every prediction are fixed in two files, with SHA-256 values in
`data/candidates/FROZEN_PREDICTIONS.sha256`:

- `FROZEN_PREDICTIONS.csv` (40 test pairs plus 8 anchors): `b0d90357…e7f9dc`
- `FROZEN_COMPONENT_PREDICTIONS.csv` (the 48 test molecules): `833236be…6e024f9`

The analysis is fixed in `stage2_analysis.py`, which refuses to run if either file has
changed. Items marked **[LAB]** are details the running laboratory fills in, and has
reviewed, before submission. They change procedure, never stimuli, predictions,
hypotheses or analysis. No journal has reviewed this draft yet.

**Revision history.** v1 (frozen hash `7c8eebdc…`, commit `0b6f081`) was superseded
before any rating, after an adversarial review. Three changes were made. Damascenone was
dropped because its regulatory evidence was only indirect. The calibrated liked/disliked
calls were removed, because they were calibrated on different chemistry. The P2 test was
replaced, because it could not fail informatively. Nothing has been rated under either
version.

## Background

On 198 binary mixtures rated by a trained panel (Ma et al. 2021), structure-only
predictions correlated ρ = 0.49 [0.27, 0.65] with panel pleasantness. The analysis plan
for that test was fixed in advance, and every test molecule was absent from training.
This study asks whether that holds for molecules and pairs that neither the model nor that
panel has seen, drawn from cocoa's key odorants.

The model used here comes from the same code, retrained with the 48 test molecules also
removed (358 training molecules instead of 376). On the Ma mixtures it scores ρ = 0.46
[0.22, 0.62], and its predictions correlate 0.96 with the registered model's
(`ROBUSTNESS.md`).

## Stimuli

- **40 test pairs.** Each pairs a cocoa key odorant with a second flavoring. A cocoa key
  odorant has an odor activity value of at least 1 in a published cocoa or dark-chocolate
  analysis (`data/sensomics/`). Every molecule meets all of these:
  - it has a FEMA GRAS number or an entry in the EU Union List as adopted in Reg.
    872/2012 (later amendments were not checked);
  - it is not on the FDA's 2018 delisting;
  - it does not appear in Ma 2021 or in the model's training data.
- **How pairs were chosen.** All 693 eligible pairs were scored and split into five
  quintiles by prediction, and 8 pairs were drawn from each. The test therefore spans
  predicted-unpleasant to predicted-pleasant, not just the model's favourites. No molecule
  is used more than 3 times.
- **8 anchor mixtures** from Ma 2021, taken across Ma's observed range. They are a
  descriptive check that this panel's scale behaves like Ma's, and are in no hypothesis.

## Procedure (following Ma et al. 2021)

- **Intensity matching.** Before the main study, each component is diluted to match the
  intensity of a common reference, as Ma did. Ma's pilot used ethyl 2-methylbutyrate as the
  reference. **[LAB]** solvent, reference concentration, pilot panel, acceptance criterion.
- **Presentation.** Components are combined 1:1 at matched intensity as vapor; Ma mixed
  vapors over cotton without chemical contact. **[LAB]** vessel, volume, equilibration.
- **Ratings.** Each assessor rates the pleasantness and intensity of each mixture, and of
  each component alone, on the scale Ma used. **[LAB]** take the scale, its anchors and the
  instructions from Ma et al. 2021's methods, and cite them. This repo does not
  independently verify those details.
- **Panel.** Trained assessors, with at least 30 ratings per stimulus. Ma's data has 60
  assessor IDs and at least 30 per mixture; the dataset title describes a panel of 30.
  **[LAB]** recruitment, training, sessions, rest intervals.
- **Blinding.** Each stimulus carries a random three-digit code generated with seed
  20261114. Neither assessors nor session staff see the predictions. Presentation order is
  randomised for each assessor.
- **Safety and ethics.** Smelling only, no ingestion. Concentrations at or below customary
  flavoring use. Screen assessors for allergy and asthma. Obtain ethics approval before
  data collection (`methods/real_flavor/HANDOFF.md`). **[LAB]** ethics reference.

## Hypotheses

Observed values are panel means over all assessors who rated the stimulus.

- **P1 (primary).** Over the 40 test pairs, `pred_ensemble` correlates positively with
  observed mixture pleasantness (Spearman). P1 is supported only if both of these hold:
  - a one-sided permutation p-value below 0.05 (10,000 permutations); and
  - the one-sided 95% molecule-cluster bootstrap lower bound is above 0 (10,000 resamples).

  The two conditions together are needed because pairs share molecules. In simulation on
  these exact 40 pairs, the permutation test alone has a false-positive rate of 0.077, and
  the combined rule 0.019. The combined rule's power is 0.74 if the true correlation is
  0.49, and 0.31 if it is 0.3 (`ROBUSTNESS.md`).
- **P2 (secondary).** Over the 48 test molecules rated alone, the component-level
  `pred_ensemble` from `FROZEN_COMPONENT_PREDICTIONS.csv` correlates positively with
  observed pleasantness (Spearman). One-sided permutation p-value below 0.05 (10,000
  permutations). Each molecule is one observation, so there is no sharing problem.
- **Descriptive.** The P1 correlation for each of the four model families, and the anchor
  mixtures' correlation with Ma's ratings.

P1 is the claim. P2 and the descriptive results do not rescue a failed P1.

## Exclusions, fixed now

- No assessor exclusions. Every rating that is collected is used.
- A test pair is excluded only if one of its components cannot be intensity-matched at a
  customary concentration in the pilot, which happens before any mixture is rated. If more
  than 10 pairs are excluded this way, the study stops before the main panel and is
  reported as not run as designed.
- No interim analysis. Data collection stops at the planned panel size.

## Outcomes

| Outcome | Reading |
|---|---|
| P1 supported | structure-only prediction of odor-mixture pleasantness carries over to new molecules and to cocoa's odor-active chemistry |
| P1 not supported, observed ρ > 0 | inconclusive at this sample size |
| P1 observed ρ ≤ 0 | the model does not carry over to this chemistry; reported as the main finding |

Every rating is released with the code. Any deviation from this protocol is listed in the
Stage 2 report.

## What this study cannot show

It tests odor mixtures smelled at matched intensity. It does not test tasted foods,
retronasal flavor, or full chocolate recipes. A panel mean also hides large differences
between people (Arshamian et al. 2022). A supported P1 would justify using the model to
rank candidates for food trials. It would not show that any mixture is a "superchocolate".
