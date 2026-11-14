# Prospective test protocol (Stage 1 Registered Report draft)

**Question.** Does a structure-only model, frozen before any tasting, predict how pleasant a
trained panel finds new binary odor mixtures built from chocolate's odor-active compounds?

Everything the panel will test, and every prediction, is fixed in
`data/candidates/FROZEN_PREDICTIONS.csv`, SHA-256
`7c8eebdc41303cc12ce3a638de860a19e4ce91d2168cea1198d85644897139ab` (commit `0b6f081`).
The analysis code is fixed in `stage2_analysis.py`. Sections marked **[LAB]** are details
the running laboratory must fill in, and have approved, before Stage 1 submission; they
change procedure, never the stimuli, predictions, hypotheses or analysis.

## Background in one paragraph

On 198 binary mixtures rated by a trained panel (Ma et al. 2021), the frozen model's
structure-only predictions correlated ρ = 0.49 [0.27, 0.65] with panel pleasantness, in a
pre-registered test where every test molecule was absent from training
(`RESULTS_MIXTURE_HEDONICS.md`). This study asks whether that holds for molecules and
pairs the model and the earlier panel have never seen, drawn from cocoa's key odorants.

## Stimuli

- **40 test pairs**, each a cocoa key odorant (odor activity value ≥ 1 in a published cocoa
  or dark-chocolate analysis; `data/sensomics/`) with a second approved flavoring. Every
  molecule has a FEMA GRAS number or an EU Union List entry, none is on the FDA 2018
  delisting, and none appears in Ma 2021 or in the model's training data.
- Selection was stratified across the predicted range (8 pairs per prediction quintile of
  all 711 scored pairs) so the test covers predicted-unpleasant to predicted-pleasant, not
  only the model's favourites. No molecule appears more than 3 times.
- **4 anchor mixtures** from Ma 2021, chosen across Ma's observed pleasantness range. They
  link the new panel's scale to Ma's and are not part of any hypothesis.

## Procedure (replicating Ma et al. 2021)

- **Intensity matching.** Before the main study, each component is diluted so that its
  intensity matches a common reference, as Ma did with ethyl 2-methylbutyrate. **[LAB]**
  solvent, reference concentration, pilot panel size, acceptance criterion.
- **Mixing.** Components are mixed 1:1 at matched intensity, presented as vapor (Ma mixed
  vapors in the headspace over cotton, with no chemical contact between components).
  **[LAB]** vessel, volume, equilibration time.
- **Ratings.** Pleasantness of each mixture and of each component alone, plus intensity,
  on Ma's 0-10 visual analogue scale with Ma's intensity anchors (ethyl 2-methylbutyrate =
  3, linalool = 7).
- **Panel.** Trained assessors, n = 30 per stimulus (the design power is driven by the
  number of mixtures, not raters). **[LAB]** recruitment, training, session length, rest
  intervals, number of sessions.
- **Blinding and order.** Stimuli carry random three-digit codes generated from seed
  20261114. Assessors and the experimenter running sessions do not see predictions. Order
  is randomised per assessor.
- **Safety and ethics.** Sniffing only, no ingestion; concentrations at or below customary
  flavoring use; allergen and asthma screening; ethics approval before data collection.
  `methods/real_flavor/HANDOFF.md` lists the checks. **[LAB]** ethics reference.

## Hypotheses

The score for each pair is `pred_ensemble` from the frozen file. The observed value is the
panel-mean mixture pleasantness.

- **P1 (primary).** Spearman correlation between `pred_ensemble` and observed pleasantness
  over the 40 test pairs is greater than 0. One-sided test, α = 0.05, p-value from 10,000
  permutations of the observed values. Simulated power is about 0.95 if the true
  correlation matches the Ma result (ρ = 0.49) and much lower at ρ = 0.3 (see the power
  table in `EXPLORATORY.md`; the simulation assumes independent mixtures, so real power is
  likely somewhat lower).
- **P2 (secondary).** Of the 20 pairs the model called `disliked`, more than half fall below
  Ma's median pleasantness (4.97) after scale linking. One-sided exact binomial test against
  0.5, α = 0.05. The model made no confident `liked` call among the 40, so the liked class
  is not tested; this is a property of the frozen predictions, not a choice made later.
- **P3 (descriptive).** Correlation for each of the four model families separately.

P1 is the claim. P2 and P3 do not rescue a failed P1.

**Scale linking (for P2 only).** Fit observed = a + b × Ma-observed on the 4 anchors. If the
anchors' Spearman correlation with Ma is below 0.8, linking is declared unreliable and P2 is
reported as not evaluable. P1 is rank-based and needs no linking.

## Exclusions, fixed now

- An assessor is excluded if their ratings of the duplicated anchor presentations correlate
  below 0 across the 4 anchors (one duplicate per anchor). **[LAB]** confirm the duplicate
  presentations fit the session plan.
- A pair is excluded only if the intensity-matching pilot fails for one of its components
  (the component cannot be matched at a customary concentration). Excluded pairs are listed
  with reasons; P1 is then run on the remaining pairs, and if fewer than 30 remain the study
  is reported as underpowered.
- No other exclusions. No interim analysis; data collection stops at the planned panel size.

## What counts as each outcome

| Outcome | Reading |
|---|---|
| P1 significant | structure-only prediction of mixture pleasantness generalises to new molecules and to cocoa's odor-active chemistry |
| P1 not significant, observed ρ > 0 | consistent with a smaller effect than in Ma; inconclusive at this n |
| P1 observed ρ ≤ 0 | the model does not transfer to this chemistry; reported as the main finding |

Whatever the result, all ratings are released (CC BY) with the code, and any deviation from
this protocol is listed in the Stage 2 report.

## What this study cannot show

It tests odor mixtures sniffed at matched intensity, not tasted foods or full chocolate
recipes. Panel-mean pleasantness hides large differences between people (Arshamian et al.
2022). A positive P1 supports using the model to rank candidates for real food trials; it
does not show that any mixture is a "superchocolate".
