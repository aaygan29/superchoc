# Pre-registered analysis plan: structure-only prediction of odor-mixture pleasantness

Committed to git before any structure-model prediction was compared with the Ma et al. (2021)
mixture ratings. The commit hash and timestamp of this file are the registration. Any change
after that commit is listed under "Deviations" at the bottom, with its reason.

## What was already seen before registration (disclosure)

- During data collection, an agent computed how well the *measured* component ratings
  (not structure predictions) explain mixture pleasantness in Ma 2021: an intensity²-weighted
  mean of measured component pleasantness gave r = 0.934, RMSE 0.395; a plain mean gave
  RMSE 0.450 (`data/hedonics/exploratory/`). This is the oracle ceiling (A3 below), not a test
  of the structure model.
- The triangle-test model (`DISCRIMINATION.md`) was run on Bushdid 2014: perceptual distance
  did not beat component overlap. It is not part of this plan's hypotheses.
- No structure-model prediction for any Ma molecule or mixture has been computed.

## Question

Can the pleasantness of odor mixtures that no model has seen be predicted from molecular
structure alone, i.e. without any human rating of the mixture or of its components?

## Data

- Training: Keller et al. 2016 single-molecule pleasantness (pyrfume `keller_2016`), the
  dilution used by `methods/real_flavor/real_data.py` (1/1000), averaged over subjects.
  **Every molecule that appears in Ma 2021 is removed from training** (match on PubChem CID
  and on canonical stereo-stripped SMILES).
- Test: Ma et al. 2021 (pyrfume `ma_2021`, doi 10.15454/51OVY6): 198 unique binary mixtures
  from 72 food odorants, trained panel, 0-10 scale, panel means
  (`data/hedonics/mixture_pleasantness_ma2021.csv`). Duplicate pairs are averaged.
  Molecules without a valid structure are excluded and counted.

## Models (fixed in advance)

Component predictor f(molecule) -> pleasantness, trained on Keller only. Four families, so
the ensemble is diverse by construction:
1. Random forest on ECFP4 (2048 bits, radius 2), 500 trees.
2. Random forest on RDKit 2D descriptors (all `Descriptors.descList`, NaNs -> column median
   of training set), 500 trees.
3. Gradient boosting (sklearn `HistGradientBoostingRegressor`, defaults) on RDKit 2D
   descriptors.
4. k-nearest neighbours (k = 10, Tanimoto on ECFP4, distance-weighted).
Ensemble = mean of the four, each first converted to Keller-rank scale (rank-normalised to
[0, 1] over Keller training predictions). Seeds fixed at 0. No hyperparameter tuning on Ma.

Mixture rule (primary): predicted mixture pleasantness = equal-weight mean of the two
predicted component values. Rationale: Ma presented components at roughly matched
intensity, so equal weights are the a-priori version of the intensity-weighted rule.

## Hypotheses and tests

All correlations are Spearman. Confidence intervals are 95% **molecule-cluster bootstrap**
(resample the 72 molecules with replacement, keep mixtures whose both components were drawn,
10,000 resamples), because mixtures share components.

- **H1 (components, external).** Ensemble predictions correlate positively with Ma's
  measured single-component pleasantness (per molecule, averaged across all mixtures in which
  it was rated alone). Supported if the bootstrap CI lower bound > 0.
- **H2 (mixtures, primary).** Ensemble mixture predictions correlate positively with Ma
  mixture pleasantness (n ≈ 198). Supported if CI lower bound > 0 **and** the permutation
  test (shuffle Keller labels, retrain, 1,000 permutations) gives p < 0.05.
- **H3 (classification with abstention).** "Liked" = Ma mixture pleasantness above the median
  of the 198 mixtures (fixed now). Score = ensemble mixture prediction. Report AUROC with CI.
  Abstention: a mixture is "confident" if the four model families agree on the side of the
  Keller-median for both components (ensemble disagreement as the uncertainty signal),
  otherwise "abstain". Supported if AUROC on the confident subset > AUROC on the abstained
  subset and confident-subset AUROC CI lower bound > 0.5.

Holm correction across H1-H3 at α = 0.05. All three results are reported whatever they show.

## Pre-specified secondary analyses (descriptive, no claims of significance)

- A1 Per-family results (each of the four models alone) vs ensemble.
- A2 Baselines: molecular weight alone; number of heavy atoms; a constant. If a baseline
  matches the ensemble within CI, that is stated as the finding.
- A3 Oracle ceiling: the same mixture rule using Ma's *measured* component ratings, to show
  how much of the gap is component prediction vs the mixture rule.
- A4 Intensity-weighted rule using measured component intensities in the mixture, with
  predicted component pleasantness.
- A5 Applicability domain: H2 restricted to mixtures whose components both have max ECFP4
  Tanimoto to Keller training ≥ 0.4, vs < 0.4.
- A6 Scale check: Keller (untrained raters, 0-100) vs Ma (trained panel, 0-10) differ;
  only rank statistics are used for claims for that reason.

## What would count as failure

H2 not supported = structure-only prediction of mixture pleasantness does not generalise to
this independent panel. That would be reported as the main result, and the candidate-design
stage would not claim predictive power for liking.

## Deviations

(none yet)

### Added 2026-11-14, after the analysis (the plan above is unchanged)

- **What the registration covers.** This file was committed at `61a9465` and was on GitHub
  from 01:29:49 UTC (GitHub activity log: creation of branch `chocolate-anchor` at a
  descendant commit), before any analysis output existed. The analysis *code*
  (`mixture_hedonics.py`) was written after that and is not part of the registration. The
  plan fixes the data, models, hypotheses and tests; the code's implementation choices are
  listed as interpretations below.
- **Interpretations of the plan.** Eleven are listed, with reasons, in
  `RESULTS_MIXTURE_HEDONICS.md` ("Deviations / interpretations"). The one that changes a
  verdict is H3: the plan's two criteria (confident AUROC > abstained AUROC; confident-subset
  CI lower bound > 0.5) are both met, so **under the registered text H3 is supported**. The
  analysis code added a p-value for Holm correction, defined after registration as the
  larger of two bootstrap probabilities (p = 0.42), under which H3 is not supported. Both
  readings are reported; the gap the criteria rest on is 0.02 (95% CI -0.20 to 0.26), so no
  claim is made that disagreement-based abstention helps.
