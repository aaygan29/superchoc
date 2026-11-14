# Can people tell two odor mixtures apart?

`discrimination.py` predicts the fraction of people who pick the odd mixture in a
triangle test (chance = 1/3), fitted and tested on Bushdid et al. 2014 (264 triangles,
26 people each, mixtures of 1/10/20/30 of 128 odorants). Full numbers:
`results/discrimination.json`. Rerun: `python discrimination.py`.

## What it predicts
p_correct = 1/3 + 2/3 * sigmoid(a + b.x), fitted by binomial maximum likelihood.
Features: share of components in common (and mixture size), Snitz/Ravia structural angle,
and the angle between summed Leffingwell odor-descriptor vectors. API: `fit_default()`,
`predict_p_correct(a_smiles, b_smiles, weights_a=None, weights_b=None)`.
Default model: all four features.

## Data check
The data reading was verified (code fails loudly otherwise): one odd + two identical vials
per triangle, "Components that differ" equals the computed set difference.
Leffingwell labels 122 of 128 molecules; 190 of 264 triangles contain an unlabelled
molecule (it is skipped in the sum). No triangle was dropped.

## Results (log-loss per trial, lower is better; constant-rate model 0.688)
10-fold CV, whole triangles held out, 3 seeds, all 264 triangles:

| model | log-loss | Spearman | gain over overlap baseline (95% CI) |
|---|---|---|---|
| overlap + size (baseline) | 0.6640 | 0.59 | - |
| structural angle | 0.6713 | 0.52 | -0.0073 (-0.0144, -0.0002) baseline better |
| perceptual angle | 0.6657 | 0.57 | -0.0017 (-0.0051, +0.0020) tie |
| all four | 0.6629 | 0.61 | +0.0011 (-0.0010, +0.0034) tie |

Calibration of the combined model is good (5 bins: predicted 0.41-0.72, observed within 0.03).

Molecule hold-out (5 folds over molecules; test = triangles whose differing components are
>50% held-out): only **34 triangles** qualify (30 of them differ in one component), so this
is weak evidence. Perceptual 0.6455 vs baseline 0.6531 (gain +0.0077, CI -0.0007 to +0.0178);
structural clearly worse; combined equals baseline. Training on everything outside the test
set is a choice: the strict version (no held-out molecule among training differing
components) leaves 39-54 mostly one-change triangles and cannot learn the dose-response.

**Does perceptual distance beat overlap? No.** On whole-triangle CV it ties or loses; on the
small molecule hold-out it is nominally better but the interval includes zero. Everything
beats a constant by only about 3-7% of log-loss. The signal is overlap and number of
components; descriptors add nothing we can detect here.

Consistency of the descriptor angle with human similarity ratings: Spearman -0.51
(Snitz, 349 pairs) and -0.45 (Ravia, 195 pairs); structural angle -0.51 and -0.25.
Negative is the expected sign.

## Limits
- Humans disagree more than a binomial allows (Pearson dispersion about 2.3); parameter
  uncertainty from the likelihood alone is too small. Triangle-level bootstrap CIs are used.
- Triangle sets are a small, designed grid (few distinct differing-set sizes), and the same
  128 molecules recur, so triangles are not independent draws of molecules.
- Bushdid's per-vial dilutions vary (0.25/0.5/1); we do not model them.
- Weighted predictions are an extrapolation. Bushdid's components were roughly
  intensity-matched, so weights (e.g. concentration-weighted cocoa) were never tested. Weights
  scale the structural centroid and descriptor sum but not the set-based overlap features.
- Components outside Leffingwell's labels raise unless `allow_missing_descriptors=True`, which
  changes the distance.
- One study, fixed concentrations: says nothing about pleasantness, taste or retronasal flavor.
